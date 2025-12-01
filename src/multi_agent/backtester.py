# multi_agent/backtester.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# Action coordinator (majority vote / weighted / priority)
from src.multi_agent.coordinator import MultiAgentCoordinator

# Risk layer (SL/TP/position sizing/max DD kill-switch)
from src.risk.risk_engine import RiskEngine, RiskConfig


class MultiAgentBacktesterV1:
    """
    Multi-timeframe PPO backtester.

    Versions:
    ---------
    v1: Simple demo (only prints actions)
    v2: Trading loop (PPO-only, no risk engine)
    v3: Trading loop with external RiskEngine (this file)
    """

    def __init__(
        self,
        models: dict,
        data: dict,
        window_size: int = 10,
        strategy: str = "majority_vote",
    ) -> None:
        """
        Parameters
        ----------
        models : dict
            Timeframe -> PPO model mapping.
        data : dict
            Timeframe -> feature dataframe mapping.
        window_size : int
            Observation window length (number of candles).
        strategy : str
            Coordination strategy ("majority_vote", "priority_order", ...).
        """

        self.models = models
        self.data = data
        self.window_size = window_size

        # Multi-agent fusion logic
        self.coordinator = MultiAgentCoordinator(strategy=strategy)

        # Risk engine v1 (handles SL/TP, sizing, kill-switch)
        self.risk_engine = RiskEngine(RiskConfig())

        # Common timestamp alignment across all timeframes
        self.timestamps = self._align_timestamps()

        print(f"Aligned timestamps: {len(self.timestamps)} rows")

    # ------------------------------------------------------------------
    # TIMESTAMP SYNC
    # ------------------------------------------------------------------

    def _align_timestamps(self):
        """
        Compute intersection of timestamps across all timeframes.
        Ensures all agents operate on synchronized market times.
        """
        ts = None
        for tf, df in self.data.items():
            df_ts = set(df["timestamp"])
            ts = df_ts if ts is None else ts.intersection(df_ts)

        return sorted(list(ts))

    # ------------------------------------------------------------------
    # OBSERVATION BUILDER (SAFE / PADDED)
    # ------------------------------------------------------------------

    def _build_observation(self, df: pd.DataFrame, i: int) -> np.ndarray:
        """
        Build (window_size, features) observation for a given index.

        - Pads with the first row for early indices (< window_size)
        - Removes timestamp column
        - Always returns correct shape for PPO

        PPO requires a stable observation size, so we pre-pad.
        """

        feature_cols = [c for c in df.columns if c != "timestamp"]

        if i <= 0:
            row = df.iloc[0:1][feature_cols].values
            obs = np.repeat(row, self.window_size, axis=0)
            return obs.astype(np.float32)

        if i < self.window_size:
            window = df.iloc[:i][feature_cols]
            pad_len = self.window_size - len(window)
            pad_row = df.iloc[0:1][feature_cols].values
            pad_block = np.repeat(pad_row, pad_len, axis=0)

            full = pd.concat(
                [pd.DataFrame(pad_block, columns=feature_cols), window],
                ignore_index=True,
            )

        else:
            full = df.iloc[i - self.window_size : i][feature_cols]

        obs = full.values

        if obs.shape[0] != self.window_size:
            raise ValueError(f"Obs window wrong size {obs.shape[0]} vs {self.window_size}")

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # v1: SIMPLE ACTION PRINTING
    # ------------------------------------------------------------------

    def run(self, steps: int = 5):
        """
        Diagnostic function: print actions of all agents
        without executing trades.
        """
        print("\n=== Multi-Agent Backtester v1 ===\n")

        printed = 0
        for ts in self.timestamps:
            if printed >= steps:
                break

            actions = {}
            for tf, model in self.models.items():
                df = self.data[tf]
                idx_list = df.index[df["timestamp"] == ts].tolist()
                if not idx_list:
                    continue

                idx = idx_list[0]
                obs = self._build_observation(df, idx)
                action, _ = model.predict(obs, deterministic=True)
                actions[tf] = int(action)

            if not actions:
                continue

            final_action = self.coordinator.decide(actions)
            print(f"{ts} | actions={actions} -> final={final_action}")
            printed += 1

        print("\n=== Backtester v1 complete ===\n")

    # ------------------------------------------------------------------
    # v3: TRADING LOOP WITH RISK ENGINE
    # ------------------------------------------------------------------

    def run_trading(
        self,
        initial_balance=10000,
        leverage=5.0,
        fee_rate=0.0004,
        max_steps=None,
        base_timeframe="1h",
        verbose=True,
        atr_col="ATR_14",
    ):
        """
        Multi-agent trading simulation with integrated RiskEngine.

        Removes:
        - manual SL/TP
        - manual trailing stop
        - manual partials
        - manual sizing logic

        Adds:
        - structured risk handling (risk-per-trade, SL/TP generation)
        - position sizing from risk engine
        - max drawdown kill-switch
        - clean trading loop (agent → coordinator → risk engine → trade)
        """

        df_base = self.data[base_timeframe]

        balance = initial_balance
        position = 0      # Deprecated; replaced by current_position (dict)
        current_position = None
        entry_price = None
        position_size = 0.0

        # Monitoring buffers
        equity_curve = []
        trades = []
        per_agent_actions = {tf: [] for tf in self.models.keys()}

        # Reset risk engine internal state
        self.risk_engine.reset(initial_balance)

        if verbose:
            print("\n=== Multi-Agent Backtester ===")
            print(f"Initial balance: {initial_balance:,.2f}")

        # ------------------------------------------------------------
        # MAIN LOOP
        # ------------------------------------------------------------
        for step_i, ts in enumerate(self.timestamps):

            if max_steps and step_i >= max_steps:
                break

            row = df_base[df_base["timestamp"] == ts]
            if row.empty:
                continue

            price = float(row["close"].iloc[0])
            atr_value = row[atr_col].iloc[0]

            # --------------------------------------------------------
            # 1) Multi-agent action voting
            # --------------------------------------------------------
            actions = {}
            for tf, model in self.models.items():
                df = self.data[tf]

                idxs = df.index[df["timestamp"] == ts].tolist()
                if not idxs:
                    continue
                idx = idxs[0]

                # Skip until window history is long enough
                if idx < self.window_size:
                    continue

                obs = self._build_observation(df, idx)
                action, _ = model.predict(obs, deterministic=True)

                actions[tf] = int(action)
                per_agent_actions[tf].append({"timestamp": ts, "action": int(action)})

            # Fill in missing agents with HOLD
            for tf in self.models.keys():
                if tf not in actions:
                    per_agent_actions[tf].append({"timestamp": ts, "action": 0})

            if not actions:
                continue

            final_action = self.coordinator.decide(actions)

            # --------------------------------------------------------
            # 2) Risk Engine Decision (entry / exit / kill-switch)
            # --------------------------------------------------------
            risk_output = self.risk_engine.process_signal(
                direction=final_action,
                price=price,
                equity=balance,
                atr=atr_value,
                position=current_position,
            )

            # Kill-switch immediately stops the backtest
            if risk_output["kill_switch"]:
                if current_position is not None:
                    realized = (
                        current_position["size"]
                        * (price - current_position["entry_price"])
                        * current_position["direction"]
                    )
                    balance += realized

                    trades.append({
                        "timestamp": ts,
                        "price": price,
                        "action": "EXIT_KILL_SWITCH",
                        "realized_pnl": realized,
                        "balance_after": balance,
                    })

                current_position = None
                break

            # Exit logic from risk engine
            if risk_output["should_exit"] and current_position is not None:
                realized = (
                    current_position["size"]
                    * (price - current_position["entry_price"])
                    * current_position["direction"]
                )
                balance += realized

                trades.append({
                    "timestamp": ts,
                    "price": price,
                    "action": f"EXIT_{risk_output['exit_reason']}",
                    "realized_pnl": realized,
                    "balance_after": balance,
                })

                current_position = None

            # Entry logic (risk engine calculates SL/TP/size)
            if risk_output["should_enter"]:
                pos = risk_output["new_position"]

                # Entry fee
                balance -= balance * fee_rate

                current_position = pos.copy()
                current_position["notional_size"] = pos["size"]

                trades.append({
                    "timestamp": ts,
                    "price": price,
                    "action": "ENTRY",
                    "position": pos["direction"],
                    "size": pos["size"],
                    "stop_loss": pos["stop_loss"],
                    "take_profit": pos["take_profit"],
                    "balance_after": balance,
                })

            # --------------------------------------------------------
            # 3) Mark-to-market equity
            # --------------------------------------------------------
            if current_position is None:
                unrealized = 0.0
            else:
                d = current_position["direction"]
                entry = current_position["entry_price"]
                size = current_position["size"]
                unrealized = size * (price - entry) * d

            equity = balance + unrealized

            equity_curve.append([
                ts,
                price,
                equity,
                balance,
                0 if current_position is None else current_position["direction"],
                unrealized,
                risk_output["current_drawdown"],
            ])

        # ------------------------------------------------------------
        # Build outputs
        # ------------------------------------------------------------
        agent_action_dfs = {
            tf: pd.DataFrame(records)
            for tf, records in per_agent_actions.items()
        }

        equity_df = pd.DataFrame(
            equity_curve,
            columns=[
                "timestamp",
                "price",
                "equity",
                "balance",
                "position",
                "unrealized_pnl",
                "drawdown",
            ],
        )

        trade_df = pd.DataFrame(trades)

        # Summary statistics
        final_equity = float(equity_df["equity"].iloc[-1])
        roi = (final_equity - initial_balance) / initial_balance * 100
        max_dd = (
            (equity_df["equity"].cummax() - equity_df["equity"])
            / equity_df["equity"].cummax()
        ).max()

        summary = {
            "final_equity": final_equity,
            "roi_pct": roi,
            "max_drawdown_pct": float(max_dd * 100),
            "trades": int(len(trade_df)),
        }

        if verbose:
            print(f"Final equity  : {final_equity:,.2f}")
            print(f"ROI           : {roi:.2f}%")
            print(f"Max drawdown  : {max_dd*100:.2f}%")
            print(f"Trades        : {len(trade_df)}")
            print("------------------------------------------")

        return summary, equity_df, trade_df, agent_action_dfs