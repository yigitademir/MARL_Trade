# multi_agent/backtester.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.multi_agent.coordinator import MultiAgentCoordinator


class MultiAgentBacktesterV1:
    """
    Multi-timeframe PPO backtester.

    v1: simple action printing
    v2: trading loop with leverage + liquidation
    """

    def __init__(
        self,
        models: dict,
        data: dict,
        window_size: int = 10,
        strategy: str = "majority_vote",
    ) -> None:
        """
        Args:
            models: { "5m": PPO_model, "15m": PPO_model, ... }
            data:   { "5m": df,       "15m": df,       ... } (feature dfs)
            window_size: number of candles per observation
            strategy: coordination strategy for MultiAgentCoordinator
        """
        self.models = models
        self.data = data
        self.window_size = window_size

        self.coordinator = MultiAgentCoordinator(strategy=strategy)

        # Align timestamps across all dataframes (inner join)
        self.timestamps = self._align_timestamps()

        print(f"Aligned timestamps: {len(self.timestamps)} rows")

    # ------------------------------------------------------------------
    # TIMESTAMP SYNC
    # ------------------------------------------------------------------

    def _align_timestamps(self):
        """Return intersection of timestamps across all timeframes."""
        ts = None
        for tf, df in self.data.items():
            df_ts = set(df["timestamp"])
            ts = df_ts if ts is None else ts.intersection(df_ts)

        aligned = sorted(list(ts))
        return aligned

    # ------------------------------------------------------------------
    # OBSERVATION BUILDER (SAFE / PADDED)
    # ------------------------------------------------------------------

    def _build_observation(self, df: pd.DataFrame, i: int) -> np.ndarray:
        """
        Build (window_size, features) observation for a given timeframe.

        - If i < window_size, pad from the top using the first row.
        - Always returns shape (window_size, n_features) with no timestamp column.
        """
        feature_cols = [c for c in df.columns if c != "timestamp"]

        if i <= 0:
            # No history yet → just repeat first row
            row = df.iloc[0:1][feature_cols].values
            obs = np.repeat(row, self.window_size, axis=0)
            return obs.astype(np.float32)

        if i < self.window_size:
            # Use available part + pad the rest with first row
            window = df.iloc[:i][feature_cols]
            pad_len = self.window_size - len(window)

            pad_row = df.iloc[0:1][feature_cols].values
            pad_block = np.repeat(pad_row, pad_len, axis=0)

            full = pd.concat(
                [pd.DataFrame(pad_block, columns=feature_cols), window],
                ignore_index=True,
            )
        else:
            # Normal case: enough history
            full = df.iloc[i - self.window_size : i][feature_cols]

        obs = full.values
        # Safety check
        if obs.shape[0] != self.window_size:
            raise ValueError(
                f"Obs window has wrong length: got {obs.shape[0]}, "
                f"expected {self.window_size}"
            )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # v1: SIMPLE ACTION PRINTING (sanity check)
    # ------------------------------------------------------------------

    def run(self, steps: int = 5):
        """
        Run v1 backtester: only prints agent actions for a few steps.
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
                if len(idx_list) == 0:
                    continue

                idx = idx_list[0]
                obs = self._build_observation(df, idx)
                action, _ = model.predict(obs, deterministic=True)
                actions[tf] = int(action)

            if not actions:
                continue

            final_action = self.coordinator.decide(actions)
            print(f"{ts} | actions={actions} → final={final_action}")
            printed += 1

        print("\n=== Backtester v1 complete ===\n")

    # ------------------------------------------------------------------
    # v2: TRADING LOOP WITH LEVERAGE + LIQUIDATION
    # ------------------------------------------------------------------

    def run_trading(
        self,
        initial_balance=10_000,
        leverage=5.0,
        fee_rate=0.0004,
        atr_mult=2.0,
        rr_partial=2.5,
        max_steps=None,
        base_timeframe="1h",
        verbose=True):
        """
        Multi-agent trading backtester with:
        - ATR Chandelier Trailing Stop (A1)
        - 2.5 RR partial close (50%)
        - Realistic PnL
        - Fees
        - Multi-timeframe MARL decision-making
        """

        df_base = self.data[base_timeframe]

        balance = initial_balance
        position = 0           # 1 long, -1 short, 0 flat
        entry_price = None
        position_size = 0.0

        highest_price = None
        lowest_price = None
        trail_stop = None
        partial_taken = False

        equity_curve = []
        trades = []

        if verbose:
            print("\n=== Multi-Agent Backtester v3 (ATR Chandelier Risk Engine) ===")
            print(f"Initial balance: {initial_balance:,.2f}")

        # ------------------------------------------------------------
        # MAIN LOOP
        # ------------------------------------------------------------
        for step_i, ts in enumerate(self.timestamps):

            if max_steps and step_i >= max_steps:
                break

            # price for evaluation TF
            row = df_base[df_base["timestamp"] == ts]
            if row.empty:
                continue
            price = float(row["close"].iloc[0])
            atr = float(row["ATR_14"].iloc[0])

            # 1) Build multi-agent decision
            actions = {}
            for tf, model in self.models.items():
                df = self.data[tf]
                idxs = df.index[df["timestamp"] == ts].tolist()
                if not idxs:
                    continue
                idx = idxs[0]

                if idx < self.window_size:
                    continue

                obs = self._build_observation(df, idx)
                action, _ = model.predict(obs, deterministic=True)
                actions[tf] = int(action)

            if not actions:
                continue

            final_action = self.coordinator.decide(actions)

            prev_position = position
            prev_balance = balance
            prev_equity = balance

            # ------------------------------------------------------------
            # 2) Maintain ATR-based trailing stop
            # ------------------------------------------------------------
            if position == 1:  # long
                highest_price = price if highest_price is None else max(highest_price, price)
                trail_stop = highest_price - atr_mult * atr

            elif position == -1:  # short
                lowest_price = price if lowest_price is None else min(lowest_price, price)
                trail_stop = lowest_price + atr_mult * atr

            # ------------------------------------------------------------
            # 3) Check trailing stop hit BEFORE new signals
            # ------------------------------------------------------------
            stop_hit = False

            if position == 1 and price <= trail_stop:
                stop_hit = True
            if position == -1 and price >= trail_stop:
                stop_hit = True

            # ------------------------------------------------------------
            # 4) 2.5 RR partial close
            # ------------------------------------------------------------
            if position != 0 and not partial_taken:
                if position == 1:
                    reward_ratio = (price - entry_price) / (entry_price - (highest_price - atr_mult * atr))
                else:
                    reward_ratio = (entry_price - price) / ((lowest_price + atr_mult * atr) - entry_price)

                if reward_ratio >= rr_partial:
                    # Close HALF the position
                    realized = 0.5 * position_size * (price - entry_price) * position
                    balance += realized
                    position_size *= 0.5
                    partial_taken = True

                    trades.append({
                        "timestamp": ts,
                        "price": price,
                        "action": "PARTIAL",
                        "realized_pnl": realized,
                        "balance_after": balance
                    })

            # ------------------------------------------------------------
            # 5) Execute exit: either from stop-loss or agent signal
            # ------------------------------------------------------------
            exit_signal = (final_action != position)

            if stop_hit or exit_signal:
                if position != 0:
                    realized_pnl = position_size * (price - entry_price) * position
                    balance += realized_pnl
                else:
                    realized_pnl = 0

                trades.append({
                    "timestamp": ts,
                    "price": price,
                    "action": "EXIT_STOP" if stop_hit else "EXIT_SIGNAL",
                    "realized_pnl": realized_pnl,
                    "balance_after": balance
                })

                # Close position
                position = 0
                entry_price = None
                position_size = 0
                highest_price = None
                lowest_price = None
                trail_stop = None
                partial_taken = False

            # ------------------------------------------------------------
            # 6) Enter new position if signaled
            # ------------------------------------------------------------
            if final_action != 0 and position == 0:
                position = final_action
                entry_price = price
                balance -= balance * fee_rate  # entry fee
                position_size = (balance * leverage) / price

                highest_price = price
                lowest_price = price
                partial_taken = False

                trades.append({
                    "timestamp": ts,
                    "price": price,
                    "action": "ENTRY",
                    "position": position,
                    "balance_after": balance
                })

            # ------------------------------------------------------------
            # 7) EQUITY (mark-to-market)
            # ------------------------------------------------------------
            if position == 0:
                unrealized = 0
            else:
                unrealized = position_size * (price - entry_price) * position

            equity = balance + unrealized
            equity_curve.append([ts, price, equity, balance, position, unrealized])

        # ------------------------------------------------------------
        # Return DataFrames
        # ------------------------------------------------------------
        equity_df = pd.DataFrame(equity_curve, columns=[
            "timestamp", "price", "equity", "balance", "position", "unrealized_pnl"
        ])
        trade_df = pd.DataFrame(trades)

        # Stats
        final_equity = float(equity_df["equity"].iloc[-1])
        roi = (final_equity - initial_balance) / initial_balance * 100
        max_dd = (
            (equity_df["equity"].cummax() - equity_df["equity"]) /
            equity_df["equity"].cummax()
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
            print("==========================================")

        return summary, equity_df, trade_df
