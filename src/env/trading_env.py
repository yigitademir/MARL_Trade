import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.risk.risk_engine import RiskEngine, RiskConfig


class TradingEnv(gym.Env):
    """
    PPO-friendly trading environment with:

    - Leverage & transaction costs
    - RiskEngine integration (kill-switch + ATR-based SL/TP)
    - ATR_14_norm etc. as state features
    - Reward based on log equity returns

    Actions (Discrete):
        0 = HOLD
        1 = LONG
        2 = SHORT
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0004,
        leverage: float = 5.0,
        max_episode_steps: int = 2000,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Core data and hyperparameters
        # ------------------------------------------------------------------
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.leverage = float(leverage)
        self.max_episode_steps = int(max_episode_steps)

        # ------------------------------------------------------------------
        # ATR columns: raw ATR for risk, normalized ATR for state
        # ------------------------------------------------------------------
        self.atr_col = "ATR_14"
        if self.atr_col not in self.df.columns:
            raise ValueError("Column 'ATR_14' missing. Update feature pipeline.")

        self.atr_norm_col = "ATR_14_norm"
        if self.atr_norm_col not in self.df.columns:
            raise ValueError("Column 'ATR_14_norm' missing. Update feature pipeline.")

        # ------------------------------------------------------------------
        # Action & observation spaces
        # ------------------------------------------------------------------
        self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: LONG, 2: SHORT

        preferred = [
            "open", "high", "low", "close", "volume",
            "log_return",
            "EMA_10_norm", "EMA_50_norm",
            "RSI_14_norm",
            "volatility_20_norm",
            "ATR_14_norm",   # normalized ATR for PPO input
        ]

        # Include any extra engineered features
        extra = [
            c
            for c in self.df.columns
            if c not in preferred and c not in ["timestamp"]
        ]

        self.feature_cols = preferred + extra
        self.n_features = len(self.feature_cols)

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.window_size, self.n_features),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Risk Engine
        # ------------------------------------------------------------------
        self.risk_engine = RiskEngine(RiskConfig())

        # Initialize episode state
        self.reset()

    # ============================================================
    # RESET
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.episode_step = 0

        self.equity = float(self.initial_balance)
        self.balance = float(self.initial_balance)

        self.position = 0            # -1 short, 1 long, 0 flat
        self.entry_price = None
        self.current_position: dict | None = None

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.returns_history: list[float] = []
        self.last_step_return = 0.0

        # Drawdown tracking
        self.peak_equity = float(self.initial_balance)
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Reset risk engine
        self.risk_engine.reset(self.initial_balance)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action: int):
        # ---- 1. Time update ----
        self.current_step += 1
        self.episode_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = self.episode_step >= self.max_episode_steps

        prev_equity = float(self.equity)
        curr_price = float(self.df.loc[self.current_step, "close"])

        # Keep position's current_price in sync with market
        if self.current_position is not None:
            self.current_position["current_price"] = curr_price

        # ---- 2. Map PPO action -> direction ----
        # 0 -> HOLD, 1 -> LONG (+1), 2 -> SHORT (-1)
        if action == 1:
            direction = 1
        elif action == 2:
            direction = -1
        else:
            direction = 0

        # ---- 3. ATR_14 for risk engine (SL/TP) ----
        atr_val = float(self.df.loc[self.current_step, self.atr_col])

        # ---- 4. Risk Engine ----
        risk = self.risk_engine.process_signal(
            direction=direction,
            price=curr_price,
            equity=self.balance,
            atr=atr_val,
            position=self.current_position,
        )

        trade_executed = False

        # ---- 5. Kill-switch ----
        if risk["kill_switch"]:
            if self.current_position is not None:
                self._close_position(curr_price)
                trade_executed = True

            reward = self._finalize_step(prev_equity)
            obs = self._get_observation()
            info = self._get_info()
            info["kill_switch"] = True
            info["trade_executed"] = trade_executed
            return obs, reward, True, False, info

        # ---- 6. Exit Logic ----
        if risk["should_exit"] and self.current_position is not None:
            self._close_position(curr_price)
            trade_executed = True

        # ---- 7. Entry Logic ----
        if risk["should_enter"] and risk["new_position"] is not None:
            self._open_position(risk["new_position"])
            trade_executed = True

        # ---- 8. Mark-to-market equity ----
        self._update_equity(curr_price)

        # ---- 9. Drawdown tracking ----
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.current_drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-8)
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # ---- 10. Reward ----
        reward = self._compute_reward(prev_equity)

        # ---- 11. Output ----
        obs = self._get_observation()
        info = self._get_info()
        info["trade_executed"] = trade_executed

        return obs, reward, terminated, truncated, info

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _open_position(self, pos: dict):
        """
        Open a new position:

        - Charge transaction fee on trade notional (size * price)
        - Update balance & position state
        """

        entry_price = float(pos["entry_price"])
        size = float(pos["size"])

        notional = size * entry_price
        fee = self.transaction_cost * notional
        self.balance -= fee

        self.current_position = pos.copy()
        self.position = int(pos["direction"])
        self.entry_price = entry_price

        self.total_trades += 1

    def _close_position(self, price: float):
        """
        Close the current position:

        - Realize PnL
        - Charge transaction fee on closing notional
        """

        if self.current_position is None:
            return

        d = int(self.current_position["direction"])
        entry = float(self.current_position["entry_price"])
        size = float(self.current_position["size"])
        price = float(price)

        notional_close = size * price

        # Realized PnL on the position
        realized = size * (price - entry) * d

        # Fee only on closing side (open side already charged)
        fee_close = self.transaction_cost * notional_close

        self.balance += realized
        self.balance -= fee_close

        if realized > 0:
            self.winning_trades += 1

        self.position = 0
        self.entry_price = None
        self.current_position = None

    def _update_equity(self, price: float):
        """
        Update equity = balance + unrealized PnL.
        """

        price = float(price)

        if self.current_position is not None:
            d = int(self.current_position["direction"])
            entry = float(self.current_position["entry_price"])
            size = float(self.current_position["size"])
            unrealized = size * (price - entry) * d
        else:
            unrealized = 0.0

        self.equity = self.balance + unrealized

    def _compute_reward(self, prev_equity: float) -> float:
        """
        Reward based on log equity return between steps.
        """

        prev_equity = max(float(prev_equity), 1e-8)
        ratio = self.equity / prev_equity
        log_ret = float(np.log(ratio))

        self.last_step_return = log_ret
        self.returns_history.append(log_ret)

        reward = 150.0 * log_ret
        return float(np.clip(reward, -1.0, 1.0))

    # ============================================================
    # OBSERVATION & INFO
    # ============================================================

    def _get_observation(self):
        """
        Return a rolling window of features.
        Shape: (window_size, n_features)
        """

        start = self.current_step - self.window_size
        end = self.current_step
        win = self.df.iloc[start:end]
        obs = win[self.feature_cols].values.astype(np.float32)
        return obs

    def _get_info(self):
        """
        Diagnostics for monitoring & logging.
        """

        sharpe = 0.0
        if len(self.returns_history) >= 20:
            arr = np.array(self.returns_history[-20:])
            std = arr.std()
            if std > 1e-8:
                sharpe = float(arr.mean() / std)

        win_rate = float(self.winning_trades / max(self.total_trades, 1))

        return {
            "equity": float(self.equity),
            "balance": float(self.balance),
            "position": int(self.position),
            "entry_price": float(self.entry_price) if self.entry_price is not None else 0.0,
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "win_rate": win_rate,
            "max_drawdown": float(self.max_drawdown),
            "current_drawdown": float(self.current_drawdown),
            "roi": float((self.equity - self.initial_balance) / self.initial_balance * 100.0),
            "last_step_return": float(self.last_step_return),
            "sharpe_ratio": sharpe,
        }

    def _finalize_step(self, prev_equity: float) -> float:
        """
        Helper for kill-switch step (same as normal reward).
        """
        return self._compute_reward(prev_equity)