import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.risk.risk_engine import RiskEngine, RiskConfig


class TradingEnv(gym.Env):
    """
    PPO-friendly trading environment with:
    - Leverage
    - Risk Engine integration
    - ATR_pct_norm volatility scaling
    - Hybrid reward (log-return based)
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

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.max_episode_steps = max_episode_steps

        # PPO Discrete actions
        # 0 = HOLD, 1 = LONG, 2 = SHORT
        self.action_space = spaces.Discrete(3)

        # Risk engine
        self.risk_engine = RiskEngine(RiskConfig())

        # Use ATR percentage normalized as volatility measure
        self.atr_col = "ATR_pct_norm"
        if self.atr_col not in df.columns:
            raise ValueError("ATR_pct_norm missing — update feature pipeline.")

        # Feature ordering
        preferred = [
            "open", "high", "low", "close", "volume",
            "log_return",
            "EMA_10_norm", "EMA_50_norm",
            "RSI_14_norm",
            "volatility_20_norm",
            "ATR_pct_norm",
        ]

        # Add any extra engineered features
        extra = [c for c in df.columns if c not in preferred and c not in ["timestamp"]]

        self.feature_cols = preferred + extra
        self.n_features = len(self.feature_cols)

        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(window_size, self.n_features),
            dtype=np.float32,
        )

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

        self.position = 0             # -1 short, 1 long, 0 flat
        self.entry_price = None
        self.current_position = None

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.returns_history = []
        self.last_step_return = 0.0

        # Drawdown tracking
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Reset risk engine
        self.risk_engine.reset(self.initial_balance)

        return self._get_observation(), self._get_info()

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action: int):

        # ---- 1. Time update ----
        self.current_step += 1
        self.episode_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = self.episode_step >= self.max_episode_steps

        prev_price = float(self.df.loc[self.current_step - 1, "close"])
        curr_price = float(self.df.loc[self.current_step, "close"])
        prev_equity = self.equity

        # ---- 2. Map PPO action -> direction ----
        direction = 1 if action == 1 else -1 if action == 2 else 0

        # ---- 3. Volatility input using ATR_pct_norm ----
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
            if self.current_position:
                self._close_position(curr_price)
                trade_executed = True

            reward = self._finalize_step(prev_equity, curr_price, trade_executed)
            obs = self._get_observation()
            info = self._get_info()
            info["kill_switch"] = True
            return obs, reward, True, False, info

        # ---- 6. Exit Logic ----
        if risk["should_exit"] and self.current_position:
            self._close_position(curr_price)
            trade_executed = True

        # ---- 7. Entry Logic ----
        if risk["should_enter"]:
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

    def _open_position(self, pos):
        fee = self.transaction_cost * self.balance
        self.balance -= fee

        self.current_position = pos.copy()
        self.position = pos["direction"]
        self.entry_price = pos["entry_price"]

        self.total_trades += 1

    def _close_position(self, price):
        d = self.current_position["direction"]
        entry = self.current_position["entry_price"]
        size = self.current_position["size"]

        realized = size * (price - entry) * d
        self.balance += realized

        if realized > 0:
            self.winning_trades += 1

        self.position = 0
        self.entry_price = None
        self.current_position = None

    def _update_equity(self, price):
        if self.current_position:
            d = self.current_position["direction"]
            entry = self.current_position["entry_price"]
            size = self.current_position["size"]
            unrealized = size * (price - entry) * d
        else:
            unrealized = 0.0

        self.equity = self.balance + unrealized

    def _compute_reward(self, prev_equity):
        # log-return reward
        ratio = self.equity / max(prev_equity, 1e-8)
        log_ret = float(np.log(ratio))
        self.last_step_return = log_ret
        self.returns_history.append(log_ret)

        # no turnover penalty anymore
        reward = 150.0 * log_ret

        return float(np.clip(reward, -1.0, 1.0))

    # ============================================================
    # OBS & INFO
    # ============================================================

    def _get_observation(self):
        win = self.df.iloc[self.current_step - self.window_size : self.current_step]
        return win[self.feature_cols].values.astype(np.float32)

    def _get_info(self):
        sharpe = 0.0
        if len(self.returns_history) >= 20:
            arr = np.array(self.returns_history[-20:])
            if arr.std() > 1e-8:
                sharpe = float(arr.mean() / arr.std())

        return {
            "equity": float(self.equity),
            "balance": float(self.balance),
            "position": int(self.position),
            "entry_price": float(self.entry_price) if self.entry_price else 0.0,
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "win_rate": float(self.winning_trades / max(self.total_trades, 1)),
            "max_drawdown": float(self.max_drawdown),
            "current_drawdown": float(self.current_drawdown),
            "roi": float((self.equity - self.initial_balance) / self.initial_balance * 100.0),
            "last_step_return": float(self.last_step_return),
            "sharpe_ratio": sharpe,
        }