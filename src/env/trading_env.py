# ENV_VERSION = "1.0.0"

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Realistic + PPO-Friendly Trading Environment (Hybrid Reward v5)
    
    Key additions:
        ✔ leverage as argument
        ✔ leverage applied to equity dynamics
        ✔ leverage applied to directional signal
        ✔ hybrid reward remains stable with leverage
        ✔ PPO-friendly clipping
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

        self.current_step = self.window_size

        # Actions: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)

        # Features (exclude timestamp)
        # Preferred consistent ordering
        preferred_order = [
            "open", "high", "low", "close", "volume",
            "log_return",
            "EMA_10", "EMA_50",
            "RSI_14",
            "volatility_20",
            "ATR_14",
        ]

        # Normalized versions if they exist
        norm_cols = [c for c in df.columns if c.endswith("_norm")]

        # Fallback for any extra engineered features
        other_cols = [
            c for c in df.columns
            if c not in preferred_order and c not in norm_cols and c != "timestamp"
        ]

        self.feature_cols = preferred_order + norm_cols + other_cols
        
        if "ATR_14" not in self.feature_cols:
            raise ValueError("ATR_14 missing from dataframe. Run feature pipeline again.")
        
        self.n_features = len(self.feature_cols)

        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(window_size, self.n_features),
            dtype=np.float32,
        )

        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.episode_step = 0

        self.equity = float(self.initial_balance)
        self.position = 0
        self.entry_price = 0.0

        self.total_trades = 0
        self.winning_trades = 0

        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        self.last_step_return = 0.0
        self.returns_history = []

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        self.current_step += 1
        self.episode_step += 1

        # -----------------------------
        # 1) Episode termination flags
        # -----------------------------
        terminated_data_end = self.current_step >= len(self.df) - 1
        truncated_horizon = self.episode_step >= self.max_episode_steps

        terminated = terminated_data_end
        truncated = truncated_horizon and not terminated_data_end

        prev_price = self.df.loc[self.current_step - 1, "close"]
        curr_price = self.df.loc[self.current_step, "close"]

        prev_equity = self.equity
        prev_position = self.position

        # -----------------------------
        # 2) ACTION LOGIC
        # -----------------------------
        if action == 1:
            target_position = 1    # Long
        elif action == 2:
            target_position = -1   # Short
        else:
            target_position = prev_position  # Hold

        trade_executed = (target_position != prev_position)

        # -----------------------------
        # 3) FEES
        # -----------------------------
        if trade_executed:
            fee = self.transaction_cost * prev_equity
            self.equity -= fee
            self.total_trades += 1
        else:
            fee = 0.0

        # -----------------------------
        # 4) POSITION UPDATE
        # -----------------------------
        self.position = target_position
        if trade_executed:
            self.entry_price = curr_price

        # -----------------------------
        # 5) EQUITY DYNAMICS
        #    (price move + leverage)
        # -----------------------------
        price_return = (curr_price - prev_price) / prev_price
        effective_return = self.position * price_return * self.leverage

        # Update equity with leveraged return
        self.equity *= (1 + effective_return)

        # Log-return on EQUITY (PPO-friendly)
        equity_ratio = self.equity / max(prev_equity, 1e-8)
        log_ret = float(np.log(equity_ratio))

        # We keep the field name for compatibility, but now it stores log-return
        self.last_step_return = log_ret
        self.returns_history.append(log_ret)

        if trade_executed and log_ret > 0:
            self.winning_trades += 1

        # -----------------------------
        # 6) DRAWDOWN TRACKING
        # -----------------------------
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.current_drawdown = (
            self.peak_equity - self.equity
        ) / max(self.peak_equity, 1e-8)

        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # -----------------------------
        # 7) UNREALIZED PNL (for hold bonus)
        # -----------------------------
        if self.position != 0 and self.entry_price is not None:
            unrealized_pnl = (curr_price - self.entry_price) * self.position * self.leverage
            unrealized_pnl_pct = unrealized_pnl / max(self.initial_balance, 1e-8)
        else:
            unrealized_pnl = 0.0
            unrealized_pnl_pct = 0.0

        # -----------------------------
        # 8) REWARD DESIGN (log-return based)
        # -----------------------------
        # Base term: scaled equity log-return
        base_reward = 150.0 * log_ret 

        # Drawdown penalty: higher drawdown → stronger negative reward
        dd_penalty = 0.25 * self.current_drawdown  

        # Turnover penalty: discourage overtrading
        turnover_penalty = 0.01 if trade_executed else 0.0

        # Small bonus for holding profitable positions (unrealized PnL > 0)
        hold_bonus = 0.02 * unrealized_pnl_pct if unrealized_pnl > 0 else 0.0

        reward = (
            base_reward
            - dd_penalty
            - turnover_penalty
            + hold_bonus
        )

        # Final clipping for PPO stability
        reward = float(np.clip(reward, -1.0, 1.0))

        # -----------------------------
        # 9) BUILD OUTPUTS
        # -----------------------------
        obs = self._get_observation()
        info = self._get_info()

        info.update({
            "leverage": self.leverage,
            "price_return": float(price_return),
            "effective_return": float(effective_return),
            "equity_log_return": log_ret,
            "trade_executed": trade_executed,
            "unrealized_pnl": float(unrealized_pnl),
            "reward_breakdown": {
                "base_log_return": base_reward,
                "dd_penalty": dd_penalty,
                "turnover_penalty": turnover_penalty,
                "hold_bonus": hold_bonus,
            },
        })

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _get_observation(self):
        win = self.df.iloc[self.current_step - self.window_size : self.current_step]
        return win[self.feature_cols].values.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _get_info(self):
        sharpe = 0.0
        if len(self.returns_history) >= 20:
            arr = np.array(self.returns_history[-20:])
            if arr.std() > 1e-8:
                sharpe = float(arr.mean() / arr.std())

        return {
            "equity": float(self.equity),
            "balance": float(self.equity),
            "position": int(self.position),
            "entry_price": float(self.entry_price),
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "win_rate": float(self.winning_trades / max(1, self.total_trades)),
            "max_drawdown": float(self.max_drawdown),
            "current_drawdown": float(self.current_drawdown),
            "roi": float((self.equity - self.initial_balance) / self.initial_balance * 100.0),
            "last_step_return": float(self.last_step_return),
            "sharpe_ratio": float(sharpe),
        }