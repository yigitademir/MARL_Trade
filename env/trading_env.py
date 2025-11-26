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
        
    Hybrid Reward (Option C + leverage):
        effective_return = leverage * position * price_return

        reward = 0.7 * (step_return * 100)
               + 0.3 * (effective_return * 100)
               - 0.05 * trade_executed

        reward clipped to [-1, 1]
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

        terminated_data_end = self.current_step >= len(self.df) - 1
        truncated_horizon = self.episode_step >= self.max_episode_steps

        terminated = terminated_data_end
        truncated = truncated_horizon and not terminated_data_end

        prev_price = self.df.loc[self.current_step - 1, "close"]
        curr_price = self.df.loc[self.current_step, "close"]

        prev_equity = self.equity
        prev_position = self.position

        # --- ACTION LOGIC ---
        if action == 1:
            target_position = 1
        elif action == 2:
            target_position = -1
        else:
            target_position = prev_position

        trade_executed = (target_position != prev_position)

        # --- FEES ---
        if trade_executed:
            fee = self.transaction_cost * prev_equity
            self.equity -= fee
            self.total_trades += 1
        else:
            fee = 0.0

        # --- POSITION UPDATE ---
        self.position = target_position
        if trade_executed:
            self.entry_price = curr_price

        # --- RETURN / EQUITY DYNAMICS ---
        price_return = (curr_price - prev_price) / prev_price
        effective_return = self.position * price_return * self.leverage

        self.equity *= (1 + effective_return)

        step_return = (self.equity - prev_equity) / max(prev_equity, 1e-8)
        self.last_step_return = float(step_return)
        self.returns_history.append(step_return)

        if trade_executed and step_return > 0:
            self.winning_trades += 1

        # --- DRAWDOWN TRACKING ---
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.current_drawdown = \
            (self.peak_equity - self.equity) / max(self.peak_equity, 1e-8)

        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # ------------------------------------------------------------
        # HYBRID REWARD FUNCTION (with leverage amplification)
        # ------------------------------------------------------------
        profit_component = step_return * 100.0
        directional_component = effective_return * 100.0
        trade_penalty = -0.05 if trade_executed else 0.0

        reward = (
            0.7 * profit_component +
            0.3 * directional_component +
            trade_penalty
        )

        reward = float(np.clip(reward, -1.0, 1.0))
        # ------------------------------------------------------------

        obs = self._get_observation()
        info = self._get_info()

        info.update({
            "leverage": self.leverage,
            "price_return": float(price_return),
            "effective_return": float(effective_return),
            "step_return": float(step_return),
            "trade_executed": trade_executed,
            "reward_breakdown": {
                "profit_component": profit_component,
                "directional_component": directional_component,
                "trade_penalty": trade_penalty,
            }
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