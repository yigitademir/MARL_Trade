# ENV_VERSION = "1.0.0"

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.risk.risk_engine import RiskEngine, RiskConfig


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

        # Risk engine and risk-aware state
        self.risk_engine = RiskEngine(RiskConfig())
        self.atr_col = "ATR_14" 

        # Split equity into balance + unrealized PnL (backtester style)
        self.balance = float(initial_balance)
        self.current_position = None  # dict or None

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

        # Equity / balance split
        self.equity = float(self.initial_balance)
        self.balance = float(self.initial_balance)
        self.current_position = None

        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0

        self.total_trades = 0
        self.winning_trades = 0

        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        self.last_step_return = 0.0
        self.returns_history = []

        # Reset risk engine
        self.risk_engine.reset(self.initial_balance)

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        # --------------------------------
        # 0) Time / episode bookkeeping
        # --------------------------------
        self.current_step += 1
        self.episode_step += 1

        terminated_data_end = self.current_step >= len(self.df) - 1
        truncated_horizon = self.episode_step >= self.max_episode_steps

        terminated = terminated_data_end
        truncated = truncated_horizon and not terminated_data_end

        # If episode is over because of data, still build a last obs/info
        prev_price = self.df.loc[self.current_step - 1, "close"]
        curr_price = self.df.loc[self.current_step, "close"]

        prev_equity = self.equity

        # --------------------------------
        # 1) Map PPO action -> direction
        # --------------------------------
        # PPO action space: 0 = HOLD, 1 = LONG, 2 = SHORT
        if action == 1:
            desired_direction = 1
        elif action == 2:
            desired_direction = -1
        else:
            desired_direction = 0

        # --------------------------------
        # 2) Call RiskEngine
        # --------------------------------
        atr_value = float(self.df.loc[self.current_step, self.atr_col])

        risk_output = self.risk_engine.process_signal(
            direction=desired_direction,
            price=float(curr_price),
            equity=float(self.balance),   # realized equity for sizing
            atr=atr_value,
            position=self.current_position,
        )

        trade_executed = False
        unrealized_pnl = 0.0
        unrealized_pnl_pct = 0.0

        # --------------------------------
        # 3) Handle kill-switch
        # --------------------------------
        if risk_output["kill_switch"]:
            # If there is an open position, close it immediately
            if self.current_position is not None:
                d = self.current_position["direction"]
                entry = self.current_position["entry_price"]
                size = self.current_position["size"]
                realized = size * (float(curr_price) - entry) * d
                self.balance += realized
                trade_executed = True

            self.current_position = None
            self.position = 0
            self.entry_price = None

            # Update equity after closing
            self.equity = self.balance
            self.current_drawdown = risk_output["current_drawdown"]
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

            # Reward for kill-switch step (equity may have changed)
            equity_ratio = self.equity / max(prev_equity, 1e-8)
            log_ret = float(np.log(equity_ratio))
            self.last_step_return = log_ret
            self.returns_history.append(log_ret)

            # Simple reward using log-return (no extra shaping here)
            reward = float(np.clip(150.0 * log_ret, -1.0, 1.0))

            obs = self._get_observation()
            info = self._get_info()
            info.update({
                "equity_log_return": log_ret,
                "trade_executed": trade_executed,
                "kill_switch": True,
            })
            # Treat kill-switch as terminal event
            return obs, reward, True, False, info

        # --------------------------------
        # 4) Handle exits (SL / TP / signal flip)
        # --------------------------------
        if risk_output["should_exit"] and self.current_position is not None:
            d = self.current_position["direction"]
            entry = self.current_position["entry_price"]
            size = self.current_position["size"]

            realized = size * (float(curr_price) - entry) * d
            self.balance += realized
            trade_executed = True
            self.total_trades += 1
            if realized > 0:
                self.winning_trades += 1

            self.current_position = None
            self.position = 0
            self.entry_price = None

        # --------------------------------
        # 5) Handle new entries
        # --------------------------------
        if risk_output["should_enter"]:
            pos = risk_output["new_position"]

            # Apply transaction fee on equity
            fee = self.transaction_cost * self.balance
            self.balance -= fee

            self.current_position = pos.copy()
            self.current_position["notional_size"] = pos["size"]

            self.position = pos["direction"]
            self.entry_price = pos["entry_price"]

            trade_executed = True
            self.total_trades += 1  # win counter updated on exit

        # --------------------------------
        # 6) Mark-to-market equity
        # --------------------------------
        if self.current_position is None:
            unrealized_pnl = 0.0
        else:
            d = self.current_position["direction"]
            entry = self.current_position["entry_price"]
            size = self.current_position["size"]
            unrealized_pnl = size * (float(curr_price) - entry) * d

        self.equity = self.balance + unrealized_pnl
        self.unrealized_pnl = unrealized_pnl

        # Log-return on equity
        equity_ratio = self.equity / max(prev_equity, 1e-8)
        log_ret = float(np.log(equity_ratio))
        self.last_step_return = log_ret
        self.returns_history.append(log_ret)

        # --------------------------------
        # 7) Drawdown tracking
        # --------------------------------
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.current_drawdown = (
            self.peak_equity - self.equity
        ) / max(self.peak_equity, 1e-8)

        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # --------------------------------
        # 8) Reward design (same structure, now using risk-aware equity)
        # --------------------------------
        # Base term: scaled equity log-return
        base_reward = 150.0 * log_ret

        # Drawdown penalty
        dd_penalty = 0.25 * self.current_drawdown

        # Turnover penalty: discourage overtrading
        turnover_penalty = 0.01 if trade_executed else 0.0

        # Unrealized PnL stats for hold bonus
        if self.current_position is not None:
            unrealized_pnl_pct = unrealized_pnl / max(self.initial_balance, 1e-8)
        else:
            unrealized_pnl_pct = 0.0

        hold_bonus = 0.02 * unrealized_pnl_pct if unrealized_pnl > 0 else 0.0

        reward = (
            base_reward
            - dd_penalty
            - turnover_penalty
            + hold_bonus
        )

        reward = float(np.clip(reward, -1.0, 1.0))

        # --------------------------------
        # 9) Build outputs
        # --------------------------------
        obs = self._get_observation()
        info = self._get_info()

        info.update({
            "price_return": float((curr_price - prev_price) / prev_price),
            "equity_log_return": log_ret,
            "trade_executed": trade_executed,
            "unrealized_pnl": float(unrealized_pnl),
            "current_drawdown": float(self.current_drawdown),
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
        # Sharpe over last 20 log-returns
        sharpe = 0.0
        if len(self.returns_history) >= 20:
            arr = np.array(self.returns_history[-20:])
            std = arr.std()
            if std > 1e-8:
                sharpe = float(arr.mean() / std)

        # Safe entry_price
        if isinstance(self.entry_price, (int, float)):
            entry_price = float(self.entry_price)
        else:
            entry_price = 0.0

        # Safe unrealized pnl
        unrealized = float(self.unrealized_pnl) if hasattr(self, "unrealized_pnl") else 0.0

        return {
            "equity": float(self.equity),
            "balance": float(self.balance),    # FIXED: was self.equity
            "position": int(self.position),

            "entry_price": entry_price,
            "unrealized_pnl": unrealized,

            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "win_rate": float(self.winning_trades / max(1, self.total_trades)),

            "max_drawdown": float(self.max_drawdown),
            "current_drawdown": float(self.current_drawdown),

            "roi": float((self.equity - self.initial_balance) / self.initial_balance * 100.0),

            "last_step_return": float(self.last_step_return),
            "sharpe_ratio": float(sharpe),
        }