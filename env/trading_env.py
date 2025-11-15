import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Trading Environment for Reinforcement Learning
    
    1. Proper Gym API compliance (gymnasium not gym)
    2. Better reward function (profit + Sharpe + drawdown)
    3. Transaction costs
    4. Position tracking
    5. Comprehensive info dict for debugging
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        window_size: int = 10,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001  # 0.1% per trade
        ):
        
        super().__init__()    
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.current_step = self.window_size
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: (window_size, num_features)
        # Assuming features are all columns except timestamp
        n_features = len(df.columns) - 1  # exclude timestamp
        
        self.observation_space = spaces.Box(
            low=-10.0,      # Normalized features typically in [-3, 3]
            high=10.0,      # Using [-10, 10] for safety margin
            shape=(window_size, n_features),
            dtype=np.float32
        )
        
        # For Sharpe ratio calculation (track returns)
        self.returns_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset variables
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        
        # Initialize tracking variables
        self.total_trades = 0
        self.winning_trades = 0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0
        self.returns_history = []
        
        obs = self._get_observation()
        info = {}
        
        return obs, info  # ✅ Returns (observation, info)  
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Returns: (observation, reward, terminated, truncated, info)
        
        - Gymnasium separates 'done' into:
          * terminated: Episode naturally ended (hit goal/fail)
          * truncated: Hit max steps (timeout)
        """
        
        # Move forward in time
        self.current_step += 1
        
        # Check if episode is over
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # Could add max_steps limit
        
        # Get current price
        current_price = self.df.loc[self.current_step, "close"]
        
        # Calculate reward BEFORE taking action
        reward = 0.0
        trade_executed = False
        
        # ============================================
        # ACTION EXECUTION
        # ============================================
        
        if action == 1:  # BUY
            if self.position == 0:  # Enter long
                self.position = 1
                self.entry_price = current_price
                # Transaction cost
                cost = self.transaction_cost * self.balance
                self.balance -= cost
                trade_executed = True
                self.total_trades += 1
                
            elif self.position == -1:  # Close short, go long
                # Calculate short profit
                profit = self.entry_price - current_price
                self.balance += profit
                if profit > 0:
                    self.winning_trades += 1
                
                # Transaction cost for closing + opening
                cost = 2 * self.transaction_cost * self.balance
                self.balance -= cost
                
                # Open long
                self.position = 1
                self.entry_price = current_price
                trade_executed = True
                self.total_trades += 2
                
                reward = profit - cost
        
        elif action == 2:  # SELL
            if self.position == 0:  # Enter short
                self.position = -1
                self.entry_price = current_price
                cost = self.transaction_cost * self.balance
                self.balance -= cost
                trade_executed = True
                self.total_trades += 1
                
            elif self.position == 1:  # Close long, go short
                # Calculate long profit
                profit = current_price - self.entry_price
                self.balance += profit
                if profit > 0:
                    self.winning_trades += 1
                
                # Transaction costs
                cost = 2 * self.transaction_cost * self.balance
                self.balance -= cost
                
                # Open short
                self.position = -1
                self.entry_price = current_price
                trade_executed = True
                self.total_trades += 2
                
                reward = profit - cost
        
        # action == 0 (HOLD) - no action taken
        
        # ============================================
        # UNREALIZED P&L (for open positions)
        # ============================================
        
        unrealized_pnl = 0
        if self.position == 1:  # Long position
            unrealized_pnl = current_price - self.entry_price
        elif self.position == -1:  # Short position
            unrealized_pnl = self.entry_price - current_price
        
        # Add unrealized P&L to reward (encourage holding winners)
        reward += unrealized_pnl * 0.01  # Small weight to not dominate
        
        # ============================================
        # RISK-ADJUSTED METRICS
        # ============================================
        
        # Track returns for Sharpe ratio
        if len(self.returns_history) > 0:
            period_return = (self.balance - self.peak_balance) / self.peak_balance
            self.returns_history.append(period_return)
            
            # Update peak and drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            else:
                drawdown = (self.peak_balance - self.balance) / self.peak_balance
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
                    # Penalty for drawdown
                    reward -= drawdown * 10  # Penalize risk
        
        # ============================================
        # SHARPE RATIO COMPONENT (if enough data)
        # ============================================
        
        if len(self.returns_history) >= 20:
            returns_array = np.array(self.returns_history[-20:])
            if returns_array.std() > 0:
                sharpe = returns_array.mean() / returns_array.std()
                reward += sharpe * 0.1  # Small Sharpe bonus
        
        # ============================================
        # PREPARE OUTPUT
        # ============================================
        
        obs = self._get_observation()
        info = self._get_info()
        info['trade_executed'] = trade_executed
        info['action'] = action
        info['reward_breakdown'] = {
            'realized_pnl': reward if trade_executed else 0,
            'unrealized_pnl': unrealized_pnl,
            'drawdown_penalty': -self.max_drawdown * 10
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get the observation window.
        
        Returns: Array of shape (window_size, n_features)
        """
        # Get last window_size candles
        window = self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ]
        
        # Drop timestamp column
        obs = window.drop(columns=["timestamp"]).values
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """
        Get detailed info for logging/debugging.
        """
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "max_drawdown": self.max_drawdown,
            "roi": ((self.balance - self.initial_balance) / self.initial_balance) * 100
        }
        return info