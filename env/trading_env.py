import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size: int = 10, initial_balance: float = 10000):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True) # ensure clean index
        self.window_size = window_size
        self.current_step = self.window_size
        self.initial_balance = initial_balance 

        # Define actions
        # 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Define observation space
        obs_shape = (self.window_size, self.df.shape[1] - 1) # minus timestamp
        self.observation_space = spaces.Box(
            low= np.inf,
            high= np.inf,
            shape= obs_shape,
            dtype= np.float32
        )

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._get_observation()
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = 0.0

        # Get current price
        current_price = self.df.iloc[self.current_step]["close"]

        # Execute action
        if action == 1: # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:
                reward = self.entry_price - current_price # short profit
                self.balance += reward
                self.position = 0
                self.entry_price = 0

        elif action == 2: # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:
                reward = current_price - self.entry_price # long profit
                self.balance += reward
                self.position = 0
                self.entry_price = 0

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position
        }
        return obs, reward, done, info
    
    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        obs = window.drop(columns=["timestamp"]).values
        return obs.astype(np.float32)