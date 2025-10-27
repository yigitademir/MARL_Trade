import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size: int = 10, initial_balance: float = 10000):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True) # ensure clean index
        self.window_size = window_size
        self.current_step = self.window_size
        self.initial_balance = self.initial_balance 

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
        obs = self._get_observation()
        info = {}
        return obs, reward, done, info
    
    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        obs = window.drop(columns=["timestamp"]).values
        return obs.astype(np.float32)