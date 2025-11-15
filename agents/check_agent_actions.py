import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env.trading_env import TradingEnv

# Load test data
df = pd.read_parquet("../data/processed/BTCUSDT_1h_features.parquet")
test_df = df.iloc[int(len(df)*0.8):].copy()

# Load model
model = PPO.load("../models/final/BTCUSDT_1h_20251115_144430")

# Create environment
env = TradingEnv(df=test_df, window_size=10, initial_balance=10000)

# Run one episode and track actions
obs, info = env.reset()
actions_taken = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions_taken.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Analyze actions
unique, counts = np.unique(actions_taken, return_counts=True)
action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}

print("\n📊 AGENT ACTION DISTRIBUTION")
print("="*60)
total = len(actions_taken)
for action, count in zip(unique, counts):
    pct = (count / total) * 100
    print(f"{action_names[action]:6s}: {count:5d} ({pct:5.1f}%)")
print("="*60)

if counts[unique == 0][0] > total * 0.9:
    print("\n⚠️  PROBLEM: Agent is holding >90% of the time!")
    print("   The agent learned NOT to trade.")
else:
    print("\n✅ Agent is trading actively")
