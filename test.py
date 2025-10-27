import pandas as pd
from env.trading_env import TradingEnv

# === Load processed features ===
df = pd.read_parquet("data/processed/BTCUSDT_15m_features.parquet")

# === Create environment instance ===
env = TradingEnv(df=df, window_size=10, initial_balance=10000)

# === Reset environment ===
obs = env.reset()
print("Initial observation shape:", obs.shape)

# === Simulate a few steps manually ===
for step in range(5):
    action = step % 3  # rotate: 0 (hold), 1 (buy), 2 (sell)
    obs, reward, done, info = env.step(action)
    
    print(f"\n🔁 Step {step + 1}")
    print(f"Action: {action} (0=Hold, 1=Buy, 2=Sell)")
    print(f"Reward: {reward:.2f}")
    print(f"Position: {info['position']}")
    print(f"Balance: {info['balance']:.2f}")
    print(f"Observation shape: {obs.shape}")
    
    if done:
        print("🔚 Episode finished!")
        break