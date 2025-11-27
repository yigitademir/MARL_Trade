# Clean Single-Agent Backtest (No Risk Engine)

import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from src.env.trading_env import TradingEnv


def load_latest(prefix):
    """Return newest model matching prefix under models/single_agents."""
    import glob
    files = glob.glob(f"models/single_agents/{prefix}*.zip")
    if not files:
        raise FileNotFoundError(f"No model found for prefix: {prefix}")
    return max(files, key=os.path.getmtime)


def run_backtest(model, df, window_size=10, initial_balance=10000, leverage=5.0, fee_rate=0.0004):
    """Simple baseline backtest without ATR / risk engine."""
    
    env = TradingEnv(
        df=df,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=fee_rate,
        leverage=leverage,
        max_episode_steps=len(df)  # full run
    )

    obs, info = env.reset()
    done = False

    equity_curve = []
    trades = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        equity_curve.append({
            "timestamp": df.loc[env.current_step, "timestamp"],
            "equity": info["equity"],
            "position": info["position"]
        })

        if info["trade_executed"]:
            trades.append({
                "timestamp": df.loc[env.current_step, "timestamp"],
                "position": info["position"],
                "entry_price": info["entry_price"],
                "equity": info["equity"]
            })

        done = terminated or truncated

    final_equity = equity_curve[-1]["equity"]
    roi_pct = (final_equity - initial_balance) / initial_balance * 100
    max_dd = info["max_drawdown"]

    summary = {
        "final_equity": final_equity,
        "roi_pct": roi_pct,
        "max_drawdown_pct": max_dd * 100,
        "trades": len(trades),
    }

    return summary, pd.DataFrame(equity_curve), pd.DataFrame(trades)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, required=True, choices=["5m", "15m", "1h", "4h"])
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    prefix = f"{args.symbol}_{args.timeframe}"

    # === Load model ===
    print(f"\n=== Loading model for {args.timeframe} ===")
    model_path = load_latest(prefix)
    model = PPO.load(model_path)

    # === Load data ===
    data_path = f"data/processed/{args.symbol}_{args.timeframe}_features.parquet"
    df = pd.read_parquet(data_path)

    # === Create output directory ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"logs/single_agent/backtests/{timestamp}_{args.timeframe}"
    os.makedirs(out_dir, exist_ok=True)

    # === Run backtest ===
    print("\n=== Running baseline backtest ===")
    summary, equity_df, trades_df = run_backtest(
        model,
        df,
        window_size=10,
        initial_balance=10000,
        leverage=5.0
    )

    # === Save files ===
    equity_df.to_csv(f"{out_dir}/equity_curve.csv", index=False)
    trades_df.to_csv(f"{out_dir}/trades.csv", index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # === Print summary ===
    print("\nBACKTEST SUMMARY:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved results to:")
    print(out_dir)


if __name__ == "__main__":
    main()