"""
Multi-Agent Backtest Runner
============================

Runs the multi-agent policy using the trained single-agent models from
models/single_agents/, loads their predictions, passes them through
the coordinator, and executes the full trading simulation using the
Backtester v3 (risk engine).

This script is used for:
 - Checking that all 4 agents work together
 - Running clean multi-agent backtests
 - Saving results for thesis charts
"""

import os
import glob
import json
from datetime import datetime
import pandas as pd
from stable_baselines3 import PPO

# Backtester
from src.multi_agent.backtester import MultiAgentBacktesterV1


# ============================================================
# Helper: Load LATEST model by prefix (per timeframe)
# ============================================================

def load_latest(prefix: str) -> str:
    """
    Return the path of the most recently saved model for this timeframe.

    Example:
        load_latest("BTCUSDT_5m")
        → models/single_agents/BTCUSDT_5m_20251127_153210_final.zip
    """
    files = glob.glob(f"models/single_agents/{prefix}*.zip")
    if not files:
        raise FileNotFoundError(f"No saved model found for prefix: {prefix}")
    return max(files, key=os.path.getmtime)


# ============================================================
# Load All 4 Agents
# ============================================================

def load_models():
    """Load the latest trained PPO agent for each timeframe."""
    print("\n=== Loading Agents ===")
    return {
        "5m": PPO.load(load_latest("BTCUSDT_5m")),
        "15m": PPO.load(load_latest("BTCUSDT_15m")),
        "1h": PPO.load(load_latest("BTCUSDT_1h")),
        "4h": PPO.load(load_latest("BTCUSDT_4h")),
    }


# ============================================================
# Load Feature Data
# ============================================================

def load_data():
    """Load all processed feature datasets required for multi-agent backtesting."""
    print("\n=== Loading Data ===")
    return {
        "5m": pd.read_parquet("data/processed/BTCUSDT_5m_features.parquet"),
        "15m": pd.read_parquet("data/processed/BTCUSDT_15m_features.parquet"),
        "1h": pd.read_parquet("data/processed/BTCUSDT_1h_features.parquet"),
        "4h": pd.read_parquet("data/processed/BTCUSDT_4h_features.parquet"),
    }


# ============================================================
# Main Backtest Logic
# ============================================================

def main():

    # Ensure folder structure exists
    os.makedirs("logs/multi_agent/backtests", exist_ok=True)

    # 1) Load the four PPO agents
    models = load_models()

    # 2) Load datasets for each timeframe
    data = load_data()

    # 3) Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"logs/multi_agent/backtests/{timestamp}_majority_vote"
    os.makedirs(run_dir, exist_ok=True)

    print("\n=== Initializing Multi-Agent Backtester ===")

    # 4) Create the backtester engine
    bt = MultiAgentBacktesterV1(
        models=models,
        data=data,
        window_size=10,
        strategy="majority_vote"
    )

    # 5) Run the full multi-agent trading simulation
    print("\n=== Running Backtest ===")
    summary, equity_df, trade_df, agent_actions = bt.run_trading(
        initial_balance=10_000,
        leverage=5.0,
        fee_rate=0.0004,
        base_timeframe="1h",
        verbose=True
    )

    # 6) Save backtest results
    equity_df.to_csv(f"{run_dir}/equity_curve.csv", index=False)
    trade_df.to_csv(f"{run_dir}/trades.csv", index=False)
    for tf, df in agent_actions.items():
        df.to_csv(f"{run_dir}/actions{tf}.csv", index=False)

    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved multi-agent results to:")
    print(f" - {run_dir}/equity_curve.csv")
    print(f" - {run_dir}/trades.csv")
    print(f" - {run_dir}/summary.json")
    print("==============================================")


if __name__ == "__main__":
    main()