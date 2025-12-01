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
 - Performing controlled experiments
 - Saving results for charts
"""

import os
import glob
import json
from datetime import datetime
import pandas as pd
from scipy.stats import entropy
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
# Controlled Experiments
# Each experiment isolates one factor for analysis.
# ============================================================

def experiment_no_leverage(bt, run_dir):
    """Experiment A — No leverage (L=1 baseline test)."""
    out_dir = f"{run_dir}/experiment_A_no_leverage"
    os.makedirs(out_dir, exist_ok=True)

    summary, eq, trades, acts = bt.run_trading(
        initial_balance=10_000,
        leverage=1.0,              # LEVERAGE = 1
        fee_rate=0.0004,
        base_timeframe="1h",
        verbose=False
    )

    eq.to_csv(f"{out_dir}/equity_curve.csv", index=False)
    trades.to_csv(f"{out_dir}/trades.csv", index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("✔ Experiment A (No leverage) done")
    print("\nExperiment A — No Leverage:")
    print(f"  ROI: {summary['roi_pct']:.2f}%")
    print(f"  MaxDD: {summary['max_drawdown_pct']:.2f}%")
    print(f"  Trades: {summary['trades']}")
    print("-" * 60)


def experiment_random_agent(data, models, run_dir):
    """Experiment B — Replace 5m agent with a random-action agent."""

    class RandomAgent:
        def predict(self, obs, deterministic=True):
            import numpy as np
            return np.random.choice([0, 1, 2]), None

    models_B = models.copy()
    models_B["5m"] = RandomAgent()   # Replace 5m agent

    bt_B = MultiAgentBacktesterV1(
        models=models_B,
        data=data,
        window_size=10,
        strategy="majority_vote"
    )

    out_dir = f"{run_dir}/experiment_B_random_5m"
    os.makedirs(out_dir, exist_ok=True)

    summary, eq, trades, acts = bt_B.run_trading(
        initial_balance=10_000,
        leverage=5.0,
        fee_rate=0.0004,
        base_timeframe="1h",
        verbose=False
    )

    eq.to_csv(f"{out_dir}/equity_curve.csv", index=False)
    trades.to_csv(f"{out_dir}/trades.csv", index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("✔ Experiment B (Random 5m) done")
    print("\nExperiment B — Random 5m Agent:")
    print(f"  ROI: {summary['roi_pct']:.2f}%")
    print(f"  MaxDD: {summary['max_drawdown_pct']:.2f}%")
    print(f"  Trades: {summary['trades']}")
    print("-" * 60)


def experiment_drop_worst(data, models, run_dir):
    """Experiment C — Remove worst agent (4h) and retest multi-agent system."""

    models_C = {tf: m for tf, m in models.items() if tf != "4h"}
    data_C = {tf: df for tf, df in data.items() if tf != "4h"}

    bt_C = MultiAgentBacktesterV1(
        models=models_C,
        data=data_C,
        window_size=10,
        strategy="majority_vote"
    )

    out_dir = f"{run_dir}/experiment_C_drop_4h"
    os.makedirs(out_dir, exist_ok=True)

    summary, eq, trades, acts = bt_C.run_trading(
        initial_balance=10_000,
        leverage=5.0,
        fee_rate=0.0004,
        base_timeframe="1h",
        verbose=False
    )

    eq.to_csv(f"{out_dir}/equity_curve.csv", index=False)
    trades.to_csv(f"{out_dir}/trades.csv", index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("✔ Experiment C (Drop 4h) done")
    print("\nExperiment C — Drop 4h Agent:")
    print(f"  ROI: {summary['roi_pct']:.2f}%")
    print(f"  MaxDD: {summary['max_drawdown_pct']:.2f}%")
    print(f"  Trades: {summary['trades']}")
    print("-" * 60)


def experiment_single_agent(data, models, run_dir):
    """Experiment D — Run only the 15m agent for isolation analysis."""

    models_D = {"15m": models["15m"]}
    data_D = {"15m": data["15m"]}

    bt_D = MultiAgentBacktesterV1(
        models=models_D,
        data=data_D,
        window_size=10,
        strategy="majority_vote"
    )

    out_dir = f"{run_dir}/experiment_D_single_15m"
    os.makedirs(out_dir, exist_ok=True)

    summary, eq, trades, acts = bt_D.run_trading(
        initial_balance=10_000,
        leverage=5.0,
        fee_rate=0.0004,
        base_timeframe="15m",
        verbose=False
    )

    eq.to_csv(f"{out_dir}/equity_curve.csv", index=False)
    trades.to_csv(f"{out_dir}/trades.csv", index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("✔ Experiment D (Single 15m) done")
    print("\nExperiment D — Single 15m Agent:")
    print(f"  ROI: {summary['roi_pct']:.2f}%")
    print(f"  MaxDD: {summary['max_drawdown_pct']:.2f}%")
    print(f"  Trades: {summary['trades']}")
    print("-" * 60)


# ============================================================
# Main Backtest Logic
# ============================================================

def main():

    # Ensure folder structure exists
    os.makedirs("logs/multi_agent/backtests", exist_ok=True)

    # 1) Load PPO agents
    models = load_models()

    # 2) Load all timeframe datasets
    data = load_data()

    # 3) Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"logs/multi_agent/backtests/{timestamp}_majority_vote"
    os.makedirs(run_dir, exist_ok=True)

    print("\n=== Initializing Multi-Agent Backtester ===")

    # 4) Instantiate the backtester
    bt = MultiAgentBacktesterV1(
        models=models,
        data=data,
        window_size=10,
        strategy="majority_vote"
    )

    # 5) PRIMARY multi-agent backtest
    print("\n=== Running Backtest ===")
    summary, equity_df, trade_df, agent_actions = bt.run_trading(
        initial_balance=10_000,
        leverage=5.0,
        fee_rate=0.0004,
        base_timeframe="1h",
        verbose=True
    )

    # === Save main backtest results ===
    equity_df.to_csv(f"{run_dir}/equity_curve.csv", index=False)
    trade_df.to_csv(f"{run_dir}/trades.csv", index=False)

    actions_dir = f"{run_dir}/agent_actions"
    os.makedirs(actions_dir, exist_ok=True)
    for tf, df in agent_actions.items():
        df.to_csv(f"{actions_dir}/{tf}_actions.csv", index=False)

    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved multi-agent results to:")
    print(f" - {run_dir}/equity_curve.csv")
    print(f" - {run_dir}/trades.csv")
    print(f" - {run_dir}/summary.json")
    print("==============================================")

    # ============================================================
    # Controlled Experiments
    # ============================================================

    print("\n Running Controlled Experiments...\n")

    experiment_no_leverage(bt, run_dir)
    experiment_random_agent(data, models, run_dir)
    experiment_drop_worst(data, models, run_dir)
    experiment_single_agent(data, models, run_dir)

    print("\n All controlled experiments completed.\n")


    # ============================================================
    # Per-Agent Behavioral Statistics
    # ============================================================

    stats = {}
    for tf, df in agent_actions.items():
        actions = df["action"]

        freq = actions.value_counts(normalize=True).to_dict()
        variance = float(actions.var())
        ent = float(entropy(actions.value_counts(normalize=True)))
        ac = float(actions.autocorr(lag=1)) if len(actions) > 1 else 0.0

        transitions = {}
        for a_from in [0, 1, 2]:
            for a_to in [0, 1, 2]:
                mask = (actions.shift(0) == a_from) & (actions.shift(-1) == a_to)
                transitions[f"{a_from}->{a_to}"] = mask.mean()

        stats[tf] = {
            "freq": freq,
            "variance": variance,
            "entropy": ent,
            "autocorrelation": ac,
            "transitions": transitions,
        }

    with open(f"{run_dir}/agent_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("\nSaved per-agent statistics:")
    print(f" - {run_dir}/agent_stats.json")

    # ============================================================
    # Action Correlation Matrices
    # ============================================================

    merged = None
    for tf, df in agent_actions.items():
        df = df.rename(columns={"action": tf})
        merged = df if merged is None else merged.merge(df, on="timestamp", how="inner")

    action_cols = ["5m", "15m", "1h", "4h"]
    corr_pearson = merged[action_cols].corr(method="pearson")
    corr_spearman = merged[action_cols].corr(method="spearman")

    corr_pearson.to_csv(f"{run_dir}/correlation_pearson.csv")
    corr_spearman.to_csv(f"{run_dir}/correlation_spearman.csv")

    print("\nSaved correlation matrices:")
    print(f" - {run_dir}/correlation_pearson.csv")
    print(f" - {run_dir}/correlation_spearman.csv")


if __name__ == "__main__":
    main()