# src/eval/batch_runner.py

"""
Batch Experiment Runner for Multi-Agent PPO
===========================================

This module runs repeated MARL evaluations using the frozen
environment and the MultiAgentBacktester.

Responsibilities:
 - Run N repeated full backtests
 - Save raw outputs (equity, trades, actions)
 - Compute high-level summary table
 - Delegate metrics to metrics.py
 - Delegate plotting to plots.py
"""

import os
import json
from datetime import datetime
import pandas as pd
import argparse

from src.multi_agent.backtester import MultiAgentBacktesterV1
from src.eval.metrics import compute_all_metrics
from src.eval.plots import generate_all_plots

from tests.multiagent_test import load_models, load_data  # clean reuse


def run_single_experiment(models, data, output_dir, leverage=5.0):
    """
    Runs one full multi-agent backtest and saves:
     - equity curve
     - trades
     - per-agent actions
     - summary.json
    Returns summary dict and equity DataFrame for meta-analysis.
    """

    bt = MultiAgentBacktesterV1(
        models=models,
        data=data,
        window_size=10,
        strategy="majority_vote"
    )

    summary, equity_df, trade_df, actions = bt.run_trading(
        initial_balance=10_000,
        leverage=leverage,
        fee_rate=0.0004,
        base_timeframe="1h",
        verbose=False
    )

    # Save raw data
    equity_df.to_csv(f"{output_dir}/equity.csv", index=False)
    trade_df.to_csv(f"{output_dir}/trades.csv", index=False)

    actions_dir = f"{output_dir}/actions"
    os.makedirs(actions_dir, exist_ok=True)
    for tf, df in actions.items():
        df.to_csv(f"{actions_dir}/{tf}.csv", index=False)

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    return summary, equity_df


def run_marl_batch(n=10, leverage=5.0):
    """
    Runs N full MARL experiments back-to-back.
    Stores everything under:
       experiments/marl_v1/runs/<timestamp>/

    Produces:
      - run_01/, run_02/, ... run_N/
      - metrics_summary.csv
      - aggregate_plots/
    """

    # Folder structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/marl_v1/runs/{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    # Load once for efficiency
    models = load_models()
    data = load_data()

    all_summaries = []
    all_equities = []

    for i in range(1, n + 1):
        print(f"=== Running experiment {i}/{n} ===")

        run_dir = f"{base_dir}/run_{i:02d}"
        os.makedirs(run_dir, exist_ok=True)

        summary, equity_df = run_single_experiment(
            models=models,
            data=data,
            output_dir=run_dir,
            leverage=leverage
        )

        summary["run_id"] = i
        all_summaries.append(summary)
        all_equities.append(equity_df.assign(run_id=i))

    # Combine summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(f"{base_dir}/metrics_summary.csv", index=False)

    # Combine all equity curves for meta-analysis
    equity_big = pd.concat(all_equities, ignore_index=True)
    equity_big.to_csv(f"{base_dir}/equity_all_runs.csv", index=False)

    # Global metrics + plots
    compute_all_metrics(summary_df, base_dir)
    generate_all_plots(equity_big, base_dir)

    print("\nAll batch experiments completed.")
    print(f"Results saved under: {base_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of experiments")
    parser.add_argument("--leverage", type=float, default=5.0)
    args = parser.parse_args()

    run_marl_batch(n=args.n, leverage=args.leverage)