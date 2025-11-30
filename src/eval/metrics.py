# src/eval/metrics.py

"""
Performance Metrics for MARL Evaluation
=======================================

This module computes all high-level performance metrics:
 - ROI
 - Max Drawdown
 - Sharpe
 - Sortino
 - Calmar Ratio
 - Volatility
 - Win/Loss statistics
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns):
    return returns.mean() / returns.std() if returns.std() != 0 else 0.0


def sortino_ratio(returns):
    downside = returns[returns < 0]
    dd = downside.std()
    return returns.mean() / dd if dd != 0 else 0.0


def calmar_ratio(roi, max_dd):
    return roi / max_dd if max_dd != 0 else 0.0


def compute_all_metrics(summary_df: pd.DataFrame, output_dir: str):
    """
    Compute aggregate metrics across runs and save to:
        output_dir/aggregate_metrics.json
    """

    metrics = {
        "ROI_mean": float(summary_df["roi_pct"].mean()),
        "ROI_std": float(summary_df["roi_pct"].std()),
        "DD_mean": float(summary_df["max_drawdown_pct"].mean()),
        "DD_std": float(summary_df["max_drawdown_pct"].std()),
        "Trades_mean": float(summary_df["trades"].mean()),
        "Trades_std": float(summary_df["trades"].std()),
    }

    # Save
    import json
    with open(f"{output_dir}/aggregate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)