# src/eval/plots.py

"""
Plot Generation for MARL Evaluation
===================================

Produces:
 - Equity curve overlay plot
 - Drawdown plot
 - Daily returns histogram
 - Rolling Sharpe chart
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_all_plots(equity_big: pd.DataFrame, output_dir: str):
    """High-level function that calls all plot sub-functions."""
    plot_equity_curves(equity_big, output_dir)
    plot_drawdowns(equity_big, output_dir)
    plot_daily_returns(equity_big, output_dir)
    plot_rolling_sharpe(equity_big, output_dir)


def plot_equity_curves(equity_big, output_dir):
    plt.figure(figsize=(12,6))
    for rid, group in equity_big.groupby("run_id"):
        plt.plot(group["equity"], alpha=0.5, label=f"Run {rid}")
    plt.title("Equity Curves Across MARL Runs")
    plt.savefig(f"{output_dir}/equity_curves.png")
    plt.close()


def plot_drawdowns(equity_big, output_dir):
    plt.figure(figsize=(12,6))
    for rid, group in equity_big.groupby("run_id"):
        dd = (group["equity"].cummax() - group["equity"]) / group["equity"].cummax()
        plt.plot(dd, alpha=0.5)
    plt.title("Drawdowns Across MARL Runs")
    plt.savefig(f"{output_dir}/drawdowns.png")
    plt.close()


def plot_daily_returns(equity_big, output_dir):
    plt.figure(figsize=(12,6))
    daily_ret = equity_big.groupby("run_id")["equity"].pct_change()
    daily_ret.hist(bins=50)
    plt.title("Daily Returns Distribution")
    plt.savefig(f"{output_dir}/daily_returns.png")
    plt.close()


def plot_rolling_sharpe(equity_big, output_dir, window=200):
    plt.figure(figsize=(12,6))
    for rid, group in equity_big.groupby("run_id"):
        ret = group["equity"].pct_change()
        roll = ret.rolling(window).mean() / ret.rolling(window).std()
        plt.plot(roll, alpha=0.5)
    plt.title("Rolling Sharpe Ratio")
    plt.savefig(f"{output_dir}/rolling_sharpe.png")
    plt.close()