"""
Train All Timeframes Script
===========================
Runs train_single_agent.py sequentially for multiple timeframes
and prints a summary of the new results.

Usage (from project root):
    python agents/train_all_timeframes.py

Optional arguments:
    --symbol BTCUSDT
    --timeframes 5m,15m,1h,4h
    --total_timesteps 100000
    --leverage 5.0
    --initial_balance 10000
    --window_size 10
"""

import os
import sys
import argparse
import csv
from datetime import datetime
import subprocess


# ----------------------------
# Helpers for results tracking
# ----------------------------

def load_existing_results(logfile):
    """Load existing results into a set of (symbol, timeframe, model_path) keys."""
    if not os.path.exists(logfile):
        return set()

    keys = set()
    with open(logfile, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row.get("symbol"), row.get("timeframe"), row.get("model_path")))
    return keys


def load_new_results(logfile, old_keys):
    """Return list of new result rows (dicts) based on difference in keys."""
    if not os.path.exists(logfile):
        return []

    new_rows = []
    with open(logfile, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("symbol"), row.get("timeframe"), row.get("model_path"))
            if key not in old_keys:
                new_rows.append(row)
    return new_rows


# ----------------------------
# Argument parsing
# ----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train single-agent PPO on multiple timeframes")

    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading pair symbol (default: BTCUSDT)")
    parser.add_argument("--timeframes", type=str, default="5m,15m,1h,4h",
                        help="Comma-separated list of timeframes (default: 5m,15m,1h,4h)")

    # Shared training / env args
    parser.add_argument("--total_timesteps", type=int, default=100000,
                        help="Total timesteps per timeframe (default: 100000)")
    parser.add_argument("--leverage", type=float, default=5.0,
                        help="Leverage (default: 5.0)")
    parser.add_argument("--initial_balance", type=float, default=10000.0,
                        help="Initial balance (default: 10000)")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Observation window size (default: 10)")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Train ratio (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation ratio (default: 0.15)")
    parser.add_argument("--data_path", type=str, default="data/processed",
                        help="Path to processed features (default: data/processed)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Model output directory (default: models)")
    parser.add_argument("--save_freq", type=int, default=10000,
                        help="Save / eval frequency (default: 10000)")

    return parser.parse_args()


# ----------------------------
# Main multi-timeframe runner
# ----------------------------

def main():
    args = parse_arguments()

    # Parse timeframe list
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]

    print("\n" + "=" * 70)
    print("🚀 MULTI-TIMEFRAME TRAINING LAUNCHER")
    print("=" * 70)
    print(f"Symbol           : {args.symbol}")
    print(f"Timeframes       : {', '.join(timeframes)}")
    print(f"Total timesteps  : {args.total_timesteps:,}")
    print(f"Leverage         : {args.leverage}x")
    print(f"Initial balance  : {args.initial_balance}")
    print(f"Window size      : {args.window_size}")
    print(f"Data path        : {args.data_path}")
    print(f"Output dir       : {args.output_dir}")
    print("=" * 70 + "\n")

    # Track existing results to identify new runs at the end
    results_log = "logs/results.csv"
    old_keys = load_existing_results(results_log)

    for tf in timeframes:
        print("\n" + "=" * 70)
        print(f"🎯 TRAINING TIMEFRAME: {tf}")
        print("=" * 70)

        cmd = [
            sys.executable,
            "-m",
            "src.agents.train_single_agent",
            "--symbol", args.symbol,
            "--timeframe", tf,
            "--total_timesteps", str(args.total_timesteps),
            "--leverage", str(args.leverage),
            "--initial_balance", str(args.initial_balance),
            "--window_size", str(args.window_size),
            "--train_ratio", str(args.train_ratio),
            "--val_ratio", str(args.val_ratio),
            "--data_path", args.data_path,
            "--output_dir", args.output_dir,
            "--save_freq", str(args.save_freq),
        ]

        print(f"Running: {' '.join(cmd)}\n")

        try:
            # Run train_single_agent.py as a subprocess
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("\n" + "=" * 70)
            print(f"❌ Training failed for timeframe {tf}")
            print(f"Return code: {e.returncode}")
            print("=" * 70 + "\n")
            # Continue to next timeframe instead of exiting
            continue

    # After all runs, load only the new results from logs/results.csv
    new_rows = load_new_results(results_log, old_keys)

    print("\n" + "=" * 70)
    print("📊 SUMMARY OF NEW RUNS")
    print("=" * 70)

    if not new_rows:
        print("No new results found in logs/results.csv (maybe logging failed?).")
        return

    # Sort by timeframe for nicer output
    def tf_sort_key(row):
        order = {"5m": 0, "15m": 1, "1h": 2, "4h": 3}
        return order.get(row.get("timeframe", ""), 99)

    new_rows_sorted = sorted(new_rows, key=tf_sort_key)

    # Print header
    print(f"{'Timeframe':<8} {'Timesteps':>10} {'ROI %':>10} {'PnL':>12} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 70)

    for row in new_rows_sorted:
        tf = row.get("timeframe", "")
        steps = row.get("total_timesteps", "")
        roi = row.get("eval_mean_roi", "")
        pnl = row.get("eval_mean_pnl", "")
        sharpe = row.get("eval_mean_sharpe", "")
        max_dd = row.get("eval_mean_max_dd", "")

        # Safe formatting
        try:
            roi_f = float(roi)
        except (TypeError, ValueError):
            roi_f = float("nan")

        try:
            pnl_f = float(pnl)
        except (TypeError, ValueError):
            pnl_f = float("nan")

        try:
            sharpe_f = float(sharpe)
        except (TypeError, ValueError):
            sharpe_f = float("nan")

        try:
            dd_f = float(max_dd)
        except (TypeError, ValueError):
            dd_f = float("nan")

        print(f"{tf:<8} {steps:>10} {roi_f:>10.2f} {pnl_f:>12.2f} {sharpe_f:>10.3f} {dd_f:>10.3f}")

    print("-" * 70)
    print("✅ Multi-timeframe training completed.\n")


if __name__ == "__main__":
    main()