"""
Train All Timeframes Script
===========================
Runs train_single_agent.py sequentially for multiple timeframes
and prints a clean summary table of ROI, PnL, Sharpe, MaxDD.

Usage:
    python agents/train_all_timeframes.py
"""

import os
import sys
import argparse
import subprocess


# ============================================================
# Argument parsing
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PPO agents for multiple timeframes")

    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframes", type=str, default="5m,15m,1h,4h")

    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--initial_balance", type=float, default=10000.0)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--data_path", type=str, default="data/processed")

    return parser.parse_args()


# ============================================================
# Main Multi-Timeframe Runner
# ============================================================

def main():
    args = parse_arguments()
    all_results = []

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]

    print("\n" + "=" * 70)
    print(" MULTI-TIMEFRAME TRAINING LAUNCHER")
    print("=" * 70)
    print(f"Symbol           : {args.symbol}")
    print(f"Timeframes       : {', '.join(timeframes)}")
    print(f"Total timesteps  : {args.total_timesteps:,}")
    print(f"Leverage         : {args.leverage}x")
    print(f"Initial balance  : {args.initial_balance}")
    print(f"Window size      : {args.window_size}")
    print(f"Data path        : {args.data_path}")
    print("=" * 70 + "\n")

    # ---------------------------------------------------------
    # TRAIN EACH TIMEFRAME ONCE (Popen version with live output)
    # ---------------------------------------------------------
    for tf in timeframes:
        print("\n" + "=" * 70)
        print(f" TRAINING TIMEFRAME: {tf}")
        print("=" * 70)

        cmd = [
            sys.executable,
            "-m", "src.agents.train_single_agent",
            "--symbol", args.symbol,
            "--timeframe", tf,
            "--total_timesteps", str(args.total_timesteps),
            "--leverage", str(args.leverage),
            "--initial_balance", str(args.initial_balance),
            "--window_size", str(args.window_size),
            "--train_ratio", str(args.train_ratio),
            "--val_ratio", str(args.val_ratio),
            "--data_path", args.data_path,
        ]

        print(f"Running: {' '.join(cmd)}\n")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        for line in process.stdout:
            print(line, end="")  # keep progress bar
            output_lines.append(line)

        process.wait()
        if process.returncode != 0:
            print(f"Training failed for timeframe {tf}")
            continue

        # -----------------------------------------------------
        # Extract metrics from output
        # -----------------------------------------------------
        final_equity = roi = pnl = sharpe = max_dd = None

        for line in output_lines:
            line = line.strip()

            if line.startswith("Final equity"):
                final_equity = float(line.split(":")[-1].replace(",", "").strip())

            elif line.startswith("ROI"):
                roi = float(line.split(":")[-1].replace("%", "").strip())

            elif line.startswith("PnL"):
                pnl = float(line.split(":")[-1].strip())

            elif line.startswith("Sharpe"):
                sharpe = float(line.split(":")[-1].strip())

            elif line.startswith("Max drawdown"):
                max_dd = float(line.split(":")[-1].replace("%", "").strip())

        all_results.append({
            "timeframe": tf,
            "timesteps": args.total_timesteps,
            "ROI": roi,
            "PnL": pnl,
            "Sharpe": sharpe,
            "MaxDD": max_dd
        })

    # ---------------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(" SUMMARY OF TRAINING RUNS")
    print("=" * 70)
    print(f"{'TF':<8} {'Timesteps':>10} {'ROI %':>10} {'PnL':>12} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 70)

    order = {"5m": 0, "15m": 1, "1h": 2, "4h": 3}
    all_results_sorted = sorted(all_results, key=lambda r: order.get(r["timeframe"], 99))

    for r in all_results_sorted:
        print(
            f"{r['timeframe']:<8} "
            f"{r['timesteps']:>10} "
            f"{(r['ROI'] if r['ROI'] is not None else float('nan')):>10.2f} "
            f"{(r['PnL'] if r['PnL'] is not None else float('nan')):>12.2f} "
            f"{(r['Sharpe'] if r['Sharpe'] is not None else float('nan')):>10.3f} "
            f"{(r['MaxDD'] if r['MaxDD'] is not None else float('nan')):>10.3f}"
        )

    print("-" * 70)
    print(" Training completed.\n")


if __name__ == "__main__":
    main()