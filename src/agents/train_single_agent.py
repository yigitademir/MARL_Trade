"""
Single-Agent PPO Training Script
=================================

This script:

1) Loads processed feature data for a given timeframe
2) Splits the data into train/validation/test sets
3) Builds the TradingEnv for PPO training
4) Trains a PPO agent with monitoring + checkpointing
5) Evaluates the agent on unseen test data
6) Saves final model + config + evaluation results into a clean
   timestamped directory for reproducibility:
   
   logs/single_agent/train_runs/<timestamp>/
   ├── model.zip
   ├── config.json
   ├── eval_results.json
   ├── training_monitor.csv
   ├── validation_monitor.csv
   └── tensorboard/

7) Appends a summary row into logs/single_agent/results.csv
"""

import os
import argparse
import json
import csv
from datetime import datetime

import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.env.trading_env import TradingEnv


# ============================================================
# 1) ARGUMENT PARSER
# ============================================================

def parse_arguments():
    """Parse all CLI arguments used for training a single PPO agent."""
    
    parser = argparse.ArgumentParser(description="Train single-agent PPO")

    # Data & timeframe selection
    parser.add_argument("--timeframe", type=str, default="1h",
                        choices=["5m", "15m", "1h", "4h"])
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--data_path", type=str, default="data/processed")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    # Environment settings
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--initial_balance", type=float, default=10000.0)
    parser.add_argument("--leverage", type=float, default=5.0)

    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Training process
    parser.add_argument("--total_timesteps", type=int, default=100000)

    return parser.parse_args()


# ============================================================
# 2) DIRECTORY HELPERS
# ============================================================

def create_run_directories(args):
    """
    Create a timestamped directory for all logs / outputs of a single training run:

    logs/single_agent/train_runs/<timestamp>/
    models/single_agents/
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model folder (shared across runs)
    model_dir = "models/single_agents"
    os.makedirs(model_dir, exist_ok=True)

    # Log folder (specific to this run)
    run_dir = f"logs/single_agent/train_runs/{timestamp}"
    tb_dir = f"{run_dir}/tensorboard"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    return timestamp, run_dir, tb_dir, model_dir


# ============================================================
# 3) DATA LOADING
# ============================================================

def load_data(args):
    """Load processed feature data for the selected timeframe."""
    
    print("\n" + "=" * 70)
    print(f"LOADING DATA FOR {args.symbol} @ {args.timeframe}")
    print("=" * 70)

    filename = f"{args.symbol}_{args.timeframe}_features.parquet"
    filepath = os.path.join(args.data_path, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature file not found: {filepath}")

    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} rows.")
    print(df.head(3))

    return df


def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split the dataset into train, val, test partitions."""
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    return train_df, val_df, test_df


# ============================================================
# 4) ENVIRONMENT CREATION
# ============================================================

def create_environments(train_df, val_df, args, run_dir):
    """
    Create monitored training and validation environments.
    Monitor(...) automatically logs step-level info into CSV.
    """

    train_env = TradingEnv(
        df=train_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=0.0004,
        leverage=args.leverage
    )

    train_env = Monitor(
        train_env,
        filename=f"{run_dir}/training_monitor.csv",
        info_keywords=("roi", "sharpe_ratio", "max_drawdown")
    )

    val_env = TradingEnv(
        df=val_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=0.0004,
        leverage=args.leverage
    )

    val_env = Monitor(
        val_env,
        filename=f"{run_dir}/validation_monitor.csv",
        info_keywords=("roi", "sharpe_ratio", "max_drawdown")
    )

    return train_env, val_env


# ============================================================
# 5) PPO MODEL CREATION
# ============================================================

def create_ppo_model(train_env, tb_dir, args):
    """Create PPO model with configured hyperparameters."""

    model = PPO(
        "MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tb_dir,
        policy_kwargs=dict(net_arch=[128, 128, 64]),
        verbose=1,
        seed=42,
        device="auto"
    )
    return model


# ============================================================
# 6) EVALUATION ROUTINE
# ============================================================

def evaluate_model(model, test_df, args):
    """Evaluate trained model on unseen test data."""

    test_env = TradingEnv(
        df=test_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=0.0004,
        leverage=args.leverage
    )

    episodes = 10
    rois, pnls, sharpe, dd, rewards = [], [], [], [], []

    for _ in range(episodes):
        obs, info = test_env.reset()
        done = False
        total_r = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_r += reward
            done = terminated or truncated

        rewards.append(total_r)
        rois.append(info["roi"])
        pnls.append(info["balance"] - args.initial_balance)
        sharpe.append(info["sharpe_ratio"])
        dd.append(info["max_drawdown"])

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_roi": float(np.mean(rois)),
        "mean_pnl": float(np.mean(pnls)),
        "mean_sharpe": float(np.mean(sharpe)),
        "mean_max_dd": float(np.mean(dd)),   # fraction (0.25 = 25%)
    }


# ============================================================
# 7) LOGGING RESULTS TO CSV
# ============================================================

def append_results(args, eval_stats, model_path):
    """Append summary line to logs/single_agent/results.csv."""
    
    os.makedirs("logs/single_agent", exist_ok=True)
    csv_path = "logs/single_agent/results.csv"

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_path": model_path,
        "initial_balance": args.initial_balance,
        "leverage": args.leverage,
        "total_timesteps": args.total_timesteps,
        "eval_mean_reward": eval_stats["mean_reward"],
        "eval_mean_roi": eval_stats["mean_roi"],
        "eval_mean_pnl": eval_stats["mean_pnl"],
        "eval_mean_sharpe": eval_stats["mean_sharpe"],
        "eval_mean_max_dd": eval_stats["mean_max_dd"],
    }

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_results_to_csv(args, eval_stats, save_path):
    """
    Legacy / additional summary CSV:
    logs/results.csv
    """
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/results.csv"

    file_exists = os.path.exists(csv_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_path": save_path,
        "initial_balance": args.initial_balance,
        "leverage": args.leverage,
        "total_timesteps": args.total_timesteps,
        "eval_mean_reward": eval_stats["mean_reward"],
        "eval_mean_roi": eval_stats["mean_roi"],
        "eval_mean_pnl": eval_stats["mean_pnl"],
        "eval_mean_sharpe": eval_stats.get("mean_sharpe", ""),
        "eval_mean_max_dd": eval_stats.get("mean_max_dd", "")
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 8) MAIN TRAINING LOGIC
# ============================================================

def main():

    # Parse CLI arguments
    args = parse_arguments()

    # Prepare folder structure for this run
    timestamp, run_dir, tb_dir, model_dir = create_run_directories(args)

    # Load + split data
    df = load_data(args)
    train_df, val_df, test_df = split_data(df, args.train_ratio, args.val_ratio)

    # Create train + validation environments
    train_env, val_env = create_environments(train_df, val_df, args, run_dir)

    # Build PPO model
    model = create_ppo_model(train_env, tb_dir, args)

    # Validation callback (best model tracking)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=f"{run_dir}/",
        log_path=f"{run_dir}/",
        eval_freq=5000,
        deterministic=True,
        verbose=1
    )

    # Train PPO model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    # Save final trained model
    model_name = f"{args.symbol}_{args.timeframe}_final"
    model_path = f"{model_dir}/{model_name}"
    model.save(model_path)

    # Save config for reproducibility
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Evaluate model on unseen test set
    eval_stats = evaluate_model(model, test_df, args)

    # Append to simple global CSV
    append_results_to_csv(args, eval_stats, model_path)

    # Print metrics in a stable format
    final_equity = args.initial_balance + eval_stats["mean_pnl"]
    print("\n====== EVALUATION SUMMARY ======")
    print(f"Final equity  : {final_equity:.2f}")
    print(f"ROI           : {eval_stats['mean_roi']:.2f}%")
    print(f"PnL           : {eval_stats['mean_pnl']:.2f}")
    print(f"Sharpe        : {eval_stats['mean_sharpe']:.4f}")
    print(f"Max drawdown  : {eval_stats['mean_max_dd'] * 100:.2f}%")
    print("================================\n")

    # Save evaluation results (JSON)
    with open(f"{run_dir}/eval_results.json", "w") as f:
        json.dump(eval_stats, f, indent=4)

    # Append detailed summary line into logs/single_agent/results.csv
    append_results(args, eval_stats, model_path)

    print("🎉 Training complete!")
    print(f"Model saved → {model_path}.zip")
    print(f"Eval ROI: {eval_stats['mean_roi']:.2f}%")
    print(f"Logs saved under: {run_dir}/\n")


if __name__ == "__main__":
    main()