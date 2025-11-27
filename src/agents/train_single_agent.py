"""
Single-Agent PPO Training Script v3
===========================================
"""

import os
import argparse
import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import glob

from src.env.trading_env import TradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


# ============================================================
# 1) ARGUMENT PARSER
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train single-agent PPO")

    # Data
    parser.add_argument("--timeframe", type=str, default="1h",
                        choices=["5m", "15m", "1h", "4h"])
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--data_path", type=str, default="data/processed")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    # Environment
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--initial_balance", type=float, default=10000.0)
    parser.add_argument("--leverage", type=float, default=5.0)

    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Training
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--save_freq", type=int, default=10000)

    # Output
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models")

    return parser.parse_args()


# ============================================================
# 2) DIRECTORY SETUP
# ============================================================

def create_directories(args):
    os.makedirs("models/single_agents", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

# ============================================================
# 3) DATA LOADING
# ============================================================

def load_data(args):
    filename = f"{args.symbol}_{args.timeframe}_features.parquet"
    filepath = os.path.join(args.data_path, filename)

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} rows")
    print(df.head(3))

    return df


def split_data(df, train_ratio=0.7, val_ratio=0.15):
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

def create_environments(train_df, val_df, args):

    train_env = TradingEnv(
        df=train_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=0.0004,
        leverage=args.leverage
    )
    train_env = Monitor(train_env, filename="logs/training",
                        info_keywords=("roi", "sharpe_ratio", "max_drawdown"))

    val_env = TradingEnv(
        df=val_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=0.0004,
        leverage=args.leverage
    )
    val_env = Monitor(val_env, filename="logs/validation",
                      info_keywords=("roi", "sharpe_ratio", "max_drawdown"))

    return train_env, val_env


# ============================================================
# 5) CALLBACKS
# ============================================================

def setup_callbacks(args, val_env):

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path="models/checkpoints",
        name_prefix=f"{args.symbol}_{args.timeframe}_step"
    )

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="models/best",
        log_path="logs",
        eval_freq=args.save_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1)
    
    return [checkpoint_callback, eval_callback]

def cleanup_checkpoints(pattern="models/checkpoints/*.zip", keep_last=3):
    """
    Deletes old checkpoint files, keeping only the newest N.
    """
    files = sorted(
        glob.glob(pattern),
        key=os.path.getmtime,   # sort by modification time
        reverse=True            # newest first
    )

    if len(files) <= keep_last:
        return  # nothing to delete

    to_delete = files[keep_last:]

    for f in to_delete:
        try:
            os.remove(f)
            print(f"Deleted old checkpoint: {f}")
        except Exception as e:
            print(f"Could not delete checkpoint {f}: {e}")


# ============================================================
# 6) CREATE PPO MODEL
# ============================================================

def create_ppo_model(train_env, args):

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
        policy_kwargs=dict(net_arch=[128, 128, 64]),
        tensorboard_log="logs/tensorboard",
        verbose=1,
        seed=42,
        device="auto"
    )

    return model


# ============================================================
# 7) TRAINING
# ============================================================

def train(model, args, callbacks):

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # Cleanup checkpoints after each learn cycle
    cleanup_checkpoints("models/checkpoints/*.zip", keep_last=3)

    return model


# ============================================================
# 8) SAVE MODEL
# ============================================================

def save_model(model, args):
    clean_name = f"{args.symbol}_{args.timeframe}_final"
    save_path = f"models/single_agents/{clean_name}"
    model.save(save_path)

    # Save hyperparameters
    with open(f"{save_path}_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    return save_path


# ============================================================
# 9) EVALUATION
# ============================================================

def evaluate_model(model, test_df, args):

    test_env = TradingEnv(
        df=test_df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=0.0004,
        leverage=args.leverage
    )

    n_eval = 10
    rewards = []
    rois = []
    pnls = []
    sharpes = []
    dds = []

    for _ in range(n_eval):
        obs, info = test_env.reset()
        done = False
        total_r = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_r += reward

        rewards.append(total_r)
        rois.append(info["roi"])
        pnls.append(info["balance"] - args.initial_balance)
        sharpes.append(info["sharpe_ratio"])
        dds.append(info["max_drawdown"])

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_roi": float(np.mean(rois)),
        "mean_pnl": float(np.mean(pnls)),
        "mean_sharpe": float(np.mean(sharpes)),
        "mean_max_dd": float(np.mean(dds))
    }


# ============================================================
# 10) LOGGING RESULTS
# ============================================================

def append_results_to_csv(args, eval_stats, save_path):
    """
    Append single-run training + eval summary to logs/results.csv

    args: argparse.Namespace from parse_arguments()
    eval_stats: dict returned by evaluate_model(...)
    save_path: path without .zip (where model was saved)
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
        # training stats – fill later if/when we log them
        "train_episodes": "",
        "train_reward_max": "",
        "train_reward_min": "",
        "train_reward_mean": "",
        "train_length_mean": "",
        # eval stats
        "eval_mean_reward": eval_stats["mean_reward"],
        "eval_mean_roi": eval_stats["mean_roi"],
        "eval_mean_pnl": eval_stats["mean_pnl"],
        # optional fields (will be NaN in summary if missing)
        "eval_mean_sharpe": "",
        "eval_mean_max_dd": "",
    }

    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 11) MAIN
# ============================================================

def main():

    args = parse_arguments()
    create_directories(args)

    df = load_data(args)
    train_df, val_df, test_df = split_data(df, args.train_ratio, args.val_ratio)

    train_env, val_env = create_environments(train_df, val_df, args)

    callbacks = setup_callbacks(args, val_env)
    model = create_ppo_model(train_env, args)
    model = train(model, args, callbacks)

    # Rename best model file if it exists
    default_best = "models/best/best_model.zip"
    clean_best = f"models/best/{args.symbol}_{args.timeframe}_best.zip"

    if os.path.exists(default_best):
        os.replace(default_best, clean_best)

    save_path = save_model(model, args)
    eval_stats = evaluate_model(model, test_df, args)


    append_results_to_csv(args, eval_stats, save_path)

    print("\n🎉 Training complete!")
    print(f"Model saved → {save_path}")
    print(f"Eval ROI: {eval_stats['mean_roi']:.2f}%\n")


if __name__ == "__main__":
    main()
