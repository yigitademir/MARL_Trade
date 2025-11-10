"""
Single-Agent PPO Training Script
=================================
This script trains a single PPO agent on one timeframe.

Usage:
    python agents/train_single_agent.py
"""

import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.trading_env import TradingEnv
# ---- RL ----
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


# === Configuration ===
def parse_arguments():
    """
    Parse command line arguments for flexible training
    """
    parser = argparse.ArgumentParser(description="Train single agent PPO")

    # Data arguments
    parser.add_argument("--timeframe",type=str,default="1h",
        choices=["5m", "15m", "1h", "4h"], help="Timeframe to train on(default 1h)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--data_path", type=str, default="data/processed")

    # Environment arguments
    parser.add_argument("--window_size", type=int, default=10, help="Number of candles to observe(default 10)")
    parser.add_argument("--initial_balance", type=float, default=10000.0)

    # PPO Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Training arguments
    parser.add_argument("--total_timestamps", type=int, default=100000)
    parser.add_argument("--save_freq", type=int, default=10000)

    # Output arguments
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models")

    return parser.parse_args()


def create_directories(args):
    """Create necessary directories for outputs"""

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.output_dir}/best", exist_ok=True)
    os.makedirs(f"{args.output_dir}/final", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)