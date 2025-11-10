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

# === Data Handling ===
def load_data(args):
    """
    Load processed feature data from disk.
    
    1. Constructs the filename from symbol and timeframe
    2. Loads the parquet file
    3. Validates the data
    4. Returns the dataframe
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        pd.DataFrame: Loaded feature data
    """

    # Construct filename
    filename = f"{args.symbol}_{args.timeframe}_features.parquet"
    filepath = os.path.join(args.data_path, filename)

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"File: {filepath}")

    # Check if File exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}\n Did you run main.py?")
    
    # Load data
    df = pd.read_parquet(filepath)
    print("Data Loaded")
    print(f"\n Data Statistics:") 
    print(f"Total candles: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"Features: {len(df.columns)} columns")
    print(f"\nFirst 3 rows:")
    print(df.head(3))

    return df

def split_data(df, train_rate = 0.7, val_ratio = 0.15):
    """
    Split data into train, validation, and test sets.
    
    IMPORTANT: We use TIME-BASED splitting, NOT random!
    
    Why?
    - Random split = Data leakage (future info in training)
    - Time-based = Realistic (train on past, test on future)
    
    Args:
        df: Full dataset
        train_ratio: Proportion for training (default: 0.7 = 70%)
        val_ratio: Proportion for validation (default: 0.15 = 15%)
        # test_ratio is implicit: 1 - train - val = 0.15 (15%)
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """