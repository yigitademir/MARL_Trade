"""
Single-Agent PPO Training Script
=================================
This script trains a single PPO agent on one timeframe.

Usage:
    python agents/train_single_agent.py
"""

import os
import argparse
import json
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

def split_data(df, train_ratio = 0.7, val_ratio = 0.15):
    """
    Split data into train, validation, and test sets.
    
    IMPORTANT: TIME-BASED splitting, NOT random!
    
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

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop= True)

    # Calculate split indices
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    # Split chronogically
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    print("\n" + "="*70)
    print("DATA SPLITTING")
    print("="*70)
    print(f"Total samples: {n:,}")
    print(f"\n Training set:")
    print(f"Size: {len(train_df):,} ({train_ratio*100:.0f}%)")
    print(f"Date range: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    
    print(f"\n Validation set:")
    print(f"Size: {len(val_df):,} ({val_ratio*100:.0f}%)")
    print(f"Date range: {val_df['timestamp'].min()} → {val_df['timestamp'].max()}")
    
    print(f"\n Test set:")
    print(f"Size: {len(test_df):,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    print(f"Date range: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")
    
    return train_df, val_df, test_df

# Creating Environments
def create_environments(train_df, val_df, args):
    """
    Create training and validation environments.
    
    Why two environments?
    - Training env: Agent learns here
    - Validation env: Agent is tested here
    
    Args:
        train_df: Training data
        val_df: Validation data
        args: Configuration arguments
        
    Returns:
        tuple: (train_env, val_env)
    """
    print("\n" + "="*70)
    print("CREATING ENVIRONMENTS")
    print("="*70)

    # Create training environment
    print("Training Environment:")
    train_env = TradingEnv(
        df= train_df,
        window_size= args.window_size,
        initial_balance= args.initial_balance,
        transaction_cost= 0.001 # 0.1% fee
    )

    # Monitor logging
    train_env = Monitor(
        train_env,
        filename= "logs/training",
        info_keywords= ("roi", "sharpe_ratio", "max_drawdown")
    )

    print(f"Window size:{args.window_size} candles")
    print(f"Initial balance: ${args.initial_balance:,.2f}")
    print(f"Transaction cost: 0.1%")
    print(f"Action space: {train_env.action_space}")
    print(f"Observation space: {train_env.observation_space.shape}")

    # Create validation environment
    print("Validation environment:")
    val_env = TradingEnv(
        df= val_env,
        window_size= args.window_size,
        initial_balance= args.initial_balance,
        transaction_cost= 0.001
    )

    # Monitor logging
    val_env = Monitor(
        val_env,
        filename= "logs/validation",
        info_keywords= ("roi", "sharpe_ratio", "max_drawdown")
    )

    print(f"Window size: {args.window_size} candles")
    print(f"Initial balance: ${args.initial_balance:,.2f}")
    print(f"Same configuration as training")
    
    return train_env, val_env

# === PPO Configuration ===
def setup_callbacks(args, val_env):
    """
    Set up training callbacks for checkpointing and evaluation.
    
    Callbacks are functions that run during training:
    - CheckpointCallback: Saves model every N steps (crash recovery)
    - EvalCallback: Tests agent on validation data (detect overfitting)
    
    Args:
        args: Configuration arguments
        val_env: Validation environment for testing
        
    Returns:
        list: List of callback objects
    """
    print("\n" + "="*70)
    print("SETTING UP CALLBACKS")
    print("="*70)

    # Checkpoint call back
    # Saves model regularly during training (every save_freq steps)
    checkpoint_callback = CheckpointCallback(
        save_freq= args.save_freq,
        save_path= f"{args.output_dir}/checkpoints",
        name_prefix= f"{args.symbol}_{args.timeframe}",
        save_replay_buffer= False,
        save_vecnormalize= False
    )

    print(f"Checkpoint Callback:")
    print(f"Save every: {args.save_freq:,} steps")
    print(f"Save path: {args.output_dir}/checkpoints/")

    # Evaluation Check
    # Tests agent on validation data during training
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path= f"{args.output_dir}/best",
        log_path= "logs",
        eval_freq= args.save_freq, # Evaluate as often we save
        n_eval_episodes= 5, # 5 episodes per evaluation
        deterministic= True, # greedy policy for fair comparison
        render= False,
        verbose= 1
    )

    print(f"\nEvaluation Callback:")
    print(f"Evaluate every: {args.save_freq:,} steps")
    print(f"Evaluation episodes: 5")
    print(f"Best model saved to: {args.output_dir}/best/")
    print(f"Mode: Deterministic (no exploration)")

    return [checkpoint_callback, eval_callback]

def create_ppo_model(train_env, args):
    """
    Create and configure the PPO model.
    
    This is where we set all the hyperparameters:
    - Learning rate
    - Batch size
    - Number of steps
    - etc.
    
    Args:
        train_env: Training environment
        args: Configuration arguments
        
    Returns:
        PPO: Configured PPO model ready for training
    """
    print("\n" + "="*70)
    print("CREATING PPO MODEL")
    print("="*70)

    # PPO Model with all hyperparameters
    model = PPO(
        policy= "MlpPolicy", # Multi layer Perceptron (standard NN)
        env= train_env,
        learning_rate= args.learning_rate,
        n_steps= args.n_steps,
        batch_size= args.batch_size,
        n_epochs= args.n_epochs,
        gamma= args.gamma,
        gae_lambda= 0.95, # standard GAE value
        clip_range= 0.2, # PPO clipping range
        clip_range_vf= None, # No value function clipping
        ent_coef= 0.01, # Entropy coefficient (exploring)
        vf_coef= 0.5, # value function coefficient
        max_grad_norm= 0.5, # Gradient clipping
        use_sde= False, # Don't use State Dependent Exploration
        sde_sample_freq= -1,
        target_kl= None, # No KL divergence constraint
        tensorboard_log= "logs/tensorboard",
        policy_kwargs= dict(net_arch = [128, 128, 64]), # Network architecture 3 hidden layers
        verbose= 1, # Print training info
        seed= 42,
        device= "auto"  # Use GPU if available, else CPU
    )

    print("\n PPO Configuration:")
    print(f"Policy: MlpPolicy")
    print(f"Network architecture: [128, 128, 64]")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Rollout steps: {args.n_steps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs per update: {args.n_epochs}")
    print(f"Gamma (discount): {args.gamma}")
    print(f"Clip range: 0.2")
    print(f"Entropy coefficient: 0.01")
    print(f"Device: {model.device}")
    
    print(f"\n Model Size:")
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

# === Training ===
def train(model, args, callbacks):
    """
    Train the PPO agent.
    
    This is where the magic happens!
    
    Args:
        model: Configured PPO model
        args: Configuration arguments
        callbacks: List of callbacks (checkpoint, eval)
    """
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Expected duration: ~{args.total_timesteps // 2048 * 30:.0f} seconds")
    print(f"\n Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPress Ctrl+C to stop training...")
    print("="*70 + "\n")

    # Training
    try:
        model.learn(
            total_timestamps = args.total_timestamps,
            callback= callbacks,
            progress_bar = True
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("TRAINING INTERRUPTED BY USER")
        print("="*70)
        print("Model will be saved with current progress...")
    
    except Exception as e:
        print("\n" + "="*70)
        print("TRAINING FAILED")
        print("="*70)
        print(f"Error: {e}")
        raise
    
    return model

# === Saving ===
def save_model(model, args):
    """
    Save the final trained model.
    
    Args:
        model: Trained PPO model
        args: Configuration arguments
    """
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)

    # Generate model name
    if args.model_name:
        model_name = args.model_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.symbol}_{args.time_frame}_{timestamp}"

    # Save 
    save_path = f"{args.output_dir}/final/{model_name}"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Save hyperparameters
    hyperparams = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "window_size": args.window_size,
        "initial_balance": args.initial_balance,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "total_timesteps": args.total_timesteps,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(f"{save_path}_config.json", 'w') as f:
        json.dump(hyperparams, f, indent= 4)

    print(f"Configuration saved to {save_path}_config.json")

    return save_path

def evaluate_model(model, test_df, args):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PPO model
        test_df: Test dataset
        args: Configuration arguments
    """
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    # Create test environment
    test_env = TradingEnv(
        df= test_df,
        window_size= args.window_size,
        initial_balance= args.initial_balance,
        transaction_cost= 0.001
    )

    # Run evaluation episodes
    n_eval_episodes = 10
    episode_rewards = []
    episode_rois = []
    episode_sharpes = []
    episode_drawdowns = []

    print(f"Running {n_eval_episodes} evaluation episodes...")

    for episode in range(n_eval_episodes):
        obs, info = test_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ =model.predict(obs, deterministic = True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_rois.append(info["roi"])

        print(f"  Episode {episode+1}/{n_eval_episodes}: "
              f"Reward={episode_reward:.2f}, ROI={info['roi']:.2f}%")
        

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_roi = np.mean(episode_rois)
    std_roi = np.std(episode_rois)

    print(f"\nTest Set Results:")
    print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Mean ROI: {mean_roi:.2f}% ± {std_roi:.2f}%")
    print(f"   Best Episode: {max(episode_rewards):.2f} (ROI: {max(episode_rois):.2f}%)")
    print(f"   Worst Episode: {min(episode_rewards):.2f} (ROI: {min(episode_rois):.2f}%)")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_roi": mean_roi,
        "std_roi": std_roi,
        "episode_rewards": episode_rewards,
        "episode_rois": episode_rois
    }


# === Main Function ===
def main():
    """
    Main training pipeline.
    
    This function orchestrates the entire training process:
    1. Parse arguments
    2. Create directories
    3. Load and split data
    4. Create environments
    5. Set up callbacks
    6. Create PPO model
    7. Train
    8. Save
    9. Evaluate
    """
    print("\n" + "="*70)
    print("🎯 SINGLE-AGENT PPO TRAINING PIPELINE")
    print("="*70)

    # Step 1
    args = parse_arguments()

    # Step 2
    create_directories(args)

    # Step 3
    df = load_data(args)
    train_df, val_df, test_df = split_data(df)

    # Step 4
    train_env, val_env = create_environments(train_df, val_df, args)

    # Step 5
    callbacks = setup_callbacks(args, val_env)

    # Step 6
    model = create_ppo_model(train_env, args)

    # Step 7
    model = train(model, args, callbacks)

    # Step 8
    save_path = save_model(model, args)

    # Step 9
    results = evaluate_model(model, test_df, args)

    # Final summary
    print("\n" + "="*70)
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print("="*70)
    print(f"Model saved: {save_path}.zip")
    print(f"Test ROI: {results['mean_roi']:.2f}%")
    print(f"\nTo view training progress:")
    print(f"   tensorboard --logdir=logs/tensorboard")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()