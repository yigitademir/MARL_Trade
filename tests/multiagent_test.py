# multi_agent/multiagent_test.py

import pandas as pd
from stable_baselines3 import PPO
from multi_agent.backtester import MultiAgentBacktesterV1


def load_models():
    """Load all timeframe agents."""
    return {
        "5m": PPO.load("models/final/BTCUSDT_5m_20251124_104950"),
        "15m": PPO.load("models/final/BTCUSDT_15m_20251124_105309"),
        "1h": PPO.load("models/final/BTCUSDT_1h_20251124_105134"),
        "4h": PPO.load("models/final/BTCUSDT_4h_20251124_105449"),
    }


def load_data():
    """Load all processed feature datasets."""
    return {
        "5m": pd.read_parquet("data/processed/BTCUSDT_5m_features.parquet"),
        "15m": pd.read_parquet("data/processed/BTCUSDT_15m_features.parquet"),
        "1h": pd.read_parquet("data/processed/BTCUSDT_1h_features.parquet"),
        "4h": pd.read_parquet("data/processed/BTCUSDT_4h_features.parquet"),
    }


def main():

    # 1) Load agents
    models = load_models()

    # 2) Load datasets
    data = load_data()

    # 3) Create backtester
    bt = MultiAgentBacktesterV1(
        models=models,
        data=data,
        window_size=10,
        strategy="majority_vote"
    )

    # 4) Run the full trading backtest — Backtester v3 (risk engine)
    summary, equity_df, trade_df = bt.run_trading(
        initial_balance=10_000,
        leverage=5.0,
        fee_rate=0.0004,
        max_steps=None,       # None = full dataset
        base_timeframe="1h",  # always use stable timeframe for price
        verbose=True
    )

    # 5) Print summary cleanly
    print("\nBACKTEST SUMMARY:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # 6) OPTIONAL: Save results for thesis plots
    equity_df.to_csv("multi_agent_equity_curve.csv", index=False)
    trade_df.to_csv("multi_agent_trades.csv", index=False)

    print("\nSaved:")
    print(" - multi_agent_equity_curve.csv")
    print(" - multi_agent_trades.csv")
    print("==============================================")


if __name__ == "__main__":
    main()