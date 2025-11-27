import numpy as np
import pandas as pd

from src.env.trading_env import TradingEnv


def make_dummy_feature_df(n_rows: int = 200) -> pd.DataFrame:
    """
    Create a dummy dataframe that matches the schema of our real feature data.
    This is critical so TradingEnv.feature_cols aligns with the test df.
    """
    ts = pd.date_range("2020-01-01", periods=n_rows, freq="h")

    rng = np.random.default_rng(42)

    # Base OHLC
    open_ = 10000 + rng.normal(0, 50, size=n_rows)
    high_ = open_ + rng.normal(10, 20, size=n_rows).clip(min=1)
    low_  = open_ - rng.normal(10, 20, size=n_rows).clip(min=1)
    close = open_ + rng.normal(0, 30, size=n_rows)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high_,
            "low": low_,
            "close": close,
            "volume": rng.normal(100, 10, size=n_rows).clip(min=1),
        }
    )

    # Dummy "true range" and ATR_14 (not exact, but good enough for shape tests)
    tr = (df["high"] - df["low"]).abs()
    df["ATR_14"] = tr.rolling(window=14, min_periods=1).mean()

    # Raw features used in your real pipeline
    df["log_return"] = np.log(df["close"]).diff().fillna(0)
    df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["RSI_14"] = rng.uniform(0, 100, size=n_rows)
    df["volatility_20"] = (
        df["log_return"].rolling(20, min_periods=1).std().fillna(0.001)
    )

    # Normalised features (just fake standardised values)
    df["log_return_norm"] = (df["log_return"] - df["log_return"].mean()) / (
        df["log_return"].std() + 1e-8
    )
    df["EMA_10_norm"] = (df["EMA_10"] - df["EMA_10"].mean()) / (
        df["EMA_10"].std() + 1e-8
    )
    df["EMA_50_norm"] = (df["EMA_50"] - df["EMA_50"].mean()) / (
        df["EMA_50"].std() + 1e-8
    )
    df["RSI_14_norm"] = (df["RSI_14"] - df["RSI_14"].mean()) / (
        df["RSI_14"].std() + 1e-8
    )
    df["volatility_20_norm"] = (
        df["volatility_20"] - df["volatility_20"].mean()
    ) / (df["volatility_20"].std() + 1e-8)

    return df


def main():
    df = make_dummy_feature_df(n_rows=200)

    env = TradingEnv(
        df=df,
        window_size=10,
        initial_balance=10_000.0,
        transaction_cost=0.0004,
        leverage=5.0,
        max_episode_steps=200,
    )

    # --- Reset checks ---
    obs, info = env.reset()
    print("Obs shape after reset:", obs.shape)
    assert obs.shape == (10, env.n_features)
    assert np.isfinite(obs).all(), "Obs contains NaN/inf after reset"

    # --- Step checks ---
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (10, env.n_features)
        assert np.isfinite(obs).all(), "Obs contains NaN/inf during step"
        assert np.isfinite(reward), "Reward is NaN/inf"
        assert np.isfinite(info["equity"]), "Equity is NaN/inf"

        if terminated or truncated:
            break

    print("Environment shape & finiteness checks: ✅ passed")


if __name__ == "__main__":
    main()
