import pandas as pd
import numpy as np
import os


def generate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds log returns to dataframe."""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_ema(df: pd.DataFrame, spans=[10, 50]) -> pd.DataFrame:
    """Adds EMA indicators."""
    df = df.copy()
    for span in spans:
        df[f"EMA_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Adds RSI indicator."""
    df = df.copy()
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

    return df


def add_rolling_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Adds rolling volatility using std(log_return)."""
    df = df.copy()
    if "log_return" not in df.columns:
        raise ValueError("log_return column required to compute volatility.")

    df[f"volatility_{period}"] = df["log_return"].rolling(window=period).std()
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Computes ATR(14) using standard True Range.
    """
    df = df.copy()

    df["prev_close"] = df["close"].shift(1)

    df["high_low"] = df["high"] - df["low"]
    df["high_prev"] = (df["high"] - df["prev_close"]).abs()
    df["low_prev"] = (df["low"] - df["prev_close"]).abs()

    df["true_range"] = df[["high_low", "high_prev", "low_prev"]].max(axis=1)

    df["ATR_14"] = df["true_range"].rolling(window=period).mean()

    df.drop(columns=["prev_close", "high_low", "high_prev", "low_prev"], inplace=True)

    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all features after 'volume' using z-score.

    This will automatically create:
        - log_return_norm
        - EMA_10_norm, EMA_50_norm
        - RSI_14_norm
        - volatility_20_norm
        - true_range_norm
        - ATR_14_norm

    We do NOT create ATR_pct or ATR_pct_norm anymore.
    """
    df = df.copy()
    if "volume" not in df.columns:
        raise ValueError("'volume' column not found.")

    start_idx = df.columns.get_loc("volume") + 1
    feature_cols = df.columns[start_idx:]

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_norm"] = (df[col] - mean) / (std + 1e-8)

    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans NaN, inf, extreme values.
    """
    df = df.copy()
    print(" Cleaning features...")

    df = df.ffill()
    df = df.bfill()
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(mean - 10 * std, mean + 10 * std)

    return df


def save_feature_dataset(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_dir: str = "data/processed",
    file_format: str = "parquet"
):
    """Saves features to disk."""
    os.makedirs(output_dir, exist_ok=True)

    symbol_clean = symbol.replace("/", "")
    filename = f"{symbol_clean}_{timeframe}_features.{file_format}"
    filepath = os.path.join(output_dir, filename)

    if file_format == "csv":
        df.to_csv(filepath, index=False)
    else:
        df.to_parquet(filepath, index=False)

    print(f"Features saved to: {filepath}")


def generate_all_features(df: pd.DataFrame, filename: str, config: dict) -> pd.DataFrame:
    """
    Full feature engineering pipeline:

    - log_return
    - EMA(10,50)
    - RSI(14)
    - volatility_20
    - true_range
    - ATR_14
    - *_norm columns (including ATR_14_norm)
    """
    import os
    base = os.path.basename(filename).replace(".parquet", "").replace(".csv", "")
    parts = base.split("_")
    symbol_part = "_".join(parts[:-1])
    tf = parts[-1]

    print(f"Generating features for {symbol_part} {tf}...")

    df = generate_log_returns(df)
    df = add_ema(df, spans=[10, 50])
    df = add_rsi(df, period=14)
    df = add_rolling_volatility(df, period=20)
    df = compute_atr(df, period=14)

    df = normalize_features(df)
    df = clean_features(df)

    save_feature_dataset(
        df,
        symbol=symbol_part,
        timeframe=tf,
        output_dir=config.get("feature_output_path", "data/processed"),
        file_format=config.get("format", "parquet")
    )

    return df


def inspect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Print summary of outliers in normalized columns.
    """
    norm_cols = [col for col in df.columns if col.endswith("_norm")]

    for col in norm_cols:
        outliers = df[(df[col] > threshold) | (df[col] < -threshold)]
        count = len(outliers)
        pct = (count / len(df)) * 100
        print(f"{col:20} -> {count:5} outliers ({pct:.2f}%)")