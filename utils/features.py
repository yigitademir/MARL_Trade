import pandas as pd
import numpy as np
import os

def generate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log returns to dataframe.
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df

def add_ema(df: pd.DataFrame, spans = [10,50]) -> pd.DataFrame:
    """
    Adds exponential moving average to the dataframe
    Default spans: 10 and 50 periods
    """
    df = df.copy()
    for span in spans:
        df[f"EMA_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds RSI to the dataframe
    Default period is 14
    """
    df = df.copy()
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10) # to avoid div by 0
    df[f"RSI_{period}"] = 100 - (100 / (1+rs))
    
    return df

def add_rolling_volatility(df:pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Adds rolling volatility to the dataframe using standard deviation of log returns.
    Default period is 20
    !! Requires "log_return" column 
    """
    df = df.copy()
    if "log_return" not in df.columns:
        raise ValueError("log_return column required to compute volatility.")

    df[f"volatility_{period}"] = df["log_return"].rolling(window=period).std()

    return df

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all feature columns(comes after "volume" column).
    Applies Z-score normalization
    """

    df = df.copy()
    if "volume" not in df.columns:
        raise ValueError("'volume' column not found.")
    
    # Find feature columns
    start_idx = df.columns.get_loc("volume") + 1
    feature_cols = df.columns[start_idx:]

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_norm"] = (df[col] - mean) / (std + 1e-8) # Avoid div by 0

    return df

def save_feature_dataset(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_dir: str = "data/processed",
    file_format: str = "parquet") -> None:
    """
    Saves the feature-enhanced DataFrame to disk.
    File name: SYMBOL_TIMEFRAME_features.{parquet/csv}
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    symbol_clean = symbol.replace("/", "")
    filename = f"{symbol_clean}_{timeframe}_features.{file_format}"
    filepath = os.path.join(output_dir, filename)

    if file_format == "csv":
        df.to_csv(filepath, index=False)
    else:
        df.to_parquet(filepath, index=False)

    print(f"Features saved to: {filepath}")

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean features: handle NaN, inf, and extreme values.
    """
    df = df.copy()
    
    print(" Cleaning features...")
    
    # 1. Forward fill NaN values (use previous valid value)
    df = df.ffill()
    
    # 2. Backward fill remaining NaN (for start of series)
    df = df.bfill()
    
    # 3. Replace any remaining NaN with 0
    df = df.fillna(0)
    
    # 4. Replace infinity with large but finite values
    df = df.replace([np.inf, -np.inf], [1e10, -1e10])
    
    # 5. Clip extreme values (prevent explosions)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'timestamp':
            # Clip to reasonable range
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                # Keep values within ±10 standard deviations
                df[col] = df[col].clip(mean - 10*std, mean + 10*std)
    
    # 6. Final safety check
    nan_count = df.isnull().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"NaN remaining: {nan_count}")
    print(f"Inf remaining: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print(" WARNING: Still have invalid values, replacing with 0")
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
    
    print("Features cleaned")
    
    return df

def generate_all_features(df: pd.DataFrame, filename: str, config: dict) -> pd.DataFrame:
    """
    Full feature pipeline with cleaning!
    """
    import os
    
    # Parse symbol and timeframe from filename
    base = os.path.basename(filename).replace(".parquet", "").replace(".csv", "")
    parts = base.split("_")
    symbol_part = "_".join(parts[:-1])
    tf = parts[-1]
    
    print(f"Generating features for {symbol_part} {tf}...")
    
    # Generate features
    df = generate_log_returns(df)
    df = add_ema(df, spans=[10, 50])
    df = add_rsi(df, period=14)
    df = add_rolling_volatility(df, period=20)
    df = normalize_features(df)
    
    # CRITICAL: Clean features!
    df = clean_features(df)
    
    # Save
    save_feature_dataset(
        df,
        symbol=symbol_part,
        timeframe=tf,
        output_dir=config.get("feature_output_path", "data/processed"),
        file_format=config.get("format", "parquet")
    )
    
    return df
def inspect_outliers(df:pd.DataFrame, threshold: float=3.0) -> pd.DataFrame:
    """
    Prints a summary of outliers falls outside of threshold for each normalized column
    """

    # Select normalized columns
    norm_cols= [col for col in df.columns if col.endswith("_norm")]

    for col in norm_cols:
        outliers = df[(df[col] > threshold) | (df[col] < -threshold) ]
        count = len(outliers)
        total = len(df)
        pct = (count/total) * 100
        print(f"{col:20} -> {count:5} outliers ({pct:.2f}%)")