import pandas as pd
import os

def check_data_file(filepath, expected_timeframe_minutes=None):
    print(f"Checking file {filepath}")
    
    # Read file
    try:
        df = pd.read_parquet(filepath)

    except Exception as e:
        print("Failed to load file.")
        return
    
    # Check columns
    required_cols = {"timestamp", "open", "high", "close", "low", "volume"}
    if not required_cols.issubset(df.columns):
        print(f"Missing columns: {required_cols-set(df.columns)}")
        return
    
    # Check NaN values
    if df.isnull().sum().any():
        print("Null values detected:" ,df.isnull().sum())

    # Check for duplicates
    if df.duplicated(subset="timestamp").any():
        print("Duplicate timestamps:", df.duplicated(subset="timestamp").sum())

    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Check if timeframe correct in data
    df = df.sort_values("timestamp")
    delta = df["timestamp"].diff().dropna()

    if expected_timeframe_minutes:
        expected = pd.Timedelta(minutes=expected_timeframe_minutes)
        gap_violations = delta[delta != expected]

    if not gap_violations.empty:
        print(f"Time gaps detected")
    else:
        print(f"All time intervals are consistent, {expected_timeframe_minutes}")
