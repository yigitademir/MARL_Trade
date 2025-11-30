import yaml
import os
import ccxt
import pandas as pd
from datetime import datetime, timezone
import tqdm
from src.utils.data_fetcher import fetch_ohlcv
from src.utils.data_checker import check_data_file
from src.utils.features import generate_all_features

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

symbols = config["symbols"]
timeframes = config["timeframes"]
start_date = config["start_date"]
exchange_name = config.get("exchange")
limit = config.get("limit")
raw_output_path = config.get("raw_output_path")
feature_output_path = config.get("feature_output_path")
file_format = config.get("format")

# Start date in milliseconds
start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

# Initialize exchange
exchange_class = getattr(ccxt, exchange_name)
exchange = exchange_class()

# Ensure output directory exists
os.makedirs(raw_output_path, exist_ok=True)
os.makedirs(feature_output_path, exist_ok=True)

# Loop over symbols and timeframes
for symbol in symbols:
    for tf in timeframes:
        filename = symbol.replace("/", "") + f"_{tf}.{file_format}"
        filepath = os.path.join(raw_output_path, filename)
        
        print(f"Fetching {symbol} @ {tf}")

        # Determine start timestamp
        if os.path.exists(filepath):
            print(f"Resuming from existing file {filename}")
            df_existing = pd.read_parquet(filepath) if file_format == "parquet" else pd.read_csv(filepath)
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
            last_ts = int(df_existing["timestamp"].max().timestamp() * 1000) + 1
        else:
            df_existing = None
            last_ts = start_ts

        # Fetch from last timestamp
        df_new = fetch_ohlcv(exchange, symbol, tf, since=last_ts, limit=limit, show_progress=True)

        # Flag for feature generating
        should_generate_features = False

        if df_new.empty:
            print("No new data to fetch.")
            df = df_existing

            # Check for processed file
            symbol_clean = symbol.replace("/","")
            features_filename = f"{symbol_clean}_{tf}_features.{file_format}"
            features_path = os.path.join(feature_output_path, features_filename)

            if not os.path.exists(features_path):
                print("No features file found, will generate from existing file.")
                should_generate_features = True
            else:
                print("Feature file already exists.")

        else:
            # Merge new data with existing
            if df_existing is not None:
                df = pd.concat([df_existing, df_new]).drop_duplicates("timestamp").sort_values("timestamp")
            else:
                df = df_new

            # Trim dates before start date
            start_date_ts = pd.to_datetime(start_date)
            df = df[df["timestamp"] >= start_date_ts]

            # Save file
            if file_format == "csv":
                df.to_csv(filepath, index=False)
            else:
                df.to_parquet(filepath, index=False)

            print(f"Saved to {filepath}")
            should_generate_features = True

            # Check time gaps
            tf_minutes = int(tf.replace("m","")) if "m" in tf else int(tf.replace("h","")) * 60
            check_data_file(filepath, expected_timeframe_minutes=tf_minutes)

            # Feature generation
            if should_generate_features and df is not None and not df.empty:
                print("Generating features")
                generate_all_features(df, filename, config)