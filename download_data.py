import yaml
import os
import ccxt
import pandas as pd
from datetime import datetime, timezone
import tqdm
from utils.data_fetcher import fetch_ohlcv

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

symbols = config["symbols"]
timeframes = config["timeframes"]
exchange_name = config.get("exchange")
limit = config.get("limit")
output_path = config.get("output_path")
file_format = config.get("format")

# Start date in milliseconds
start_ts = int(datetime.strptime(config["start_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

# Initialize exchange
exchange_class = getattr(ccxt, exchange_name)
exchange = exchange_class()

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Loop over symbols and timeframes
for symbol in symbols:
    for tf in timeframes:
        filename = symbol.replace("/", "") + f"_{tf}.{file_format}"
        filepath = os.path.join(output_path, filename)
        
        print(f"Fetching {symbol} @ {tf}")

        # Determine start timestamp
        if os.path.exists(filepath):
            print(f"Resuming from existing file {filename}")
            df_existing = pd.read_parquet(filepath) if file_format == "parquet" else pd.read_csv(filename)
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
            last_ts = int(df_existing["timestamp"].max().timestamp() * 1000) + 1
        else:
            df_existing = None
            last_ts = start_ts

        # Fetch from last timestamp
        df_new = fetch_ohlcv(exchange, symbol, tf, since=last_ts, limit=limit, show_progress=True)

        if df_new.empty:
            print("No new data to fetch.")
            continue

        # Merge new data with existing data
        if df_existing is not None:
            df = pd.concat([df_existing, df_new]).drop_duplicates("timestamp").sort_values("timestamp")
        else:
            df = df_new

        if file_format == "csv":
            df.to_csv(filepath, index=False)
        else:
            df.to_parquet(filepath, index=False)

        print(f"Saved to {filepath}")
