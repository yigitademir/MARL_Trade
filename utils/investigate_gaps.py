# investigate_gaps.py

import pandas as pd
import ccxt
from datetime import datetime, timezone

# === Step 1: Load your data ===
file_path = "data/BTCUSDT_15m.parquet"
df = pd.read_parquet(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"\n✅ Loaded {len(df)} rows from {file_path}")
print(f"🕒 Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# === Step 2: Compute time gaps ===
df = df.sort_values("timestamp")
df["gap"] = df["timestamp"].diff()

# Expected delta for 15-minute candles
expected_delta = pd.Timedelta(minutes=15)
gap_violations = df[df["gap"] != expected_delta]

print(f"\n⚠️ Found {len(gap_violations)} time gaps")

# === Step 3: Show top 5 gap violations ===
print("\n🔍 Top 5 gap violations:")
print(gap_violations[["timestamp", "gap"]].head())

# === Step 4: Investigate the first gap ===
if not gap_violations.empty:
    # Skip the first gap if it’s just the first row (NaT)
    gap_violations = gap_violations[gap_violations.index != 0]

    if not gap_violations.empty:
        gap_row = gap_violations.iloc[0]
        gap_index = gap_row.name

    if gap_index == 0:
        print("\n⚠️ First row has no previous timestamp — skipping this gap.")
    else:
        prev_timestamp = df.loc[gap_index - 1, "timestamp"]
        curr_timestamp = gap_row["timestamp"]

        print(f"\n🔬 Investigating first gap:")
        print(f"  Start: {prev_timestamp}")
        print(f"  End:   {curr_timestamp}")
        print(f"  Missing approx. {(curr_timestamp - prev_timestamp) / expected_delta:.0f} candles")

        # === Step 5: Try refetching from Binance ===
        exchange = ccxt.binance()
        symbol = "BTC/USDT"
        timeframe = "15m"

        since_ts = int(prev_timestamp.replace(tzinfo=timezone.utc).timestamp() * 1000)
        limit = 100  # fetch up to 100 candles

        print(f"\n📡 Fetching from Binance between {prev_timestamp} and {curr_timestamp}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ts, limit=limit)

        df_gap = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_gap["timestamp"] = pd.to_datetime(df_gap["timestamp"], unit="ms")

        # Show only relevant part
        df_gap_filtered = df_gap[df_gap["timestamp"] < curr_timestamp]

        print(f"\n🔎 Candles returned by Binance:")
        print(df_gap_filtered)

        print(f"\n🧮 Candles received: {len(df_gap_filtered)}")
        print(f"🧮 Expected: {(curr_timestamp - prev_timestamp) / expected_delta:.0f}")