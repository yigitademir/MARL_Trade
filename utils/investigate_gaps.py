import os
import pandas as pd
import ccxt
from datetime import datetime, timezone

# === Config ===
DATA_FOLDER = "data"
EXPECTED_DELTAS = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}

def get_expected_delta_from_filename(filename):
    for tf in EXPECTED_DELTAS:
        if f"_{tf}" in filename:
            return EXPECTED_DELTAS[tf]
    return None

def investigate_file(filepath):
    print(f"\n📂 Investigating: {filepath}")

    # Load data
    df = pd.read_parquet(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    print(f"✅ Loaded {len(df)} rows")
    print(f"🕒 Range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Compute time deltas
    df["gap"] = df["timestamp"].diff()

    expected_delta = get_expected_delta_from_filename(filepath)
    if expected_delta is None:
        print("⚠️ Could not determine expected delta from filename. Skipping.")
        return

    gap_violations = df[df["gap"] != expected_delta]

    if len(gap_violations) == 0:
        print("✅ No time gaps detected.")
        return

    print(f"⚠️ {len(gap_violations)} time gaps found.")
    print("🔍 Top 5 gap violations:")
    print(gap_violations[["timestamp", "gap"]].head())

    # Skip NaT row
    gap_violations = gap_violations[gap_violations.index != 0]
    if gap_violations.empty:
        print("ℹ️ Only first row has gap (NaT), no real gaps to inspect.")
        return

    # Investigate the first real gap
    gap_row = gap_violations.iloc[0]
    gap_index = gap_row.name

    if gap_index == 0:
        print("⚠️ First row — no previous candle to compare. Skipping.")
        return

    prev_ts = df.loc[gap_index - 1, "timestamp"]
    curr_ts = gap_row["timestamp"]

    print(f"\n🔬 Investigating first gap:")
    print(f"  Start: {prev_ts}")
    print(f"  End:   {curr_ts}")
    print(f"  Missing approx. {(curr_ts - prev_ts) / expected_delta:.0f} candles")

    # Fetch from Binance to see if missing candles are available
    exchange = ccxt.binance()
    symbol = filepath.split("/")[-1].split("_")[0]
    symbol = symbol.replace("USDT", "/USDT")
    timeframe = [tf for tf in EXPECTED_DELTAS if f"_{tf}" in filepath][0]

    since_ts = int(prev_ts.replace(tzinfo=timezone.utc).timestamp() * 1000)

    print(f"\n📡 Fetching from Binance: {symbol} @ {timeframe}")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ts, limit=100)

    df_gap = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_gap["timestamp"] = pd.to_datetime(df_gap["timestamp"], unit="ms")

    df_gap_filtered = df_gap[df_gap["timestamp"] < curr_ts]

    print(f"\n🔎 Binance returned {len(df_gap_filtered)} candles (expected: {(curr_ts - prev_ts) / expected_delta:.0f})")
    print(df_gap_filtered[["timestamp", "open", "close"]])

# === Run for all .parquet files ===
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".parquet") and file != ".gitkeep":
        filepath = os.path.join(DATA_FOLDER, file)
        investigate_file(filepath)