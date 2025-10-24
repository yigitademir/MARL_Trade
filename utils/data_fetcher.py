import ccxt
import pandas as pd
import time

def fetch_ohlcv(exchange, symbol, timeframe, since=None, limit=1000):
    all_data = []
    
    while True:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not data:
            break
        all_data.extend(data)
        since = data[-1][0] + 1 # Update timestamp for next fetch
        time.sleep(exchange.rateLimit / 1000) # Respect rate limit
        if len(data) < limit:
            break
    
    df = pd.DataFrame(all_data, columns = ["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df