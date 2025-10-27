import pandas as pd
from utils.features import generate_all_features, inspect_outliers

df = pd.read_parquet("data/BTCUSDT_15m.parquet")
df = generate_all_features(df)

print(df.filter(like="_norm").tail(10))
inspect_outliers(df)
