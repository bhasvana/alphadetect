import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "SBIN.NS",
]

WINDOW = 20
RETURN_Z_THRESH = 2.0
VOLUME_Z_THRESH = 1.5

end = datetime.today()
start = end - timedelta(days=2 * 365)

raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)

all_anomalies = []

for ticker in TICKERS:
    close = raw["Close"][ticker].dropna()
    volume = raw["Volume"][ticker].dropna()

    idx = close.index.intersection(volume.index)
    close = close.loc[idx]
    volume = volume.loc[idx]

    df = pd.DataFrame({"Close": close, "Volume": volume})
    df["Open"] = raw["Open"][ticker].reindex(idx)
    df["High"] = raw["High"][ticker].reindex(idx)
    df["Low"] = raw["Low"][ticker].reindex(idx)

    df["daily_return"] = df["Close"].pct_change()

    df["roll_mean_return"] = df["daily_return"].rolling(WINDOW).mean()
    df["roll_std_return"] = df["daily_return"].rolling(WINDOW).std()
    df["return_zscore"] = (df["daily_return"] - df["roll_mean_return"]) / df["roll_std_return"]

    df["roll_mean_volume"] = df["Volume"].rolling(WINDOW).mean()
    df["roll_std_volume"] = df["Volume"].rolling(WINDOW).std()
    df["volume_zscore"] = (df["Volume"] - df["roll_mean_volume"]) / df["roll_std_volume"]

    df.dropna(subset=["return_zscore", "volume_zscore"], inplace=True)

    anomalies = df[(df["return_zscore"].abs() > RETURN_Z_THRESH) & (df["volume_zscore"] > VOLUME_Z_THRESH)].copy()
    anomalies["ticker"] = ticker
    all_anomalies.append(anomalies)

    print(f"{ticker}: {len(anomalies)} anomaly days")

result = pd.concat(all_anomalies)
result.index.name = "date"
result = result.reset_index()[
    ["ticker", "date", "Open", "High", "Low", "Close", "Volume",
     "daily_return", "return_zscore", "volume_zscore"]
]
result.to_csv("anomalies.csv", index=False)
print(f"\nTotal anomalies saved to anomalies.csv: {len(result)}")
