import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "SBIN.NS",
]
FEATURES = [
    "return_zscore", "volume_zscore",
    "price_momentum_5d", "price_momentum_10d",
    "days_since_last_anomaly",
]

# --- Load anomalies ---
anomalies = pd.read_csv("anomalies.csv", parse_dates=["date"])
anomalies["date"] = anomalies["date"].dt.normalize()
anomalies = anomalies.sort_values(["ticker", "date"]).reset_index(drop=True)

# --- Download full price history (extra buffer for 10-day lookback + 3-day forward) ---
end = datetime.today()
start = end - timedelta(days=2 * 365 + 90)
close_all = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
close_all.index = close_all.index.normalize()


def price_at_offset(series, date, offset):
    """Return price at `offset` trading days from `date` (negative = before, positive = after)."""
    loc = series.index.searchsorted(date)
    target = loc + offset
    if target < 0 or target >= len(series):
        return np.nan
    return series.iloc[target]


# --- Feature engineering ---
rows = []
for _, row in anomalies.iterrows():
    ticker = row["ticker"]
    date = pd.Timestamp(row["date"]).normalize()
    p0 = row["Close"]

    s = close_all[ticker].dropna()

    p_5b = price_at_offset(s, date, -5)
    p_10b = price_at_offset(s, date, -10)
    p_3a = price_at_offset(s, date, 3)

    mom_5d = (p0 / p_5b - 1) if pd.notna(p_5b) and p_5b != 0 else np.nan
    mom_10d = (p0 / p_10b - 1) if pd.notna(p_10b) and p_10b != 0 else np.nan
    next_3d = (p_3a / p0 - 1) if pd.notna(p_3a) and p0 != 0 else np.nan

    rows.append({
        "ticker": ticker,
        "date": date,
        "return_zscore": row["return_zscore"],
        "volume_zscore": row["volume_zscore"],
        "price_momentum_5d": mom_5d,
        "price_momentum_10d": mom_10d,
        "next_3d_return": next_3d,
    })

df = pd.DataFrame(rows)

# days_since_last_anomaly per stock (fill first occurrence with 999)
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
df["days_since_last_anomaly"] = (
    df.groupby("ticker")["date"].diff().dt.days.fillna(999)
)

# Target: 1 if |next 3-day return| > 1%
df["target"] = (df["next_3d_return"].abs() > 0.01).astype(int)

df = df.dropna(subset=FEATURES + ["next_3d_return"]).reset_index(drop=True)

# --- Chronological split (no shuffling) ---
df = df.sort_values("date").reset_index(drop=True)
split = int(len(df) * 0.7)
train, test = df.iloc[:split], df.iloc[split:]

print(f"Dataset: {len(df)} rows | Train: {len(train)} | Test: {len(test)}")
print(f"Target distribution (full): {df['target'].value_counts().to_dict()}\n")

X_train, y_train = train[FEATURES], train["target"]
X_test, y_test = test[FEATURES], test["target"]

# --- Train ---
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Feature Importances:")
for feat, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# --- Save ---
with open("alpha_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("\nModel saved to alpha_model.pkl")
