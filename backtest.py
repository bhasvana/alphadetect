import pickle
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "SBIN.NS",
]
FEATURES = [
    "return_zscore", "volume_zscore",
    "price_momentum_5d", "price_momentum_10d",
    "days_since_last_anomaly",
]
TC = 0.0005  # 0.05% per leg

# ── Reconstruct identical feature set from model.py ───────────────────────────
anomalies = pd.read_csv("anomalies.csv", parse_dates=["date"])
anomalies["date"] = anomalies["date"].dt.normalize()
anomalies = anomalies.sort_values(["ticker", "date"]).reset_index(drop=True)

end = datetime.today()
start = end - timedelta(days=2 * 365 + 90)
close_all = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
close_all.index = close_all.index.normalize()


def price_at_offset(series, date, offset):
    loc = series.index.searchsorted(date)
    target = loc + offset
    if target < 0 or target >= len(series):
        return np.nan
    return series.iloc[target]


rows = []
for _, row in anomalies.iterrows():
    ticker = row["ticker"]
    date = pd.Timestamp(row["date"]).normalize()
    p0 = row["Close"]
    s = close_all[ticker].dropna()

    p_5b  = price_at_offset(s, date, -5)
    p_10b = price_at_offset(s, date, -10)
    p_3a  = price_at_offset(s, date,  3)

    rows.append({
        "ticker":            ticker,
        "date":              date,
        "return_zscore":     row["return_zscore"],
        "volume_zscore":     row["volume_zscore"],
        "price_momentum_5d":  (p0 / p_5b  - 1) if pd.notna(p_5b)  and p_5b  != 0 else np.nan,
        "price_momentum_10d": (p0 / p_10b - 1) if pd.notna(p_10b) and p_10b != 0 else np.nan,
        "next_3d_return":    (p_3a / p0  - 1) if pd.notna(p_3a)  and p0   != 0 else np.nan,
        "Close":             p0,
    })

df = pd.DataFrame(rows)
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
df["days_since_last_anomaly"] = df.groupby("ticker")["date"].diff().dt.days.fillna(999)
df["target"] = (df["next_3d_return"].abs() > 0.01).astype(int)
df = df.dropna(subset=FEATURES + ["next_3d_return"]).reset_index(drop=True)
df = df.sort_values("date").reset_index(drop=True)

split = int(len(df) * 0.7)
test = df.iloc[split:].copy()

# ── Load model & predict ───────────────────────────────────────────────────────
with open("alpha_model.pkl", "rb") as f:
    clf = pickle.load(f)

test = test.copy()
test["prediction"] = clf.predict(test[FEATURES])
signals = test[(test["prediction"] == 1) & (test["return_zscore"] < 0)].copy()

# 20-day rolling equal-weighted return — used as market regime filter
bm_daily_full = close_all.pct_change().mean(axis=1)
bm_roll_20d   = (1 + bm_daily_full).rolling(20).apply(np.prod, raw=True) - 1

# ── Strategy simulation ────────────────────────────────────────────────────────
# For each signal: hold the stock for 1 trading day.
# Build a per-day return contribution: each active trade contributes its
# actual daily stock return.  TC is subtracted from the entry and exit legs.
# Strategy daily return = equal-weighted mean of all active trades that day.

daily_contribs = defaultdict(list)  # date -> [daily returns from active trades]
trade_records = []

for _, sig in signals.iterrows():
    ticker = sig["ticker"]
    date   = sig["date"]
    s      = close_all[ticker].dropna()
    loc    = s.index.searchsorted(date)

    # Regime filter: skip if 20-day equal-weighted market return <= -1%
    regime_ret = bm_roll_20d.get(date, np.nan)
    if pd.isna(regime_ret) or regime_ret <= -0.01:
        continue

    hold_days = []
    for i in range(1):
        if loc + i + 1 >= len(s):
            break
        day_start = s.index[loc + i]
        day_ret   = s.iloc[loc + i + 1] / s.iloc[loc + i] - 1
        hold_days.append([day_start, day_ret])

    if len(hold_days) < 1:
        continue  # skip if near end of data

    # Deduct TC from entry and exit (both on the single holding day)
    hold_days[0][1] -= 2 * TC

    net_trade_return = hold_days[0][1]

    for day_date, day_ret in hold_days:
        daily_contribs[day_date].append(day_ret)

    trade_records.append({
        "ticker":       ticker,
        "entry_date":   date,
        "exit_date":    s.index[loc + 1],
        "net_return":   net_trade_return,
        "win":          net_trade_return > 0,
    })

trade_df = pd.DataFrame(trade_records)

# ── Build strategy equity curve ────────────────────────────────────────────────
test_start = test["date"].min()
test_dates = close_all.index[close_all.index >= test_start]

strategy_daily = pd.Series(0.0, index=test_dates)
for d, rets in daily_contribs.items():
    if d in strategy_daily.index:
        strategy_daily[d] = float(np.mean(rets))

strategy_equity = (1 + strategy_daily).cumprod()


def sharpe(daily_ret_series, periods=252):
    mu, sigma = daily_ret_series.mean(), daily_ret_series.std()
    return (mu / sigma) * np.sqrt(periods) if sigma > 0 else 0.0


def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    return ((equity_curve - roll_max) / roll_max).min()


# ── Benchmark: equal-weighted buy-and-hold ─────────────────────────────────────
bm_returns = close_all.loc[test_dates].pct_change().dropna()
bm_daily   = bm_returns.mean(axis=1)
bm_equity  = (1 + bm_daily).cumprod()

# Align strategy and benchmark to the same dates
common = strategy_daily.index.intersection(bm_daily.index)
strat_aligned = strategy_daily.loc[common]
bm_aligned    = bm_daily.loc[common]

# ── Print results ──────────────────────────────────────────────────────────────
n_trades  = len(trade_df)
win_rate  = trade_df["win"].mean() if n_trades > 0 else 0.0
tot_ret   = strategy_equity.iloc[-1] - 1
active_days = strategy_daily[strategy_daily != 0]
ann_sharp   = sharpe(active_days)
mdd       = max_drawdown(strategy_equity)

bm_tot    = bm_equity.iloc[-1] - 1
bm_sharp  = sharpe(bm_aligned)
bm_mdd    = max_drawdown(bm_equity)

print("=" * 55)
print("  STRATEGY: Model Signal + Direction Filter (bounce, 1-day hold)")
print("=" * 55)
print(f"  Test period          : {test_start.date()} → {test_dates[-1].date()}")
print(f"  Signals generated    : {len(signals)}")
print(f"  Trades executed      : {n_trades}")
print(f"  Win rate             : {win_rate:.1%}")
print(f"  Total return         : {tot_ret:+.2%}")
print(f"  Annualised Sharpe    : {ann_sharp:.2f}  (on {len(active_days)} active days)")
print(f"  Maximum drawdown     : {mdd:.2%}")

print()
print("=" * 55)
print("  BENCHMARK: Equal-Weighted Buy-and-Hold (8 stocks)")
print("=" * 55)
print(f"  Total return         : {bm_tot:+.2%}")
print(f"  Annualised Sharpe    : {bm_sharp:.2f}")
print(f"  Maximum drawdown     : {bm_mdd:.2%}")

print()
print("=" * 55)
print("  COMPARISON (Strategy vs Benchmark)")
print("=" * 55)
print(f"  Excess return (alpha): {tot_ret - bm_tot:+.2%}")
print(f"  Sharpe advantage     : {ann_sharp - bm_sharp:+.2f}")
print(f"  Drawdown improvement : {bm_mdd - mdd:+.2%}")

if n_trades > 0:
    print()
    print("  Per-trade breakdown:")
    print(f"    Avg trade return : {trade_df['net_return'].mean():+.2%}")
    print(f"    Best trade       : {trade_df['net_return'].max():+.2%}")
    print(f"    Worst trade      : {trade_df['net_return'].min():+.2%}")
