import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

st.set_page_config(page_title="AlphaDetect", page_icon="📈", layout="wide")
st.title("AlphaDetect — NSE Anomaly Signal Backtester")
st.caption(
    "Detects anomalous price + volume days on 8 NSE large-caps "
    "and tests a bounce strategy using a Random Forest classifier."
)

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS", "SBIN.NS",
]
FEATURES = [
    "return_zscore", "volume_zscore",
    "price_momentum_5d", "price_momentum_10d",
    "days_since_last_anomaly",
]
TC = 0.0005


@st.cache_data
def load_anomalies():
    df = pd.read_csv("anomalies.csv", parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    return df


@st.cache_resource
def load_model():
    with open("alpha_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def download_prices():
    end = datetime.today()
    start = end - timedelta(days=2 * 365 + 90)
    close = yf.download(
        TICKERS, start=start, end=end, auto_adjust=True, progress=False
    )["Close"]
    close.index = pd.DatetimeIndex(close.index.date)  # strip timezone, keep date only
    return close


@st.cache_data
def run_backtest(_close_all, _anomalies, _clf):
    def price_at_offset(series, date, offset):
        loc = series.index.searchsorted(date)
        target = loc + offset
        if target < 0 or target >= len(series):
            return np.nan
        return series.iloc[target]

    rows = []
    for _, row in _anomalies.iterrows():
        ticker = row["ticker"]
        date = pd.Timestamp(row["date"]).normalize()
        p0 = row["Close"]
        s = _close_all[ticker].dropna()
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
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["days_since_last_anomaly"] = df.groupby("ticker")["date"].diff().dt.days.fillna(999)
    df["target"] = (df["next_3d_return"].abs() > 0.01).astype(int)
    df = df.dropna(subset=FEATURES + ["next_3d_return"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    split = int(len(df) * 0.7)
    test = df.iloc[split:].copy()
    test["prediction"] = _clf.predict(test[FEATURES])
    signals = test[(test["prediction"] == 1) & (test["return_zscore"] < 0)].copy()

    bm_daily_full = _close_all.pct_change().mean(axis=1)
    bm_roll_20d   = (1 + bm_daily_full).rolling(20).apply(np.prod, raw=True) - 1

    daily_contribs = defaultdict(list)
    trade_records  = []

    for _, sig in signals.iterrows():
        ticker = sig["ticker"]
        date   = sig["date"]
        s      = _close_all[ticker].dropna()
        loc    = s.index.searchsorted(date)

        regime_ret = bm_roll_20d.get(date, np.nan)
        if pd.isna(regime_ret) or regime_ret <= -0.01:
            continue
        if loc + 1 >= len(s):
            continue

        day_ret = s.iloc[loc + 1] / s.iloc[loc] - 1 - 2 * TC
        daily_contribs[s.index[loc]].append(day_ret)
        trade_records.append({
            "Ticker":       ticker,
            "Entry Date":   date,
            "Net Return":   day_ret,
            "Win":          day_ret > 0,
        })

    test_start = test["date"].min()
    test_dates  = _close_all.index[_close_all.index >= test_start]

    strategy_daily = pd.Series(0.0, index=test_dates)
    for d, rets in daily_contribs.items():
        if d in strategy_daily.index:
            strategy_daily[d] = float(np.mean(rets))

    strategy_equity = (1 + strategy_daily).cumprod()

    # Compute pct_change on full history then slice — avoids NaN on first test row
    bm_daily  = _close_all.pct_change().mean(axis=1).dropna()
    bm_daily  = bm_daily[bm_daily.index >= test_start]
    bm_equity = (1 + bm_daily).cumprod()

    roll_max = strategy_equity.cummax()
    mdd      = ((strategy_equity - roll_max) / roll_max).min()

    trade_df = pd.DataFrame(trade_records)

    return strategy_equity, bm_equity, trade_df, mdd, test_start


# ── Load data ──────────────────────────────────────────────────────────────────
anomalies = load_anomalies()
clf       = load_model()

tab1, tab2, tab3 = st.tabs(["Anomalies", "Model", "Backtest"])

# ── Tab 1: Anomalies ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Anomaly Overview")

    counts = anomalies.groupby("ticker").size().reset_index(name="count")
    counts["label"] = counts["ticker"].str.replace(".NS", "", regex=False)

    fig_bar = go.Figure(go.Bar(
        x=counts["label"], y=counts["count"],
        marker_color="#2563EB", text=counts["count"], textposition="outside"
    ))
    fig_bar.update_layout(
        title="Anomaly days per stock (2 years)",
        xaxis_title="Stock", yaxis_title="Count",
        height=350, showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"**Total anomaly days across all stocks: {len(anomalies)}**")
    st.caption(
        "An anomaly day = |return z-score| > 2 AND volume z-score > 1.5. "
        "Both price move and volume must be unusually large simultaneously."
    )

    st.divider()
    st.subheader("Inspect a Stock")

    selected = st.selectbox("Select stock", sorted(anomalies["ticker"].unique()))
    sub = anomalies[anomalies["ticker"] == selected].sort_values("date")

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly Days", len(sub))
    c2.metric("Avg Return Z-Score", f"{sub['return_zscore'].mean():.2f}")
    c3.metric("Avg Volume Z-Score", f"{sub['volume_zscore'].mean():.2f}")

    display = sub[["date", "daily_return", "return_zscore", "volume_zscore"]].copy()
    display["daily_return"] = display["daily_return"].map("{:.2%}".format)
    display.columns = ["Date", "Daily Return", "Return Z-Score", "Volume Z-Score"]
    st.dataframe(display.reset_index(drop=True), use_container_width=True)

# ── Tab 2: Model ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Random Forest — Feature Importances")

    feat_df = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": clf.feature_importances_
    }).sort_values("Importance")

    fig_imp = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Feature"],
        orientation="h", marker_color="#16A34A",
        text=feat_df["Importance"].map("{:.3f}".format),
        textposition="outside"
    ))
    fig_imp.update_layout(
        title="What drives the model's decisions",
        xaxis_title="Importance Score",
        height=380
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "Higher = more influential. The 10-day momentum going *into* the anomaly "
        "and how long since the last anomaly on the same stock matter most. "
        "The spike magnitude (return_zscore) matters least."
    )

    with st.expander("Model details"):
        st.markdown("""
        - **Algorithm:** Random Forest (100 trees)
        - **Target:** 1 if |3-day forward return| > 1%, else 0
        - **Train/test split:** First 70% / Last 30% — chronological, no shuffling
        - **Class weighting:** `balanced` — corrects for 54/22 class imbalance
        - **Features:** 5 engineered from anomaly context and pre-anomaly price history
        """)

# ── Tab 3: Backtest ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Backtest — Strategy vs Buy & Hold")

    with st.spinner("Downloading price data and running backtest…"):
        close_all = download_prices()
        strategy_equity, bm_equity, trade_df, mdd, test_start = run_backtest(
            close_all, anomalies, clf
        )

    n_trades = len(trade_df)
    win_rate = trade_df["Win"].mean() if n_trades > 0 else 0.0
    tot_ret  = strategy_equity.iloc[-1] - 1
    bm_tot   = bm_equity.iloc[-1] - 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades Executed", n_trades)
    c2.metric("Win Rate", f"{win_rate:.0%}")
    c3.metric("Strategy Return", f"{tot_ret:+.2%}", delta=f"{tot_ret - bm_tot:+.2%} vs benchmark")
    c4.metric("Max Drawdown", f"{mdd:.2%}")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=strategy_equity.index, y=strategy_equity.values,
        name="Strategy", line=dict(color="#2563EB", width=2.5)
    ))
    fig_eq.add_trace(go.Scatter(
        x=bm_equity.index, y=bm_equity.values,
        name="Buy & Hold (8 stocks)", line=dict(color="#DC2626", width=2, dash="dash")
    ))
    fig_eq.update_layout(
        title=f"Equity Curve — Test Period from {test_start.date()}",
        yaxis_title="Portfolio Value (starts at 1.0)",
        xaxis_title="Date",
        height=420,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.caption(
        "Strategy: buy on model signal (predicted 1) + price dropped on anomaly day "
        "(return_zscore < 0) + market 20-day rolling return > -1%. Hold 1 day. "
        "0.05% transaction cost each way."
    )

    if n_trades > 0:
        st.divider()
        st.subheader("Trade Log")
        display_trades = trade_df.copy()
        display_trades["Net Return"] = display_trades["Net Return"].map("{:.2%}".format)
        display_trades["Entry Date"] = display_trades["Entry Date"].dt.date
        st.dataframe(display_trades, use_container_width=True)
