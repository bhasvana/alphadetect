# AlphaDetect

**Anomaly-driven signal detection and backtesting for NSE large-cap stocks.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alphadetect.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

AlphaDetect watches 8 major NSE stocks every trading day and flags days where both the price move *and* the trading volume are simultaneously unusual. It then trains a machine learning model to predict whether those anomalous days will be followed by a significant price move in the next 3 days — and backtests a trading strategy on those predictions.

**Live app → [alphadetect.streamlit.app](https://alphadetect.streamlit.app)**

---

## The idea

Most days in the market are noise. But some days are genuinely different — a stock moves far more than normal *and* an unusually large number of shares trade hands at the same time. These two signals together suggest something real is happening: institutional activity, news before it's public, earnings surprises, or sector rotation.

The question this project answers: **can a model learn from past anomalies to predict which ones lead to follow-through moves — and can you profit from that?**

---

## Stocks covered

| Ticker | Company |
|---|---|
| RELIANCE.NS | Reliance Industries |
| TCS.NS | Tata Consultancy Services |
| INFY.NS | Infosys |
| HDFCBANK.NS | HDFC Bank |
| ICICIBANK.NS | ICICI Bank |
| WIPRO.NS | Wipro |
| BAJFINANCE.NS | Bajaj Finance |
| SBIN.NS | State Bank of India |

---

## Pipeline

```
data.py          →    model.py         →    backtest.py
─────────────────     ─────────────────     ────────────────────
Download 2 years      Engineer features     Simulate trades on
of OHLCV data         from each anomaly     model predictions
                      day. Train Random     with transaction
Flag anomaly days:    Forest. Predict       costs and a
  |return_zscore| > 2 whether next 3        market regime
  volume_zscore > 1.5 days move > 1%.       filter.
                      Save model.
Save anomalies.csv                          Print P&L metrics
                                            vs benchmark.
```

### Step 1 — `data.py` (Anomaly Detection)

Downloads 2 years of daily OHLCV data via `yfinance`. For each stock, computes:

- **Daily return** — `pct_change()` on closing price
- **Return z-score** — how many standard deviations today's move is from the 20-day rolling mean
- **Volume z-score** — same, applied to share volume

A day is flagged as an anomaly when **both** conditions hold:

```
|return_zscore| > 2.0   AND   volume_zscore > 1.5
```

Output: `anomalies.csv` — 76 anomaly days across 8 stocks over 2 years.

### Step 2 — `model.py` (Machine Learning)

Loads `anomalies.csv`, engineers 5 features per anomaly row:

| Feature | Description |
|---|---|
| `return_zscore` | How extreme the price move was |
| `volume_zscore` | How extreme the volume was |
| `price_momentum_5d` | Stock return in the 5 trading days before the anomaly |
| `price_momentum_10d` | Stock return in the 10 trading days before the anomaly |
| `days_since_last_anomaly` | Calendar days since this stock's previous anomaly |

**Target variable:** `1` if `|3-day forward return| > 1%`, else `0`

Trains a **Random Forest classifier** (100 trees, `class_weight=balanced`) on the first 70% of anomaly days chronologically, tests on the last 30%. No shuffling — this simulates real-world conditions where you can never use future data.

**Feature importances from training:**

```
price_momentum_10d       0.2706  ←  setup before the anomaly matters most
days_since_last_anomaly  0.2303  ←  anomaly clustering is predictive
volume_zscore            0.1802
return_zscore            0.1613
price_momentum_5d        0.1575
```

Output: `alpha_model.pkl`

### Step 3 — `backtest.py` (Strategy Simulation)

Loads the test set (last 30%) and applies three filters before taking a trade:

1. **Model predicts 1** — big follow-through move expected
2. **Direction filter** — `return_zscore < 0` (price *dropped* on anomaly day — potential bounce)
3. **Regime filter** — 20-day rolling equal-weighted market return > -1% (avoid trading into a sustained downtrend)

**Trade mechanics:**
- Enter at close on the anomaly day
- Hold for **1 trading day**
- Exit at next day's close
- Transaction cost: **0.05% each way** (0.1% round trip)

**Benchmark:** Equal-weighted buy-and-hold of all 8 stocks over the same test period.

---

## Results

Test period: **Dec 2025 → May 2026** (a difficult period — Indian markets fell ~16%)

| Metric | Strategy | Benchmark |
|---|---|---|
| Total return | **-0.73%** | -16.10% |
| Max drawdown | **-1.99%** | -19.49% |
| Trades executed | 4 | — |
| Win rate | 50% | — |
| **Alpha (excess return)** | **+15.37%** | — |

The regime filter was the decisive factor — it blocked 5 of 9 model signals because the market's 20-day rolling return had already crossed below -1%. Those 5 blocked trades would have been losses in the falling market. The 4 trades taken in calmer conditions nearly broke even while the benchmark lost over 16%.

> **Note:** 4 trades over 6 months is too small a sample for statistical significance. These results validate the framework and the logic, not a production-ready strategy. A longer data window (5–10 years) is needed for reliable conclusions.

---

## Live App

The Streamlit app has three tabs:

- **Anomalies** — anomaly count per stock, per-stock inspection table with return and volume z-scores
- **Model** — feature importance chart with explanations
- **Backtest** — live equity curve (strategy vs buy-and-hold), trade metrics, full trade log

All price data is cached (1-hour TTL) so the app doesn't re-download on every interaction.

**[alphadetect.streamlit.app](https://alphadetect.streamlit.app)**

---

## Run locally

```bash
git clone https://github.com/bhasvana/alphadetect.git
cd alphadetect

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Run the full pipeline:

```bash
python data.py       # download data and detect anomalies → anomalies.csv
python model.py      # train model → alpha_model.pkl
python backtest.py   # simulate strategy and print metrics
```

Launch the app:

```bash
streamlit run app.py
```

---

## Tech stack

| Library | Purpose |
|---|---|
| `yfinance` | Download historical OHLCV data from Yahoo Finance |
| `pandas` / `numpy` | Data wrangling, rolling statistics, z-scores |
| `scikit-learn` | Random Forest classifier, classification report |
| `streamlit` | Interactive web app |
| `plotly` | Interactive charts (equity curve, feature importances) |

---

## What's next

- **Longer backtest window** — run over 5+ years to get 100+ trades for statistically meaningful results
- **Direction prediction** — extend the model to predict up vs down, not just magnitude. Currently the strategy is long-only; a short leg would make it market-neutral
- **More features** — sector index returns, FII/DII flow data, options implied volatility
- **Walk-forward validation** — retrain the model monthly on a rolling window instead of a fixed 70/30 split

---

## Project structure

```
alphadetect/
├── data.py           # anomaly detection pipeline
├── model.py          # feature engineering + Random Forest training
├── backtest.py       # strategy simulation and metrics
├── app.py            # Streamlit web app
├── anomalies.csv     # detected anomaly days (output of data.py)
├── alpha_model.pkl   # trained classifier (output of model.py)
└── requirements.txt
```
