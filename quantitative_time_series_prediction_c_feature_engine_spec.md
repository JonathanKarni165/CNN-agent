# Project Specification: High-Performance Quantitative Time-Series Forecasting Engine (C + Python / PyTorch)

## 1. Project Overview & Objective
Build an end-to-end, research-grade quantitative forecasting pipeline designed to predict short-horizon forward returns on noisy financial market data (e.g., S&P 500 `SPY` 1-minute OHLCV bars or Binance L2 Order Book / tick data).

The project demonstrates two core competencies expected in quantitative research roles:
1. **Low-Level Systems Engineering (C):** A zero-overhead, $O(N)$ rolling feature computation engine implemented in C using circular buffers and Welford's online variance algorithm, exposed to Python via `ctypes` or Python C-API.
2. **Statistical Rigor & Deep Learning (Python/PyTorch):** A leak-free, chronologically validated modeling pipeline comparing a regularized linear baseline (**Ridge Regression**) and gradient boosted trees (**XGBoost / LightGBM**) against a compact **1D-CNN + GRU** neural network, evaluated via out-of-sample Information Coefficient (IC) and realistic transaction-cost-adjusted backtesting.

---

## 2. Directory Structure
```text
quant-ts-engine/
├── c_engine/
│   ├── Makefile
│   ├── features.h
│   ├── features.c          # O(N) rolling features using circular buffers
│   └── benchmark.c         # Standalone C speed benchmark
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Fetch/load SPY 1-min data or Binance L2 CSVs
│   ├── c_bridge.py         # ctypes wrapper + speed benchmark vs Pandas
│   ├── dataset.py          # Zero-copy sliding window PyTorch Dataset
│   ├── models.py           # Ridge, XGBoost, and 1D-CNN + GRU architectures
│   ├── train.py            # Walk-forward / chronological training loop
│   └── backtest.py         # Out-of-sample IC, Sharpe Ratio, Max Drawdown & PnL
├── notebooks/
│   └── research_report.ipynb # EDA, feature importance, model comparison plots
├── tests/
│   └── test_features.py    # Unit tests verifying C outputs match NumPy/Pandas
├── requirements.txt
└── README.md
```

---

## 3. Data Pipeline & Target Formulation

### 3.1 Data Sources (Configurable)
* **Primary Option (Equities - SPY):** 1-minute OHLCV bars (`timestamp`, `open`, `high`, `low`, `close`, `volume`) fetched via `yfinance` or local CSV. Filter strictly to regular market hours (09:30–16:00 EST).
* **Secondary Option (High-Frequency L2 Crypto):** Binance Data Vision historical L2/tick CSVs (`bid_price`, `bid_qty`, `ask_price`, `ask_qty`, `timestamp`).

### 3.2 Target Variable ($y_t$)
Predict forward log-return over horizon $k$ (default $k = 5$ or $10$ steps) to ensure stationarity:
$$y_t = \ln\left(\frac{P_{t+k}}{P_t}\right)$$
*Note: Standardize $y_t$ using training-set statistics only, or predict raw forward log-returns with Huber Loss.*

---

## 4. C Feature Engineering Engine (`c_engine/features.c`)

### 4.1 Technical Requirements
* Written in standard C11 (`-O3 -march=native -Wall -Wextra -fPIC -shared`).
* **Strict Causal Execution:** Feature at index $t$ MUST ONLY use data from indices $\le t$. Zero lookahead bias.
* **Memory & Complexity:** Single pass $O(N)$ time complexity per rolling window using fixed-size ring (circular) buffers. Use **Welford's online algorithm** (or compensated summation) for rolling mean and standard deviation to prevent floating-point catastrophic cancellation.
* **Warm-up Handling:** First $W_{\max}$ rows are flagged/masked as invalid (`NaN` or boolean validity mask) until all rolling buffers are populated.

### 4.2 Feature Definitions (Computed in C & Python)
Given window sizes $W \in \{10, 30, 60\}$:

1. **Multi-Horizon Log Returns (Momentum):**
   $$r_{t, h} = \ln\left(\frac{P_t}{P_{t-h}}\right) \quad \text{for } h \in \{1, 5, 15, 60\}$$
2. **Rolling Price Z-Score (Mean Reversion):**
   $$Z_{t, W} = \frac{P_t - \mu_{t, W}}{\sigma_{t, W} + \epsilon}$$
3. **Price Position in Rolling Range:**
   $$\text{Pos}_{t, W} = \frac{P_t - \min(P_{t-W+1 \dots t})}{\max(P_{t-W+1 \dots t}) - \min(P_{t-W+1 \dots t}) + \epsilon} \in [0, 1]$$
4. **Realized Volatility & Volatility Ratio (Regime Detection):**
   $$\sigma^{\text{ret}}_{t, W} = \text{StdDev}(r_{t-W+1 \dots t, 1}), \qquad \text{VolRatio}_t = \frac{\sigma^{\text{ret}}_{t, 15}}{\sigma^{\text{ret}}_{t, 60} + \epsilon}$$
5. **Relative Volume & Signed Volume Flow:**
   $$\text{RelVol}_{t, W} = \frac{V_t}{\text{Mean}(V_{t-W+1 \dots t}) + \epsilon}, \qquad \text{SignedVol}_t = V_t \cdot \text{sign}(P_t - P_{t-1})$$
6. **Order Book Features (If L2 data is enabled):**
   * **Order Book Imbalance (OBI):** $\text{OBI}_t = \frac{V^{\text{bid}}_t - V^{\text{ask}}_t}{V^{\text{bid}}_t + V^{\text{ask}}_t + \epsilon}$
   * **Micro-Price Divergence:** $\text{MicroDiv}_t = \frac{P^{\text{micro}}_t - P^{\text{mid}}_t}{P^{\text{mid}}_t}$, where $P^{\text{micro}}_t = \frac{P^{\text{bid}}_t V^{\text{ask}}_t + P^{\text{ask}}_t V^{\text{bid}}_t}{V^{\text{bid}}_t + V^{\text{ask}}_t}$
   * **Normalized Spread:** $\text{Spread}_t = \frac{P^{\text{ask}}_t - P^{\text{bid}}_t}{P^{\text{mid}}_t}$
7. **Cyclical Time / Intraday Seasonality Features (Computed in Python):**
   * Minutes since market open ($m \in [0, 390]$):
     $$\sin\left(\frac{2\pi m}{390}\right), \quad \cos\left(\frac{2\pi m}{390}\right)$$
   * Day of week ($d \in \{0..4\}$):
     $$\sin\left(\frac{2\pi d}{5}\right), \quad \cos\left(\frac{2\pi d}{5}\right)$$

---

## 5. Validation Protocol & Leakage Prevention (`src/dataset.py`)

1. **Chronological Split with Purged Embargo Gap:**
   * **Train:** First 70% of timeline.
   * **Validation:** Next 15% of timeline.
   * **Test:** Final 15% of timeline (untouched until final evaluation).
   * **Embargo Gap:** Insert a gap of $L + k$ timesteps (where $L$ is sequence length and $k$ is forecast horizon) between Train/Val and Val/Test boundaries so no target window overlaps across splits.
2. **Feature Scaling:**
   * Fit `RobustScaler` or `StandardScaler` **exclusively on the Train split** and transform Val/Test splits. Clip extreme outliers at $\pm 10\sigma$.
3. **Zero-Copy Sliding Windows:**
   * Construct 3D tensors of shape `(Batch, Seq_Len L=30, Num_Features F)` using `torch.Tensor.unfold(0, L, 1)` to avoid memory duplication.

---

## 6. Models & Training Architecture (`src/models.py`, `src/train.py`)

Implement and compare three distinct model tiers on the exact same splits:

### Tier 1: Linear Baseline (`Ridge`)
* Flattened window features or last-step features fed into `sklearn.linear_model.Ridge` with L2 regularization tuned on Validation IC.

### Tier 2: Gradient Boosted Decision Trees (`XGBoost` / `LightGBM`)
* Tabular features + lag summaries trained with Huber/RMSE objective, `max_depth=3..4`, `subsample=0.7`, `colsample_bytree=0.7`, and early stopping on Validation IC.

### Tier 3: Compact Temporal Deep Learning (`QuantTimeSeriesNet` in PyTorch)
* **Input:** `(Batch, Seq_Len=30, Features=F)`
* **Architecture:**
  1. `Conv1d(in_channels=F, out_channels=32, kernel_size=3, padding=1)` -> `GELU` -> `Dropout(0.4)` (extracts local temporal motifs and filters tick noise).
  2. `GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True)` (aggregates sequential state across the 30-step window).
  3. Extract final hidden state `out[:, -1, :]` -> `Dropout(0.4)` -> `Linear(32, 1)`.
* **Optimization:**
  * Loss Function: `nn.HuberLoss(delta=1.0)` (robust to fat-tailed market jumps).
  * Optimizer: `AdamW(lr=1e-3, weight_decay=1e-2)` with `ReduceLROnPlateau`.
  * Checkpointing: Save best model weights based on highest **Validation Spearman Information Coefficient (IC)**, not lowest MSE.

---

## 7. Evaluation Metrics & Backtesting Engine (`src/backtest.py`)

Evaluate all three models side-by-side on the held-out **Test Set**:

1. **Predictive Quality Metrics:**
   * **Information Coefficient (IC):** Pearson and Spearman rank correlation between predicted returns $\hat{y}_t$ and realized returns $y_t$.
   * **Directional Accuracy (Hit Rate):** Percentage of correct sign predictions when $|\hat{y}_t| > \theta$ (high-confidence threshold).
2. **Vectorized Strategy Simulation (Backtest):**
   * **Signal Rule:** Enter long ($+1$) if $\hat{y}_t > \theta$, short ($-1$) if $\hat{y}_t < -\theta$, else flat ($0$), where $\theta$ is calibrated on the Validation set (e.g., top/bottom decile of predictions).
   * **Transaction Costs:** Deduct realistic slippage + commission fee $c$ (e.g., $1\text{–}2\text{ bps} = 0.0001\text{–}0.0002$) on every position change $|\text{pos}_t - \text{pos}_{t-1}|$.
   * **Financial Metrics:**
     * Annualized **Sharpe Ratio** (net of fees).
     * **Maximum Drawdown (MDD)**.
     * Cumulative PnL curve vs. Buy-and-Hold benchmark.
     * Turnover rate (trades per day).
3. **Engineering Benchmark:**
   * Measure execution time (ms) and speedup factor (`x`) of the C rolling feature engine vs. pure Pandas/NumPy across $1,000,000$ rows.

---

## 8. Deliverables & Acceptance Criteria
1. `tests/test_features.py` passes with `< 1e-9` max absolute difference between the C shared library and reference NumPy/Pandas implementations.
2. Clean, modular Python and C code with type hints, docstrings, and a one-command `Makefile` build.
3. Execution script (`python -m src.train --evaluate`) that outputs a comparison table across Ridge, XGBoost, and 1D-CNN+GRU, plus generates summary plots (IC over time, cumulative net PnL curve, and C vs. Pandas latency chart).