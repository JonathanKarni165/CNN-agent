# Project Summary — Quantitative Time-Series Forecasting Pipeline

A from-scratch research pipeline for predicting short-horizon forward returns on SPY intraday
price data, built to mirror how a quant research team iterates: modular feature engineering,
leakage-free validation, and systematic model comparison — not a single notebook with a model
fit at the bottom.

## What works today

**Data ingestion** — pulls SPY OHLCV bars (configurable ticker/interval/period) via `yfinance`,
filtered to regular US market hours.

**Feature engine (OOP, Strategy + Registry pattern)** — each feature is an independent,
unit-testable class (`Feature` subclass) registered into a shared `FeatureEngine`; adding a new
feature never requires touching the engine. All features are strictly causal (no lookahead),
with explicit warm-up masking. Implemented so far:
- Momentum: multi-horizon log returns
- Mean-reversion: rolling z-score, price position in rolling range
- Regime detection: realized volatility
- Order flow: relative volume, signed volume flow
- Seasonality: cyclical (sin/cos) time-of-day and day-of-week encodings

**Leakage-safe dataset construction** — chronological Train/Val/Test split (70/10/20) with a
**purged embargo gap** at each boundary (sized to sequence length + forecast horizon) so no
feature window or target ever straddles two splits. Feature scaling is fit on Train only and
applied to Val/Test, with outlier clipping.

**Models (common interface, swappable)** — Ridge regression, XGBoost, and a 1D-CNN + GRU deep
model ("QuantTimeSeriesNet") all behind one shared interface, so each tier plugs into the same
training/evaluation code. XGBoost and the CNN+GRU both expose a selectable training objective
(squared / MAE / Huber).

**Tier 3: 1D-CNN + GRU (implemented and trained)** — `Conv1d -> GELU -> Dropout -> GRU -> Dropout
-> Linear`, built on zero-copy sliding windows (`torch.Tensor.unfold`) over the same scaled
features and splits used by Ridge/XGBoost. Trained with AdamW + `ReduceLROnPlateau`, checkpointed
on best **validation Spearman IC** (not lowest loss, per the research-correct practice for this
kind of model), with live in-place epoch progress. On the held-out test set it reached the
**highest Pearson IC of the three models (0.080)**, though XGBoost still led on Spearman IC
(0.107) and hit rate (54.4%) — consistent with deep models needing more data than ~3,200 training
rows to fully outperform gradient-boosted trees on tabular/sequential financial features.

**Evaluation** — out-of-sample Pearson and Spearman **Information Coefficient** and hit-rate,
computed only on the untouched Test split — the standard quant-research metric, not generic
accuracy/RMSE.

**Experimentation workflow** — a sequence of Jupyter notebooks, each independently rerunnable:
1. Dataset & feature selection
2. Chronological split & scaling
3. Model definition (incl. loss-function choice)
4. Training (with a live, in-place progress widget instead of flooding notebook output)
5. Results visualization — including the predicted buy/sell signal painted directly on the
   price chart (color = direction, intensity = percentile confidence) with the model's holding
   window (fixed at the forecast horizon) overlaid
6. Feature-set ablation — trains the same models on multiple feature subsets and compares
   out-of-sample IC side by side (empirically found, for example, that calendar/time-of-day
   features help XGBoost but hurt Ridge — feature value is model-dependent)
7. CNN+GRU training & evaluation — same splits/scaling as Ridge/XGBoost, zero-copy windowing,
   validation-IC checkpointing, compared head-to-head against the other two tiers

All notebooks are executed and verified end-to-end (via `nbconvert`), not just written.

## Tools & libraries used

- **Python** (pandas, numpy) for the full pipeline
- **scikit-learn** — Ridge regression, `StandardScaler`/`RobustScaler`
- **XGBoost** — gradient-boosted trees with configurable loss objective and early stopping
- **yfinance** — market data ingestion
- **matplotlib** — all visualizations, including custom `LineCollection`-based color-gradient
  price charts
- **scipy** — Pearson/Spearman correlation for IC calculation
- **Jupyter / ipywidgets / tqdm-style live progress** — interactive, rerunnable research workflow
- **PyTorch** — the CNN+GRU model (Conv1d, GRU, `nn.HuberLoss`/`MSELoss`/`L1Loss`, AdamW,
  `ReduceLROnPlateau`), sliding windows via `Tensor.unfold`
- **conda** — isolated `quant` environment with a registered Jupyter kernel
- **LightGBM** — installed and environment-ready for further GBDT comparison, not yet used in a
  trained model

## Not yet built (in progress / specified, not implemented)

- **C feature-computation engine** — O(N) rolling features via circular buffers and Welford's
  online variance algorithm, exposed to Python via ctypes, benchmarked against pandas. Fully
  specified (see `quantitative_time_series_prediction_c_feature_engine_spec.md`) but not yet
  implemented — this is the systems/low-latency engineering component of the project.
- **Transaction-cost-adjusted backtest engine** — Sharpe ratio, max drawdown, turnover, cumulative
  PnL vs. buy-and-hold. Not yet built; current evaluation stops at IC/hit-rate.
- **Confidence-weighted position sizing** — current signal logic is binary enter/exit at a fixed
  holding horizon, not proportional sizing (though the percentile "signal strength" used for the
  visualization is a natural building block for this later).

## Notes for writing this up (resume / interview)

- Lead with the **leakage-prevention methodology** (embargo gap, train-only scaling) and the
  **IC-based evaluation discipline** — this is what signals real quant-research practice, more
  than any specific model name.
- The OOP/design-pattern structure is worth a brief mention as engineering maturity, not the
  headline.
- Be precise about what's implemented vs. specified-but-pending (the C engine and CNN+GRU tier)
  if asked in an interview — don't let a resume bullet imply more is finished than actually is.
