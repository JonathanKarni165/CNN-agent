"""Reusable plotting helpers for the results-exploration notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.stats import spearmanr


def _signal_strength(pred: np.ndarray) -> np.ndarray:
    """Map raw predictions to [-1, 1] by symmetric percentile rank, so color
    intensity reflects how strong/confident a prediction is relative to the
    rest of the sample rather than its raw (hard to interpret) magnitude.
    """
    rank = pd.Series(pred).rank(pct=True).to_numpy()  # in (0, 1]
    return 2 * rank - 1  # in (-1, 1]


def plot_price_with_signal(
    index: pd.DatetimeIndex,
    price: pd.Series,
    pred: np.ndarray,
    title: str = "Price colored by predicted signal (green=buy, red=sell)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw the close-price line colored along its length by the model's
    predicted forward return: green shades = buy signal, red shades = sell
    signal, intensity = percentile strength of the prediction (not raw value,
    since return magnitudes aren't directly comparable across regimes).
    """
    strength = _signal_strength(pred)

    points = np.array([np.arange(len(price)), price.to_numpy()]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    lc = LineCollection(segments, cmap="RdYlGn", norm=plt.Normalize(-1, 1))
    lc.set_array(strength[:-1])
    lc.set_linewidth(1.5)
    line = ax.add_collection(lc)

    tick_step = max(1, len(index) // 10)
    ax.set_xticks(np.arange(len(index))[::tick_step])
    ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in index[::tick_step]], rotation=45, ha="right")
    ax.set_xlim(0, len(price) - 1)
    ax.set_ylim(price.min() * 0.998, price.max() * 1.002)
    ax.set_ylabel("Price ($)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label("Signal strength (percentile): red = sell, green = buy")
    return ax


def plot_holding_windows(
    ax: plt.Axes,
    price: pd.Series,
    pred: np.ndarray,
    horizon: int,
    index: pd.DatetimeIndex | None = None,
    max_windows: int = 10,
    strength_threshold: float = 0.6,
) -> list[int]:
    """Overlay a handful of the strongest buy/sell entries with a shaded span
    and a double-headed arrow from entry to exit, so the holding period (fixed
    at the model's `horizon` bars — that's the only "exit rule" this model
    has, it doesn't decide holding time dynamically) is visible on the chart.

    Entries are the `max_windows` strongest predictions (|signal| >=
    `strength_threshold` percentile), chosen greedily so their [entry, exit]
    windows never overlap. Returns the list of selected entry positions.
    """
    strength = _signal_strength(pred)
    n = len(price)

    minutes_per_bar = None
    if index is not None and len(index) > 1:
        minutes_per_bar = pd.Series(index).diff().dt.total_seconds().median() / 60

    candidates = [i for i in range(n - horizon) if abs(strength[i]) >= strength_threshold]
    candidates.sort(key=lambda i: -abs(strength[i]))

    occupied = np.zeros(n, dtype=bool)
    selected: list[int] = []
    for i in candidates:
        if occupied[i : i + horizon + 1].any():
            continue
        occupied[i : i + horizon + 1] = True
        selected.append(i)
        if len(selected) >= max_windows:
            break
    selected.sort()

    price_arr = price.to_numpy()
    y_span = price_arr.max() - price_arr.min()

    duration_label = f"{horizon} bars"
    if minutes_per_bar:
        duration_label += f" (~{int(round(horizon * minutes_per_bar))}m)"

    for i in selected:
        exit_i = i + horizon
        is_long = strength[i] > 0
        color = "#2ca02c" if is_long else "#d62728"
        window_prices = price_arr[i : exit_i + 1]

        ax.axvspan(i, exit_i, color=color, alpha=0.12)
        y = (
            window_prices.max() + y_span * 0.03
            if is_long
            else window_prices.min() - y_span * 0.05
        )
        ax.annotate("", xy=(exit_i, y), xytext=(i, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
        ax.text((i + exit_i) / 2, y, duration_label, color=color, fontsize=7,
                ha="center", va="bottom" if is_long else "top")

    return selected


def rolling_ic(y_true: pd.Series, y_pred: np.ndarray, window: int, method: str = "spearman") -> pd.Series:
    """Trailing-window IC over time, to see whether predictive skill is stable
    or concentrated in specific regimes."""
    pred = pd.Series(y_pred, index=y_true.index)

    def _corr(sub_true: pd.Series) -> float:
        sub_pred = pred.loc[sub_true.index]
        if method == "spearman":
            return spearmanr(sub_true, sub_pred)[0]
        return sub_true.corr(sub_pred)

    return y_true.rolling(window).apply(lambda s: _corr(s), raw=False)
