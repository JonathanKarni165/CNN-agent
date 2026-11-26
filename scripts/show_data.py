"""Quick visual check of the fetched SPY intraday data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import fetch_ohlcv

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _break_between_days(df):
    """Insert a NaN row at each day boundary so plot lines don't bridge overnight gaps."""
    day = df.index.date
    boundary = pd.Series(day).ne(pd.Series(day).shift()).to_numpy()
    boundary[0] = False
    if not boundary.any():
        return df
    gap_index = df.index[boundary] - pd.Timedelta(seconds=1)
    gap_rows = pd.DataFrame(index=gap_index, columns=df.columns, dtype=float)
    return pd.concat([df, gap_rows]).sort_index()


def _bar_label(df: pd.DataFrame) -> str:
    minutes = int(pd.Series(df.index).diff().dt.total_seconds().median() / 60)
    return f"{minutes}-min"


def plot_price_volume(df: pd.DataFrame) -> Path:
    plot_df = _break_between_days(df)
    bar_label = _bar_label(df)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(plot_df.index, plot_df["close"], linewidth=0.8, color="#1f77b4")
    ax_price.set_title(f"SPY {bar_label} Close — {df.index[0].date()} to {df.index[-1].date()} ({len(df)} bars)")
    ax_price.set_ylabel("Price ($)")
    ax_price.grid(alpha=0.3)

    ax_vol.bar(df.index, df["volume"], width=0.0005, color="#888888")
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "spy_price_volume.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_by_day(df: pd.DataFrame, max_days: int = 12) -> Path:
    """Small multiples: one intraday close curve per trading day (most recent `max_days`)."""
    days = sorted(set(df.index.date))[-max_days:]
    n = len(days)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)

    for i, day in enumerate(days):
        ax = axes[i // cols][i % cols]
        day_df = df[df.index.date == day]
        pct = (day_df["close"] / day_df["close"].iloc[0] - 1) * 100
        ax.plot(day_df.index.strftime("%H:%M"), pct, linewidth=0.8, color="#d62728")
        ax.set_title(str(day))
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xticks(ax.get_xticks()[:: max(1, len(day_df) // 6)])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("% from open")
        ax.grid(alpha=0.3)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("SPY Intraday Path by Day")
    fig.tight_layout()
    out_path = OUT_DIR / "spy_by_day.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_returns_hist(df: pd.DataFrame) -> Path:
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log_ret, bins=80, color="#2ca02c", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Distribution of {_bar_label(df)} Log Returns (n={len(log_ret)})\n"
                 f"mean={log_ret.mean():.2e}, std={log_ret.std():.2e}")
    ax.set_xlabel("log return")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "spy_returns_hist.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    df = fetch_ohlcv()
    OUT_DIR.mkdir(exist_ok=True)

    paths = [
        plot_price_volume(df),
        plot_by_day(df),
        plot_returns_hist(df),
    ]
    for p in paths:
        print(f"Saved plot to {p}")
    print(df.describe())


if __name__ == "__main__":
    main()
