"""Fetch SPY intraday OHLCV bars via yfinance and filter to regular market hours."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"


def fetch_ohlcv(ticker: str = "SPY", interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    """Download intraday bars and restrict to regular US market hours (America/New_York).

    yfinance limits 1-minute bars to the trailing ~7-8 days regardless of `period`.
    """
    raw = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker} ({interval}, {period})")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index.name = "timestamp"

    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert("America/New_York")

    raw = raw.between_time(MARKET_OPEN, MARKET_CLOSE)
    return raw[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    df = fetch_ohlcv()
    print(df.shape)
    print(df.head())
    print(df.tail())
    print(df.describe())
