"""Cyclical time features: intraday session position and day-of-week.

These capture recurring microstructure effects a model can't infer from
price/volume alone — e.g. the midday lull ("lunch effect") and day-of-week
patterns — without needing any rolling window (warmup = 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Feature, register

MINUTES_IN_SESSION = 390  # 6.5h regular session (09:30-16:00 ET)
MARKET_OPEN_MINUTES = 9 * 60 + 30
TRADING_DAYS_PER_WEEK = 5


def _minutes_since_open(index: pd.DatetimeIndex) -> pd.Series:
    minutes_of_day = index.hour * 60 + index.minute
    return pd.Series(minutes_of_day - MARKET_OPEN_MINUTES, index=index)


@register
class SessionTimeSinFeature(Feature):
    """sin(2*pi*m/390), m = minutes since market open."""

    @property
    def name(self) -> str:
        return "session_time_sin"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _minutes_since_open(df.index)
        return np.sin(2 * np.pi * m / MINUTES_IN_SESSION)


@register
class SessionTimeCosFeature(Feature):
    """cos(2*pi*m/390), m = minutes since market open."""

    @property
    def name(self) -> str:
        return "session_time_cos"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _minutes_since_open(df.index)
        return np.cos(2 * np.pi * m / MINUTES_IN_SESSION)


@register
class DayOfWeekSinFeature(Feature):
    """sin(2*pi*d/5), d in {0..4} (Monday=0 .. Friday=4)."""

    @property
    def name(self) -> str:
        return "day_of_week_sin"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        d = pd.Series(df.index.dayofweek, index=df.index)
        return np.sin(2 * np.pi * d / TRADING_DAYS_PER_WEEK)


@register
class DayOfWeekCosFeature(Feature):
    """cos(2*pi*d/5), d in {0..4} (Monday=0 .. Friday=4)."""

    @property
    def name(self) -> str:
        return "day_of_week_cos"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        d = pd.Series(df.index.dayofweek, index=df.index)
        return np.cos(2 * np.pi * d / TRADING_DAYS_PER_WEEK)
