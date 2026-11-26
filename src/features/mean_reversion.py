"""Mean-reversion features: rolling z-score and range position."""

from __future__ import annotations

import pandas as pd

from .base import Feature, register


@register
class RollingZScoreFeature(Feature):
    """Z_{t,W} = (P_t - mu_{t,W}) / (sigma_{t,W} + eps)"""

    def __init__(self, window: int, price_col: str = "close", eps: float = 1e-8):
        self.window = window
        self.price_col = price_col
        self.eps = eps

    @property
    def name(self) -> str:
        return f"zscore_{self.window}"

    @property
    def warmup(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.price_col]
        rolling = price.rolling(self.window)
        mu = rolling.mean()
        sigma = rolling.std(ddof=0)
        return (price - mu) / (sigma + self.eps)


@register
class RangePositionFeature(Feature):
    """Pos_{t,W} = (P_t - min_W) / (max_W - min_W + eps), in [0, 1]"""

    def __init__(self, window: int, price_col: str = "close", eps: float = 1e-8):
        self.window = window
        self.price_col = price_col
        self.eps = eps

    @property
    def name(self) -> str:
        return f"range_pos_{self.window}"

    @property
    def warmup(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.price_col]
        rolling_min = price.rolling(self.window).min()
        rolling_max = price.rolling(self.window).max()
        return (price - rolling_min) / (rolling_max - rolling_min + self.eps)
