"""Regime-detection features: realized volatility of 1-step returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Feature, register


@register
class RealizedVolatilityFeature(Feature):
    """sigma^ret_{t,W} = StdDev(r_{t-W+1..t, 1})"""

    def __init__(self, window: int, price_col: str = "close"):
        self.window = window
        self.price_col = price_col

    @property
    def name(self) -> str:
        return f"realized_vol_{self.window}"

    @property
    def warmup(self) -> int:
        return self.window + 1

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.price_col]
        ret_1 = np.log(price / price.shift(1))
        return ret_1.rolling(self.window).std(ddof=0)
