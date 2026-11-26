"""Momentum features: multi-horizon log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Feature, register


@register
class LogReturnFeature(Feature):
    """r_{t,h} = ln(P_t / P_{t-h})"""

    def __init__(self, horizon: int, price_col: str = "close"):
        self.horizon = horizon
        self.price_col = price_col

    @property
    def name(self) -> str:
        return f"log_return_{self.horizon}"

    @property
    def warmup(self) -> int:
        return self.horizon

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.price_col]
        return np.log(price / price.shift(self.horizon))
