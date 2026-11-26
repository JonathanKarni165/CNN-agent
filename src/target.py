"""Target construction: forward log-return over a fixed horizon.

y_t = ln(P_{t+k} / P_t). The last k rows are invalid (no future price yet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForwardLogReturnTarget:
    def __init__(self, horizon: int = 5, price_col: str = "close"):
        self.horizon = horizon
        self.price_col = price_col

    @property
    def name(self) -> str:
        return f"fwd_log_return_{self.horizon}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.price_col]
        y = np.log(price.shift(-self.horizon) / price)
        y.name = self.name
        return y
