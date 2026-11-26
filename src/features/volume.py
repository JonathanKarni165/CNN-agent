"""Volume features: relative volume and signed volume flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Feature, register


@register
class RelativeVolumeFeature(Feature):
    """RelVol_{t,W} = V_t / (Mean(V_{t-W+1..t}) + eps)"""

    def __init__(self, window: int, volume_col: str = "volume", eps: float = 1e-8):
        self.window = window
        self.volume_col = volume_col
        self.eps = eps

    @property
    def name(self) -> str:
        return f"rel_volume_{self.window}"

    @property
    def warmup(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        volume = df[self.volume_col]
        mean_volume = volume.rolling(self.window).mean()
        return volume / (mean_volume + self.eps)


@register
class SignedVolumeFlowFeature(Feature):
    """SignedVol_t = V_t * sign(P_t - P_{t-1})"""

    def __init__(self, price_col: str = "close", volume_col: str = "volume"):
        self.price_col = price_col
        self.volume_col = volume_col

    @property
    def name(self) -> str:
        return "signed_volume"

    @property
    def warmup(self) -> int:
        return 1

    def compute(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.price_col]
        volume = df[self.volume_col]
        return volume * np.sign(price - price.shift(1))
