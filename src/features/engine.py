"""FeatureEngine: composes Feature strategies into one causal feature matrix."""

from __future__ import annotations

import pandas as pd

from .base import Feature


class FeatureEngine:
    """Context/Facade that runs a list of Feature strategies over OHLCV data.

    Adding a new feature never requires changing this class: implement a new
    Feature subclass and pass an instance into the engine (or the registry).
    """

    def __init__(self, features: list[Feature] | None = None):
        self._features: list[Feature] = list(features) if features else []

    def add(self, feature: Feature) -> "FeatureEngine":
        self._features.append(feature)
        return self

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self._features]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute every feature and append a `valid` column masking warm-up rows."""
        columns = {feature.name: feature.compute(df) for feature in self._features}
        result = pd.DataFrame(columns, index=df.index)

        max_warmup = max((f.warmup for f in self._features), default=0)
        valid = pd.Series(True, index=df.index)
        valid.iloc[:max_warmup] = False
        result["valid"] = valid
        return result
