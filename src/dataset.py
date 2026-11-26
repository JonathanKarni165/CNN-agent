"""Assemble the feature matrix + target, and split it chronologically with
a purged embargo gap so no information leaks across train/val/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .features import FeatureEngine
from .target import ForwardLogReturnTarget


def build_dataset(
    df: pd.DataFrame,
    feature_engine: FeatureEngine,
    target: ForwardLogReturnTarget,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute features + target, drop rows invalid due to feature warm-up or
    missing future price (target warm-up), while keeping chronological order.
    """
    features = feature_engine.transform(df)
    y = target.compute(df)

    valid = features["valid"] & y.notna()
    X = features.loc[valid, feature_engine.feature_names]
    y = y.loc[valid]
    return X, y


@dataclass
class SplitIndices:
    train: pd.Index
    val: pd.Index
    test: pd.Index


class ChronologicalSplitter:
    """Train/Val/Test split by time order, with an embargo gap of `embargo`
    rows purged at each boundary so no sequence window or target horizon
    straddles two splits.
    """

    def __init__(self, train_frac: float = 0.7, val_frac: float = 0.1, embargo: int = 0):
        if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
            raise ValueError("train_frac + val_frac must be < 1, each in (0, 1)")
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.embargo = embargo

    def split(self, index: pd.Index) -> SplitIndices:
        n = len(index)
        train_end = int(n * self.train_frac)
        val_end = train_end + int(n * self.val_frac)

        train_idx = index[:train_end]
        val_idx = index[train_end + self.embargo : val_end]
        test_idx = index[val_end + self.embargo :]
        return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


class CausalScaler:
    """Fits a scaler on the train split only; transforms all splits with it."""

    def __init__(self, method: str = "standard", clip_sigma: float = 10.0):
        if method not in ("standard", "robust"):
            raise ValueError("method must be 'standard' or 'robust'")
        self._scaler = StandardScaler() if method == "standard" else RobustScaler()
        self.clip_sigma = clip_sigma

    def fit(self, X_train: pd.DataFrame) -> "CausalScaler":
        self._scaler.fit(X_train.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self._scaler.transform(X.values)
        values = np.clip(values, -self.clip_sigma, self.clip_sigma)
        return pd.DataFrame(values, index=X.index, columns=X.columns)
