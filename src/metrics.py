"""Predictive-quality metrics: Information Coefficient and hit rate."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(pearsonr(y_true, y_pred)[0])


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(spearmanr(y_true, y_pred)[0])


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of correct sign predictions among |y_pred| > threshold."""
    mask = np.abs(y_pred) > threshold
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())
