"""Public API for the feature engine package.

Importing this package registers all built-in Feature implementations, so
`available_features()` reflects everything that ships here without callers
needing to import each module by hand.
"""

from .base import Feature, available_features, register
from .engine import FeatureEngine
from .mean_reversion import RangePositionFeature, RollingZScoreFeature
from .momentum import LogReturnFeature
from .seasonality import (
    DayOfWeekCosFeature,
    DayOfWeekSinFeature,
    SessionTimeCosFeature,
    SessionTimeSinFeature,
)
from .volatility import RealizedVolatilityFeature
from .volume import RelativeVolumeFeature, SignedVolumeFlowFeature

__all__ = [
    "Feature",
    "FeatureEngine",
    "available_features",
    "register",
    "LogReturnFeature",
    "RollingZScoreFeature",
    "RangePositionFeature",
    "RealizedVolatilityFeature",
    "SessionTimeSinFeature",
    "SessionTimeCosFeature",
    "DayOfWeekSinFeature",
    "DayOfWeekCosFeature",
    "RelativeVolumeFeature",
    "SignedVolumeFlowFeature",
]
