"""Strategy interface for a single causal rolling feature, plus a registry
that lets new Feature subclasses be discovered without touching the engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Feature(ABC):
    """A feature knows only how to compute one column from raw OHLCV data.

    Implementations MUST be causal: the value at row t may only depend on
    rows <= t (no lookahead bias).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique output column name."""

    @property
    def warmup(self) -> int:
        """Number of leading rows that are invalid (default: none)."""
        return 0

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the feature column from the input OHLCV frame."""


_REGISTRY: dict[str, type[Feature]] = {}


def register(cls: type[Feature]) -> type[Feature]:
    """Class decorator: add a Feature subclass to the global registry by name."""
    _REGISTRY[cls.__name__] = cls
    return cls


def available_features() -> dict[str, type[Feature]]:
    """All Feature classes registered so far (e.g. for a config-driven factory)."""
    return dict(_REGISTRY)
