"""Build a small feature set with FeatureEngine and inspect the result."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from scripts.show_data import _break_between_days
from src.data_loader import fetch_ohlcv
from src.features import (
    DayOfWeekCosFeature,
    DayOfWeekSinFeature,
    FeatureEngine,
    LogReturnFeature,
    RangePositionFeature,
    RealizedVolatilityFeature,
    RelativeVolumeFeature,
    RollingZScoreFeature,
    SessionTimeCosFeature,
    SessionTimeSinFeature,
    SignedVolumeFlowFeature,
    available_features,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def build_engine() -> FeatureEngine:
    return FeatureEngine([
        LogReturnFeature(horizon=1),
        LogReturnFeature(horizon=5),
        RollingZScoreFeature(window=30),
        RangePositionFeature(window=30),
        RealizedVolatilityFeature(window=30),
        SessionTimeSinFeature(),
        SessionTimeCosFeature(),
        DayOfWeekSinFeature(),
        DayOfWeekCosFeature(),
        RelativeVolumeFeature(window=30),
        SignedVolumeFlowFeature(),
    ])


def main() -> None:
    print("Registered features:", list(available_features().keys()))

    df = fetch_ohlcv()
    engine = build_engine()
    features = engine.transform(df)

    print("\nFeature columns:", engine.feature_names)
    print(f"Rows: {len(features)}, valid rows (post warm-up): {features['valid'].sum()}")
    print("\nHead (includes warm-up NaNs):")
    print(features.head(3))
    print("\nTail:")
    print(features.tail(3))
    print("\nDescribe (valid rows only):")
    print(features.loc[features["valid"]].describe())

    valid = _break_between_days(features.loc[features["valid"]])
    fig, axes = plt.subplots(len(engine.feature_names), 1, figsize=(12, 2.2 * len(engine.feature_names)), sharex=True)
    for ax, col in zip(axes, engine.feature_names):
        ax.plot(valid.index, valid[col], linewidth=0.7)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Feature preview (post warm-up)")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "features_preview.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
