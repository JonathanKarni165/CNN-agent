"""Render the model's buy/sell signal painted on the SPY price chart."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.visualization import plot_price_with_signal, plot_holding_windows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
SPLIT_DIR = DATA_DIR / "splits"
OUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    target_col = [c for c in test.columns if c.startswith("fwd_log_return")][0]
    feature_cols = [c for c in test.columns if c != target_col]
    horizon = int(target_col.rsplit("_", 1)[1])
    X_test = test[feature_cols]

    raw_close = pd.read_parquet(DATA_DIR / "raw_close.parquet")["close"]
    price_test = raw_close.loc[test.index]

    results = pd.read_parquet(MODEL_DIR / "results.parquet")
    best_name = results["spearman_ic"].idxmax()

    available_models = {p.stem: joblib.load(p) for p in MODEL_DIR.glob("*.joblib")}
    model_key = next(k for k, m in available_models.items() if m.name == best_name)
    model = available_models[model_key]
    print("Visualizing:", model.name)

    pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(15, 5))
    plot_price_with_signal(
        price_test.index, price_test, pred,
        title=f"{model.name} — predicted signal + holding windows over SPY close (test set)",
        ax=ax,
    )
    plot_holding_windows(
        ax, price_test, pred, horizon=horizon, index=price_test.index,
        max_windows=10, strength_threshold=0.85,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "signal_on_price.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
