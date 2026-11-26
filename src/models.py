"""Model wrappers behind one common interface (Strategy pattern, same idea as
src/features) so the training notebook can swap Ridge <-> XGBoost <-> (later)
the CNN+GRU without changing its own code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

# Loss functions available for XGBoostModel's `loss` argument.
XGB_OBJECTIVES = {
    "squared": "reg:squarederror",
    "mae": "reg:absoluteerror",
    "huber": "reg:pseudohubererror",
}


class BaseModel(ABC):
    """Common interface every model tier implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class RidgeModel(BaseModel):
    """Tier 1 linear baseline. Ridge always minimizes squared error + L2;
    there is no alternate loss to pick here (see XGBoostModel/the CNN for that).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)

    @property
    def name(self) -> str:
        return f"Ridge(alpha={self.alpha})"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self._model.fit(X_train.values, y_train.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.values)


class XGBoostModel(BaseModel):
    """Tier 2 gradient-boosted trees. `loss` selects the training objective:
    'squared' (RMSE), 'mae', or 'huber' (robust to fat-tailed market jumps).
    """

    def __init__(
        self,
        loss: str = "huber",
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.7,
        colsample_bytree: float = 0.7,
        early_stopping_rounds: int = 30,
    ):
        import xgboost as xgb

        if loss not in XGB_OBJECTIVES:
            raise ValueError(f"loss must be one of {list(XGB_OBJECTIVES)}")
        self.loss = loss
        self.early_stopping_rounds = early_stopping_rounds
        self._model = xgb.XGBRegressor(
            objective=XGB_OBJECTIVES[loss],
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
        )

    @property
    def name(self) -> str:
        return f"XGBoost(loss={self.loss})"

    def fit(self, X_train, y_train, X_val, y_val, callbacks: list | None = None) -> None:
        if callbacks:
            self._model.set_params(callbacks=callbacks)
        self._model.fit(
            X_train.values,
            y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )
        if callbacks:
            # Drop the (possibly non-picklable, e.g. notebook-widget-backed) callbacks
            # now that training is done, so the fitted model can still be saved.
            self._model.set_params(callbacks=None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.values)

    @property
    def best_iteration(self) -> int:
        return self._model.best_iteration


# Torch loss names available for CNNGRUModel's `loss` argument.
TORCH_LOSS_NAMES = ("huber", "mse", "mae")


class QuantTimeSeriesNet(nn.Module):
    """Conv1d -> GELU -> Dropout -> GRU -> last hidden state -> Dropout -> Linear(1).

    A plain module-level class (rather than built inside a function) so a
    trained CNNGRUModel — which holds one of these — can still be pickled.
    """

    def __init__(self, n_features: int, conv_channels: int = 32, gru_hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.conv = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.gru = nn.GRU(conv_channels, gru_hidden, num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(self, x):  # x: [batch, F, seq_len]
        x = self.drop1(self.act(self.conv(x)))  # [batch, C, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, C] for GRU (batch_first)
        out, _ = self.gru(x)
        last = self.drop2(out[:, -1, :])
        return self.head(last).squeeze(-1)


class CNNGRUModel(BaseModel):
    """Tier 3 temporal deep-learning model (Conv1d -> GRU -> Linear). `loss`
    selects the training objective: 'huber' (default, robust to fat-tailed
    market jumps per spec), 'mse', or 'mae'.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 30,
        loss: str = "huber",
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 50,
        patience: int = 8,
        batch_size: int = 256,
        device: str | None = None,
    ):
        if loss not in TORCH_LOSS_NAMES:
            raise ValueError(f"loss must be one of {TORCH_LOSS_NAMES}")
        self.n_features = n_features
        self.seq_len = seq_len
        self.loss_name = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = QuantTimeSeriesNet(n_features).to(self.device)
        self.best_val_ic = -float("inf")

    @property
    def name(self) -> str:
        return f"CNN+GRU(loss={self.loss_name})"

    def fit(self, X_train, y_train, X_val, y_val, progress=None) -> None:
        from torch.utils.data import DataLoader

        from .metrics import spearman_ic
        from .sequence_dataset import SlidingWindowDataset

        train_ds = SlidingWindowDataset(X_train, y_train, self.seq_len)
        val_ds = SlidingWindowDataset(X_val, y_val, self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        criterion = {"huber": nn.HuberLoss(delta=1.0), "mse": nn.MSELoss(), "mae": nn.L1Loss()}[self.loss_name]
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

        best_state = None
        epochs_without_improvement = 0
        val_X = val_ds.windows.to(self.device)
        val_y_true = val_ds.targets.numpy()

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(val_X).cpu().numpy()
            val_ic = spearman_ic(val_y_true, val_pred)
            if np.isnan(val_ic):
                val_ic = -1.0
            scheduler.step(val_ic)

            if progress is not None:
                progress.update(epoch, val_spearman_ic=val_ic, lr=optimizer.param_groups[0]["lr"])

            if val_ic > self.best_val_ic:
                self.best_val_ic = val_ic
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predictions for rows >= seq_len - 1 only (the first
        seq_len - 1 rows have no full window yet); align the caller's `y`
        with `X.index[self.seq_len - 1:]`.
        """
        values = torch.from_numpy(X.to_numpy(dtype=np.float32))
        windows = values.unfold(0, self.seq_len, 1).to(self.device)
        self.net.eval()
        with torch.no_grad():
            pred = self.net(windows).cpu().numpy()
        return pred

    def to_cpu(self) -> "CNNGRUModel":
        """Move the network to CPU before saving, so loading it later doesn't
        require the same GPU availability as when it was trained."""
        self.net = self.net.to("cpu")
        self.device = "cpu"
        return self
