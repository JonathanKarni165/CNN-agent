"""Zero-copy sliding windows over a scaled feature frame, for sequence models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """Window i covers rows [i, i+seq_len) and is paired with the target at
    the window's LAST row (i+seq_len-1). Since the target is already a
    forward-looking return from that bar, this stays causal: no row beyond
    i+seq_len-1 is used.

    Windows are built with `Tensor.unfold`, which is a view (no copy) into
    the underlying feature tensor, as called for in the spec.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, seq_len: int):
        if len(X) != len(y) or not X.index.equals(y.index):
            raise ValueError("X and y must be aligned on the same index")
        if len(X) <= seq_len:
            raise ValueError(f"Need more than seq_len={seq_len} rows, got {len(X)}")

        self.seq_len = seq_len
        values = torch.from_numpy(X.to_numpy(dtype=np.float32))  # [N, F]
        # -> [N - seq_len + 1, F, seq_len]: channels-first, directly matches Conv1d's
        # expected (batch, in_channels, length) input.
        self.windows = values.unfold(0, seq_len, 1)
        self.targets = torch.from_numpy(y.to_numpy(dtype=np.float32)[seq_len - 1 :])
        self.index = X.index[seq_len - 1 :]  # timestamp each window's prediction corresponds to

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, i: int):
        return self.windows[i], self.targets[i]
