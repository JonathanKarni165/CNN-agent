"""A single-line, in-place progress display for notebooks, so a training loop
doesn't flood limited notebook output with one printed line per step/round.
"""

from __future__ import annotations

import ipywidgets as widgets
from IPython.display import display
from xgboost.callback import TrainingCallback


class NotebookProgress:
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self._bar = widgets.IntProgress(min=0, max=total, description=description, layout=widgets.Layout(width="400px"))
        self._label = widgets.Label(value="")
        display(widgets.HBox([self._bar, self._label]))

    def update(self, step: int, **metrics: float) -> None:
        self._bar.value = step
        metrics_str = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        self._label.value = f"{step}/{self.total}  {metrics_str}"

    def close(self, final_message: str = "done") -> None:
        self._bar.value = self.total
        self._bar.bar_style = "success"
        self._label.value = final_message


class XGBProgressCallback(TrainingCallback):
    """xgboost.callback.TrainingCallback-compatible callback that drives a
    NotebookProgress bar from eval-set metrics instead of printing per round.
    """

    def __init__(self, progress: NotebookProgress, metric_name: str = "rmse"):
        super().__init__()
        self.progress = progress
        self.metric_name = metric_name

    def before_training(self, model):
        return model

    def after_training(self, model):
        return model

    def before_iteration(self, model, epoch, evals_log):
        return False

    def after_iteration(self, model, epoch, evals_log) -> bool:
        val_log = next(iter(evals_log.values()))
        metric_log = val_log.get(self.metric_name) or next(iter(val_log.values()))
        self.progress.update(epoch + 1, **{f"val_{self.metric_name}": metric_log[-1]})
        return False
