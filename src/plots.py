"""Figures written by code. No hand-drawn charts."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.types import Vector


@dataclass(frozen=True)
class OverlayFigure:
    """Path of one returns-versus-fit figure.

    Attributes:
        path: image file written by the plotter.
        split_index: row where the test segment starts, or None for train only.
    """

    path: Path
    split_index: int | None


def plot_returns_overlay(
    path: Path,
    actual: Vector,
    fitted: Vector,
    *,
    split_index: int | None = None,
) -> OverlayFigure:
    """Plot R_t and R_hat_t on one pair of axes.

    When ``split_index`` is set, ``actual`` and ``fitted`` are the train series
    concatenated with the test series, and a vertical line marks the split.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    observed = np.asarray(actual, dtype=float)
    predicted = np.asarray(fitted, dtype=float)
    figure, axes = plt.subplots(figsize=(10, 4))
    steps = np.arange(len(observed))
    axes.plot(steps, observed, label="R_t")
    axes.plot(steps, predicted, label="fitted R_t")
    if split_index is not None:
        axes.axvline(split_index, color="black", linestyle="--", label="train/test split")
    axes.set_xlabel("design row")
    axes.set_ylabel("return")
    axes.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    return OverlayFigure(destination, split_index)
