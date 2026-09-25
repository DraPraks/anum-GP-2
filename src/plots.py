"""Figures written by code. No hand-drawn charts."""

from dataclasses import dataclass
from pathlib import Path

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

    Raises:
        NotImplementedError: placeholder; nothing is written.
    """
    del path, actual, fitted, split_index
    raise NotImplementedError("plot_returns_overlay")
