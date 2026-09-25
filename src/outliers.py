"""Rank train rows by absolute residual for the flash-crash discussion."""

from dataclasses import dataclass

import numpy as np

from src.types import Vector


@dataclass(frozen=True)
class ResidualOutlier:
    """One design-matrix row among the largest absolute residuals.

    Attributes:
        row: index into A / b (0-based within the design system).
        date: calendar label of that row, if the caller stored one.
        residual: signed residual (b - A x)_row.
        abs_residual: absolute residual used for ranking.
    """

    row: int
    date: str | None
    residual: float
    abs_residual: float


def rank_residual_outliers(
    residual: Vector,
    dates: Vector | None = None,
    keep: int = 5,
) -> tuple[ResidualOutlier, ...]:
    """Return the ``keep`` worst train residuals by absolute size.

    ``residual`` is b - A x on the training design. This is evidence only;
    it does not drop rows or refit.
    """
    values = np.asarray(residual, dtype=float)
    order = np.argsort(-np.abs(values))[:keep]
    ranked: list[ResidualOutlier] = []
    for index in order:
        row = int(index)
        label = None if dates is None else str(dates[row])
        signed = float(values[row])
        ranked.append(ResidualOutlier(row, label, signed, abs(signed)))
    return tuple(ranked)
