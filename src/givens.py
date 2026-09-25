"""Thin QR least squares by Givens rotations. No library QR."""

import time
from dataclasses import dataclass

import numpy as np

from src.lu import back_substitution
from src.types import Matrix, Vector


@dataclass(frozen=True)
class GivensRotation:
    """Plane rotation that zeros one subdiagonal entry.

    Attributes:
        i: row index kept (the pivot row).
        j: row index whose entry in column ``col`` is zeroed.
        col: column being eliminated.
        c: cosine.
        s: sine.
    """

    i: int
    j: int
    col: int
    c: float
    s: float


@dataclass(frozen=True)
class FirstColumnSweep:
    """Mandatory check: rotations that clear column 0 of the original A.

    Attributes:
        rotations: G_k applied to the first column, in order.
        after_each: A after each individual rotation (first, then the sweep).
        after_sweep: A after the full first-column sweep. Targeted entries
            must be numerically zero.
    """

    rotations: tuple[GivensRotation, ...]
    after_each: tuple[Matrix, ...]
    after_sweep: Matrix


@dataclass(frozen=True)
class GivensQrResult:
    """Thin QR least-squares result. Q is not stored on the fast path.

    Attributes:
        x: solution x_QR from backsolving R x = y, shape (6,).
        R: upper-triangular factor, shape (6, 6).
        y: transformed right-hand side, first 6 entries used in the backsolve.
        rotations: all plane rotations, applied on the fly to b.
        residual_norm: ||A x_QR - b||_2 against the original A and b.
        flops: theoretical flop count of the Givens sweep plus the backsolve.
        memory_bytes: fast path (rotators + R, b updated in place).
        memory_q_explicit_bytes: cost if Q were stored. ``q_layout`` says
            whether that figure is the full m x m factor or the thin m x 6 factor.
        q_layout: ``"full"`` or ``"thin"``.
        runtime_seconds: wall time of the fast path.
        first_column: stored G_k A evidence for the first column.
    """

    x: Vector
    R: Matrix
    y: Vector
    rotations: tuple[GivensRotation, ...]
    residual_norm: float
    flops: int
    memory_bytes: int
    memory_q_explicit_bytes: int
    q_layout: str
    runtime_seconds: float
    first_column: FirstColumnSweep


def _cosine_sine(pivot: float, target: float) -> tuple[float, float]:
    """Return c, s so that the rotation maps (pivot, target) to (r, 0)."""
    if pivot == 0.0 and target == 0.0:
        return 1.0, 0.0
    radius = float(np.hypot(pivot, target))
    return pivot / radius, target / radius


def _apply_row_rotation(matrix: Matrix, i: int, j: int, c: float, s: float, col: int) -> None:
    """Apply the plane rotation to rows i and j from ``col`` onward."""
    row_i = matrix[i, col:].copy()
    row_j = matrix[j, col:].copy()
    matrix[i, col:] = c * row_i + s * row_j
    matrix[j, col:] = -s * row_i + c * row_j


def _apply_vector_rotation(values: Vector, i: int, j: int, c: float, s: float) -> None:
    entry_i = values[i]
    entry_j = values[j]
    values[i] = c * entry_i + s * entry_j
    values[j] = -s * entry_i + c * entry_j


def verify_first_column_rotations(A: Matrix) -> FirstColumnSweep:
    """Build the Givens rotations that zero the first-column subdiagonal of A.

    Store G_k A after the first rotation and after the rest of the column sweep.
    The eliminated entries of that column must be numerically zero.
    """
    work = np.array(A, dtype=float, copy=True)
    rotations: list[GivensRotation] = []
    after_each: list[Matrix] = []
    for j in range(1, work.shape[0]):
        c, s = _cosine_sine(work[0, 0], work[j, 0])
        _apply_row_rotation(work, 0, j, c, s, 0)
        rotations.append(GivensRotation(0, j, 0, c, s))
        after_each.append(work.copy())
    if work.shape[0] > 1 and not np.allclose(work[1:, 0], 0.0, atol=1e-10):
        raise ValueError("first-column sweep left a nonzero subdiagonal entry")
    return FirstColumnSweep(tuple(rotations), tuple(after_each), work.copy())


def givens_least_squares(A: Matrix, b: Vector) -> GivensQrResult:
    """Thin QR: zero subdiagonal entries column by column, accumulate on b, backsolve.

    A is m x 6. Rotations are stored; Q is not formed. The residual is measured
    on the original A and b, not on the triangularized system.
    """
    design = np.array(A, dtype=float, copy=True)
    target = np.array(b, dtype=float, copy=True)
    original_a = np.asarray(A, dtype=float)
    original_b = np.asarray(b, dtype=float)
    m, n = design.shape
    rotations: list[GivensRotation] = []
    flops = 0

    started = time.perf_counter()
    for col in range(n):
        for j in range(col + 1, m):
            c, s = _cosine_sine(design[col, col], design[j, col])
            _apply_row_rotation(design, col, j, c, s, col)
            _apply_vector_rotation(target, col, j, c, s)
            rotations.append(GivensRotation(col, j, col, c, s))
            # Six flops per updated entry of the two rows, including b.
            flops += 6 * (n - col) + 6
    factor = np.triu(design[:n, :])
    x = back_substitution(factor, target[:n])
    flops += n * n
    runtime_seconds = time.perf_counter() - started

    residual = original_a @ x - original_b
    return GivensQrResult(
        x=x,
        R=factor,
        y=target,
        rotations=tuple(rotations),
        residual_norm=float(np.sqrt(residual @ residual)),
        flops=flops,
        memory_bytes=n * n * 8 + m * 8 + len(rotations) * 16,
        memory_q_explicit_bytes=m * n * 8,
        q_layout="thin",
        runtime_seconds=runtime_seconds,
        first_column=verify_first_column_rotations(A),
    )
