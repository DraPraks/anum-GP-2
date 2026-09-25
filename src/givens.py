"""Thin QR least squares by Givens rotations. No library QR.

TODO: implement the Givens sweep, the first-column check, and the backsolve.
"""

from dataclasses import dataclass

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


def verify_first_column_rotations(A: Matrix) -> FirstColumnSweep:
    """Build the Givens rotations that zero the first-column subdiagonal of A.

    Store G_k A after the first rotation and after the rest of the column sweep.
    The eliminated entries of that column must be numerically zero.

    Raises:
        NotImplementedError: TODO.
    """
    del A
    raise NotImplementedError("verify_first_column_rotations")


def givens_least_squares(A: Matrix, b: Vector) -> GivensQrResult:
    """Thin QR: zero subdiagonal entries column by column, accumulate on b, backsolve.

    A is m x 6. Rotations are stored; Q is not formed. The residual is measured
    on the original A and b, not on the triangularized system.

    Raises:
        NotImplementedError: TODO.
    """
    del A, b
    raise NotImplementedError("givens_least_squares")
