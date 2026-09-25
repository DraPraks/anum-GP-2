"""Least squares via the normal equations G x = c."""

import time
from dataclasses import dataclass

import numpy as np

from src.lu import factor_lu, solve_lu
from src.metrics import condition_number_2
from src.types import Matrix, Vector


@dataclass(frozen=True)
class NormalEquationResult:
    """Record of one normal-equation solve on a fixed (A, b).

    Attributes:
        x: solution x_NE, shape (6,).
        G: Gram matrix A^T A, shape (6, 6).
        c: A^T b, shape (6,).
        residual_norm: ||A x_NE - b||_2 against the original system.
        kappa_A: 2-norm condition number of A.
        kappa_G: 2-norm condition number of A^T A.
        flops: theoretical flop count for forming G, c and the 6 x 6 solve.
        memory_bytes: working set of forming G plus the dense solve.
        runtime_seconds: wall time of that working set.
    """

    x: Vector
    G: Matrix
    c: Vector
    residual_norm: float
    kappa_A: float
    kappa_G: float
    flops: int
    memory_bytes: int
    runtime_seconds: float


def _normal_equation_flops(m: int, n: int) -> int:
    """Leading-term flops for the symmetric Gram product, c, and the dense LU solve.

    A multiply and an add each count as one flop. The Gram product uses the
    upper triangle, including the diagonal. Dense LU is the classical
    2 n^3 / 3 elimination count plus two triangular substitutions.
    """
    gram = (n * (n + 1) // 2) * (2 * m - 1)
    rhs = n * (2 * m - 1)
    elimination = (2 * n * n * n) // 3
    substitution = 2 * n * n
    return gram + rhs + elimination + substitution


def solve_normal_equations(A: Matrix, b: Vector) -> NormalEquationResult:
    """Form G = A^T A and c = A^T b, then solve G x = c with the dense LU solver."""
    design = np.asarray(A, dtype=float)
    target = np.asarray(b, dtype=float)
    m, n = design.shape

    started = time.perf_counter()
    gram = design.T @ design
    rhs = design.T @ target
    factors = factor_lu(gram)
    x = solve_lu(factors, rhs)
    runtime_seconds = time.perf_counter() - started

    residual = design @ x - target
    memory_bytes = (
        gram.nbytes + rhs.nbytes + factors.L.nbytes + factors.U.nbytes + factors.P.nbytes + factors.pivots.nbytes
    )
    return NormalEquationResult(
        x=x,
        G=gram,
        c=rhs,
        residual_norm=float(np.sqrt(residual @ residual)),
        kappa_A=condition_number_2(design),
        kappa_G=condition_number_2(gram),
        flops=_normal_equation_flops(m, n),
        memory_bytes=memory_bytes,
        runtime_seconds=runtime_seconds,
    )
