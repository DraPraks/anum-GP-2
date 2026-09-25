"""Least squares via the normal equations G x = c."""

from dataclasses import dataclass

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


def solve_normal_equations(A: Matrix, b: Vector) -> NormalEquationResult:
    """Form G = A^T A and c = A^T b, then solve G x = c with the dense LU solver.

    Raises:
        NotImplementedError: placeholder.
    """
    del A, b
    raise NotImplementedError("solve_normal_equations")
