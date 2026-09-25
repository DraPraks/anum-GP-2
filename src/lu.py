"""Dense LU with partial pivoting for the 6 x 6 normal matrix.

No library factorization. PA = LU is the intended identity once implemented.
"""

from dataclasses import dataclass

from src.types import Matrix, Vector


@dataclass(frozen=True)
class LuFactorization:
    """Dense factors of a square nonsingular matrix.

    Attributes:
        L: unit lower triangular, same order as the system.
        U: upper triangular.
        P: row-permutation matrix from partial pivoting (PB = LU).
        pivots: pivot row indices, length n, for reconstructing P.
    """

    L: Matrix
    U: Matrix
    P: Matrix
    pivots: Vector


def factor_lu(matrix: Matrix) -> LuFactorization:
    """Factor a dense square matrix with Gaussian elimination and partial pivoting.

    Intended for G = A^T A (6 x 6). Not a least-squares routine.

    Raises:
        NotImplementedError: placeholder.
    """
    del matrix
    raise NotImplementedError("factor_lu")


def solve_lu(factors: LuFactorization, rhs: Vector) -> Vector:
    """Solve L U y = P rhs by forward substitution then back substitution.

    Raises:
        NotImplementedError: placeholder.
    """
    del factors, rhs
    raise NotImplementedError("solve_lu")
