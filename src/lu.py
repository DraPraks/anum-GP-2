"""Dense LU with partial pivoting for the 6 x 6 normal matrix.

No library factorization. PA = LU is the intended identity once implemented.
"""

from dataclasses import dataclass

import numpy as np

from src.types import Matrix, Vector


@dataclass(frozen=True)
class LuFactorization:
    """Dense factors of a square nonsingular matrix.

    Attributes:
        L: unit lower triangular, same order as the system.
        U: upper triangular.
        P: row-permutation matrix from partial pivoting (PA = LU).
        pivots: pivot row indices, length n, for reconstructing P.
    """

    L: Matrix
    U: Matrix
    P: Matrix
    pivots: Vector


def factor_lu(matrix: Matrix) -> LuFactorization:
    """Factor a dense square matrix with Gaussian elimination and partial pivoting.

    algorithm:
    1. at each column k, search rows i=k, k+1, ..., n, the pivot row is i = argmax_{i > k} |a_ik|
    1.1 if the largest magnitude is ~0, matrix is singular and raise ValueError
    2. swap rows i and pivot_row, record in P
    2.1 swap entries in L and U at column k (keep L's diagonal elements)
    3. set pivot p = a_kk
    3. for each row i > k:
        3.1 m_i = a_ik / p
        3.2 store m_i in L_ik
        3.2 row_i = row_i - m_i * row_k, assert writes 0 at column k

    Raises:
        ValueError: the matrix is singular at the chosen pivot tolerance.
    """
    U = np.array(matrix, dtype=float, copy=True)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("matrix must be square")
    n = U.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    pivots = np.empty(n, dtype=int)

    for k in range(n):
        pivot_row = k
        for i in range(k, n):
            if abs(U[i, k]) > abs(U[pivot_row, k]):
                pivot_row = i

        pivots[k] = pivot_row
        if abs(U[pivot_row, k]) < 1e-10:
            raise ValueError("Matrix is singular")

        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        for i in range(k + 1, n):
            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            U[i, k:] -= multiplier * U[k, k:]
            U[i, k] = 0.0

    return LuFactorization(L, U, P, pivots)


def forward_substitution(L: Matrix, rhs: Vector) -> Vector:
    """Solve L y = rhs for unit lower-triangular L."""
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = rhs[i] - L[i, :i] @ y[:i]
    return y


def back_substitution(U: Matrix, rhs: Vector) -> Vector:
    """Solve U x = rhs for upper-triangular U."""
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            raise ValueError("U has a zero diagonal entry")
        x[i] = (rhs[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def solve_lu(factors: LuFactorization, rhs: Vector) -> Vector:
    """Solve L U x = P rhs by forward substitution then back substitution."""
    permuted = factors.P @ np.asarray(rhs, dtype=float)
    y = forward_substitution(factors.L, permuted)
    return back_substitution(factors.U, y)
