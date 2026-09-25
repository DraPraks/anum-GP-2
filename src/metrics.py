"""Residuals, RMSE, and the comparison row both solvers must fill."""

from dataclasses import dataclass

import numpy as np

from src.design import DesignSystem
from src.types import Matrix, Vector


@dataclass(frozen=True)
class FitMetrics:
    """Train-or-test scores for one frozen coefficient vector.

    Attributes:
        residual_norm: ||A x - b||_2 on this split. Train only for the table
            residual column; still defined on test if a design system is passed.
        rmse: sqrt(mean((R_t - R_hat_t)^2)) over the m design rows.
        kappa_A: 2-norm condition number of this split's A.
        kappa_ata: 2-norm condition number of A^T A. Reported for the normal
            equations; QR shares kappa_A and leaves kappa(A^T A) blank.
    """

    residual_norm: float
    rmse: float
    kappa_A: float
    kappa_ata: float | None


@dataclass(frozen=True)
class ComparisonRow:
    """One column of the experiment comparison table.

    ``memory_q_explicit_bytes`` is None for normal equations (table cell n/a).
    ``kappa_ata`` is None for Givens QR.
    """

    method: str
    flops: int
    memory_bytes: int
    memory_q_explicit_bytes: int | None
    kappa_A: float
    kappa_ata: float | None
    residual_norm_train: float
    rmse_train: float
    rmse_test: float
    runtime_seconds: float


def fitted_returns(system: DesignSystem, x: Vector) -> Vector:
    """SETAR fitted values R_hat_t = A_{t,:} x.

    The row recipe is already encoded in ``system.A`` (bull or bear from R_{t-1}).
    Coefficients stay frozen; this does not refit.
    """
    return np.asarray(system.A, dtype=float) @ np.asarray(x, dtype=float)


def rmse(actual: Vector, predicted: Vector) -> float:
    """Root mean square error of returns on one split."""
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(error * error)))


def _jacobi_eigenvalues(symmetric: Matrix, tol: float = 1e-15, max_sweeps: int = 100) -> Vector:
    """Eigenvalues of a symmetric matrix by classical Jacobi rotations."""
    s = np.array(symmetric, dtype=float, copy=True)
    s = 0.5 * (s + s.T)
    n = s.shape[0]
    for _ in range(max_sweeps):
        off = 0.0
        ip = iq = 0
        for p in range(n - 1):
            for q in range(p + 1, n):
                magnitude = abs(s[p, q])
                if magnitude > off:
                    off = magnitude
                    ip, iq = p, q
        if off <= tol:
            break
        app, aqq, apq = s[ip, ip], s[iq, iq], s[ip, iq]
        tau = (aqq - app) / (2.0 * apq)
        if tau >= 0.0:
            t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
        c = 1.0 / np.sqrt(1.0 + t * t)
        rot = t * c
        for k in range(n):
            if k == ip or k == iq:
                continue
            sik, siq = s[k, ip], s[k, iq]
            s[k, ip] = s[ip, k] = c * sik - rot * siq
            s[k, iq] = s[iq, k] = rot * sik + c * siq
        s[ip, ip] = c * c * app - 2.0 * rot * c * apq + rot * rot * aqq
        s[iq, iq] = rot * rot * app + 2.0 * rot * c * apq + c * c * aqq
        s[ip, iq] = s[iq, ip] = 0.0
    return np.diag(s)


def condition_number_2(matrix: Matrix) -> float:
    """2-norm condition number. Derivation may use SVD implemented in-repo later.

    Library ``cond`` / ``svd`` helpers are out of scope for the solver path.
    This hook exists so both methods can record kappa_2(A) and kappa_2(A^T A).

    Singular values are square roots of the Jacobi eigenvalues of MᵀM.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("matrix must be a non-empty 2-D array")
    eigenvalues = _jacobi_eigenvalues(a.T @ a)
    sigma_max = float(np.sqrt(max(eigenvalues.max(), 0.0)))
    sigma_min = float(np.sqrt(max(eigenvalues.min(), 0.0)))
    if sigma_min == 0.0:
        return float(np.inf)
    return sigma_max / sigma_min
