"""Residuals, RMSE, and the comparison row both solvers must fill."""

from dataclasses import dataclass

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

    Raises:
        NotImplementedError: placeholder.
    """
    del system, x
    raise NotImplementedError("fitted_returns")


def rmse(actual: Vector, predicted: Vector) -> float:
    """Root mean square error of returns on one split.

    Raises:
        NotImplementedError: placeholder.
    """
    del actual, predicted
    raise NotImplementedError("rmse")


def condition_number_2(matrix: Matrix) -> float:
    """2-norm condition number. Derivation may use SVD implemented in-repo later.

    Library ``cond`` / ``svd`` helpers are out of scope for the solver path.
    This hook exists so both methods can record kappa_2(A) and kappa_2(A^T A).

    Raises:
        NotImplementedError: placeholder.
    """
    del matrix
    raise NotImplementedError("condition_number_2")
