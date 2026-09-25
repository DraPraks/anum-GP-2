"""SETAR least squares. Normal equations are implemented; Givens QR is a stub."""

from src.design import DesignSystem, build_design_system
from src.givens import GivensQrResult, givens_least_squares, verify_first_column_rotations
from src.lu import LuFactorization, factor_lu, solve_lu
from src.metrics import ComparisonRow, FitMetrics, rmse
from src.normal_equations import NormalEquationResult, solve_normal_equations
from src.outliers import ResidualOutlier, rank_residual_outliers
from src.plots import OverlayFigure, plot_returns_overlay
from src.returns import simple_returns
from src.validation import assert_design_system

__all__ = [
    "ComparisonRow",
    "DesignSystem",
    "FitMetrics",
    "GivensQrResult",
    "LuFactorization",
    "NormalEquationResult",
    "OverlayFigure",
    "ResidualOutlier",
    "assert_design_system",
    "build_design_system",
    "factor_lu",
    "givens_least_squares",
    "plot_returns_overlay",
    "rank_residual_outliers",
    "rmse",
    "simple_returns",
    "solve_lu",
    "solve_normal_equations",
    "verify_first_column_rotations",
]
