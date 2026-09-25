"""Single entry point that writes the comparison table and both figures.

Pipeline:

1. Load Close from data/stock_train.csv and data/stock_test.csv.
2. Build A, b for train and for test with the same design builder.
3. Check the official six-price fixture against the printed A and b.
4. Solve the training system with normal equations (dense LU).
   TODO: solve the same system with Givens QR.
5. Freeze each x from train. Score RMSE on train and on test. Do not refit on test.
6. Rank the worst training residuals.
7. Print the comparison table.
8. Write figures/train_overlay and figures/train_test_overlay.
"""

from pathlib import Path

import numpy as np

from data.fixture import A_PRINTED, B_PRINTED, CLOSE
from src.design import build_design_system
from src.metrics import ComparisonRow, fitted_returns, rmse
from src.normal_equations import solve_normal_equations
from src.outliers import ResidualOutlier, rank_residual_outliers
from src.plots import plot_returns_overlay
from src.validation import assert_design_system

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TABLE = ROOT / "artifacts" / "comparison.txt"


def load_close(path: Path):
    """Read Date and Close."""
    dates = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str)
    close = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1, dtype=float)
    return dates, close


def check_printed_fixture() -> None:
    """Build the six-price system and match the printed A and b at four decimals."""
    labels = np.array([f"t{index}" for index in range(len(CLOSE))])
    system = build_design_system(labels, CLOSE)
    assert_design_system(system)
    if not np.allclose(np.round(system.A, 4), A_PRINTED):
        raise AssertionError("fixture design matrix does not match A_PRINTED")
    if not np.allclose(np.round(system.b, 4), B_PRINTED):
        raise AssertionError("fixture right-hand side does not match B_PRINTED")


def _format_optional(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_comparison(rows: list[ComparisonRow]) -> None:
    """Print and store the normal-equations versus Givens QR table."""
    header = (
        f"{'method':<20} {'flops':>12} {'memory':>12} {'Q memory':>12} "
        f"{'kappa(A)':>12} {'kappa(ATA)':>12} {'||r||_2':>12} "
        f"{'RMSE train':>12} {'RMSE test':>12} {'seconds':>12}"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.method:<20} {row.flops:12d} {row.memory_bytes:12d} "
            f"{_format_optional(row.memory_q_explicit_bytes):>12} "
            f"{row.kappa_A:12.6g} {_format_optional(row.kappa_ata):>12} "
            f"{row.residual_norm_train:12.6g} {row.rmse_train:12.6g} "
            f"{row.rmse_test:12.6g} {row.runtime_seconds:12.6g}"
        )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(text, encoding="utf-8")


def _format_outliers(title: str, ranked: tuple[ResidualOutlier, ...]) -> str:
    lines = [title]
    for item in ranked:
        lines.append(
            f"  row {item.row:4d}  date {item.date}  residual {item.residual:.6g}"
        )
    return "\n".join(lines)


def main() -> None:
    check_printed_fixture()
    train = build_design_system(*load_close(DATA / "stock_train.csv"))
    test = build_design_system(*load_close(DATA / "stock_test.csv"))
    assert_design_system(train)
    assert_design_system(test)

    normal = solve_normal_equations(train.A, train.b)
    # TODO: verify_first_column_rotations(train.A) and givens_least_squares(train.A, train.b).

    normal_train = fitted_returns(train, normal.x)
    normal_test = fitted_returns(test, normal.x)

    rows = [
        ComparisonRow(
            method="normal equations",
            flops=normal.flops,
            memory_bytes=normal.memory_bytes,
            memory_q_explicit_bytes=None,
            kappa_A=normal.kappa_A,
            kappa_ata=normal.kappa_G,
            residual_norm_train=normal.residual_norm,
            rmse_train=rmse(train.b, normal_train),
            rmse_test=rmse(test.b, normal_test),
            runtime_seconds=normal.runtime_seconds,
        ),
    ]
    print_comparison(rows)

    normal_outliers = rank_residual_outliers(train.b - train.A @ normal.x, train.dates)
    outlier_text = "\n".join(
        (
            _format_outliers("Largest |train residual|, normal equations", normal_outliers),
            "",
        )
    )
    print(outlier_text, end="")
    with TABLE.open("a", encoding="utf-8") as handle:
        handle.write("\n" + outlier_text)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_returns_overlay(FIGURES / "train_overlay.png", train.b, normal_train)
    plot_returns_overlay(
        FIGURES / "train_test_overlay.png",
        np.concatenate((train.b, test.b)),
        np.concatenate((normal_train, normal_test)),
        split_index=train.A.shape[0],
    )


if __name__ == "__main__":
    main()
