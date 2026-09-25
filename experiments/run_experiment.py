"""Single entry point that will write the comparison table and both figures.

Pipeline (each step is a placeholder and raises NotImplementedError):

1. Load Close from data/stock_train.csv and data/stock_test.csv.
2. Build A, b for train and for test with the same design builder.
3. Check the official six-price fixture against the printed A and b.
4. Solve the training system with normal equations (dense LU) and with Givens QR.
5. Freeze each x from train. Score RMSE on train and on test. Do not refit on test.
6. Rank the worst training residuals.
7. Print the comparison table.
8. Write figures/train_overlay and figures/train_test_overlay.
"""

from pathlib import Path

from src.design import build_design_system
from src.givens import givens_least_squares, verify_first_column_rotations
from src.metrics import fitted_returns, rmse
from src.normal_equations import solve_normal_equations
from src.outliers import rank_residual_outliers
from src.plots import plot_returns_overlay
from src.validation import assert_design_system

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TABLE = ROOT / "artifacts" / "comparison.txt"


def load_close(path: Path):
    """Read Date and Close. CSV parsing is deferred with the rest of the pipeline."""
    raise NotImplementedError(f"load_close({path.name})")


def print_comparison(rows: list) -> None:
    """Print and store the normal-equations versus Givens QR table."""
    raise NotImplementedError("print_comparison")


def main() -> None:
    train = build_design_system(*load_close(DATA / "stock_train.csv"))
    test = build_design_system(*load_close(DATA / "stock_test.csv"))
    assert_design_system(train)
    assert_design_system(test)

    normal = solve_normal_equations(train.A, train.b)
    verify_first_column_rotations(train.A)
    qr = givens_least_squares(train.A, train.b)

    for name, x in (("normal", normal.x), ("givens", qr.x)):
        rmse(train.b, fitted_returns(train, x))
        rmse(test.b, fitted_returns(test, x))
        del name

    rank_residual_outliers(train.b - train.A @ normal.x, train.dates)
    rank_residual_outliers(train.b - train.A @ qr.x, train.dates)

    print_comparison([])
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_returns_overlay(FIGURES / "train_overlay.png", train.b, train.b)
    plot_returns_overlay(
        FIGURES / "train_test_overlay.png",
        train.b,
        train.b,
        split_index=train.A.shape[0],
    )


if __name__ == "__main__":
    main()
