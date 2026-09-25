"""Overdetermined SETAR design system A x = b."""

from dataclasses import dataclass

from src.types import Matrix, Vector


@dataclass(frozen=True)
class DesignSystem:
    """One usable observation per row, t starting at lag order + 1 (t >= 3).

    Attributes:
        dates: observation dates aligned with each row (length m).
        returns: full simple-return series used to build the rows (length n-1).
        A: shape (m, 6). Bullish rows occupy columns 0:3; bearish rows occupy 3:6.
        b: shape (m,). Entry is the contemporaneous return R_t.
        regime_bullish: shape (m,) boolean, True when R_{t-1} >= 0.
    """

    dates: Vector
    returns: Vector
    A: Matrix
    b: Vector
    regime_bullish: Vector


def build_design_system(dates: Vector, close: Vector) -> DesignSystem:
    """Build A and b for the two-regime SETAR (p=2, c=0, d=1).

    Drop the first two returns: a row needs R_t, R_{t-1}, and R_{t-2}.
    The same builder is used for train and for test. Do not refit on test.

    Row rule:
        bullish (R_{t-1} >= 0): [1, R_{t-1}, R_{t-2}, 0, 0, 0]
        bearish (R_{t-1} < 0):  [0, 0, 0, 1, R_{t-1}, R_{t-2}]
    Target b_t = R_t.

    Raises:
        NotImplementedError: placeholder; the matrix is not assembled here.
    """
    del dates, close
    raise NotImplementedError("build_design_system")
