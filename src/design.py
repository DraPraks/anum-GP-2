"""Overdetermined SETAR design system A x = b."""

from dataclasses import dataclass

import numpy as np

from src.returns import simple_returns
from src.types import LAG_ORDER, N_PARAMS, THRESHOLD, Matrix, Vector


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
    """
    returns = simple_returns(close)
    # R_t, R_{t-1}, R_{t-2}. Drop first LAG_ORDER returns have no full lag pair.
    r_t = returns[LAG_ORDER:] # target day
    r_lag1 = returns[LAG_ORDER - 1 : -1] # target day - 1
    r_lag2 = returns[: -LAG_ORDER] # target day - 2
    
    row_dates = dates[LAG_ORDER + 1 :] # all dates except the first LAG_ORDER

    # array of booleans
    bullish = r_lag1 >= THRESHOLD 
    bearish= ~bullish 
    A = np.zeros((len(r_t), N_PARAMS), dtype=float)
    A[bullish, 0] = 1.0
    A[bullish, 1] = r_lag1[bullish]
    A[bullish, 2] = r_lag2[bullish]
    A[bearish, 3] = 1.0
    A[bearish, 4] = r_lag1[bearish]
    A[bearish, 5] = r_lag2[bearish]

    return DesignSystem(
        dates=row_dates,
        returns=returns,
        A=A,
        b=r_t,
        regime_bullish=bullish,
    )
