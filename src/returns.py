"""Close prices to simple returns."""

from src.types import Vector


def simple_returns(close: Vector) -> Vector:
    """Return R_t = (P_t - P_{t-1}) / P_{t-1} for t = 1 .. n-1.

    ``close`` is the Close Price series in time order, length n >= 2.
    The result has length n - 1 and aligns with dates[1:].

    Raises:
        ValueError: if close is too short.
    """
    if len(close) < 2:
        raise ValueError("close must have at least 2 elements")

    return (close[1:] - close[:-1]) / close[:-1]
