"""Invariants every design matrix must satisfy."""

import numpy as np

from src.design import DesignSystem
from src.types import N_PARAMS


def assert_design_system(system: DesignSystem) -> None:
    """Assert the structural checks required before any solve.

    Required checks:
        - A has 6 columns
        - each row has exactly three nonzeros
        - each row lies entirely in the bull block or the bear block
        - len(b) == A.shape[0]
        - A and b contain no NaN or Inf

    The official six-price fixture must pass this and match the printed A, b
    within the rounding shown in the brief.

    Raises:
        AssertionError: a required structural check failed.
    """
    bull_block = system.A[:, :3]
    bear_block = system.A[:, 3:]
    bull_nonzeros = np.count_nonzero(bull_block, axis=1)
    bear_nonzeros = np.count_nonzero(bear_block, axis=1)
    # in_one_block = (all zeros in 0:2 bull block OR all zeros in 3:6 bear block)
    in_one_block = ((bull_nonzeros == 3) & (bear_nonzeros == 0)) | ((bull_nonzeros == 0) & (bear_nonzeros == 3))

    assert system.A.shape[1] == N_PARAMS, "A must have 6 columns"
    assert len(system.b) == system.A.shape[0], "len(b) must equal the number of rows of A"
    assert np.all(np.count_nonzero(system.A, axis=1) == 3), "each row must have exactly three nonzeros"
    assert np.all(in_one_block), "each row must lie entirely in the bull block or the bear block"
    assert np.all(system.regime_bullish == (bull_nonzeros == 3)), "regime_bullish must match the active block"
    assert np.all(np.isfinite(system.A)), "A must contain no NaN or Inf"
    assert np.all(np.isfinite(system.b)), "b must contain no NaN or Inf"
