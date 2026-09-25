"""Invariants every design matrix must satisfy."""

from src.design import DesignSystem


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
        NotImplementedError: placeholder; checks are not executed here.
    """
    del system
    raise NotImplementedError("assert_design_system")
