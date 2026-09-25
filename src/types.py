"""Shared numeric aliases.

Arrays are NumPy ndarrays. Callers may use array construction and matmul;
library factorizations and linear solvers stay out of this package.
"""

from typing import TypeAlias

import numpy as np

Array: TypeAlias = np.ndarray
Vector: TypeAlias = np.ndarray
Matrix: TypeAlias = np.ndarray

N_PARAMS = 6
"""Length of x = [alpha_1, phi_1_1, phi_1_2, alpha_2, phi_2_1, phi_2_2]."""

PARAM_NAMES: tuple[str, ...] = (
    "alpha_1",
    "phi_1_1",
    "phi_1_2",
    "alpha_2",
    "phi_2_1",
    "phi_2_2",
)

LAG_ORDER = 2
THRESHOLD = 0.0
DELAY = 1
"""Regime delay d = 1: the indicator uses R_{t-1} against c = 0."""
