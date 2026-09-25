"""Official six-price appendix. Values are the printed illustration, not a solver.

Returns and matrix entries are stored at the rounding shown in the brief so a
later check can compare a builder against this fixture.
"""

import numpy as np

CLOSE = np.array([10000.0, 10100.0, 9900.0, 10050.0, 10000.0, 10200.0])

# Printed returns R_1 .. R_5.
RETURNS_PRINTED = np.array([0.010, -0.0198, 0.0152, -0.0050, 0.0200])

# Three design rows, t = 3, 4, 5. Row 1 bearish, row 2 bullish, row 3 bearish.
A_PRINTED = np.array(
    [
        [0.0, 0.0, 0.0, 1.0, -0.0198, 0.0100],
        [1.0, 0.0152, -0.0198, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, -0.0050, 0.0152],
    ]
)

B_PRINTED = np.array([0.0152, -0.0050, 0.0200])
