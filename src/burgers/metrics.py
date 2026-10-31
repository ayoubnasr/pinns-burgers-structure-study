"""Error metrics.

rel_l2(a, b) = ||a - b||_2 / ||b||_2: the size of the error relative to the size of the reference b,
so 0.01 means "1 % error" whatever the units or grid size (arrays are flattened first).
Gotcha: if ||b|| = 0 the ratio is undefined; we return 0.0 when a == b too, and inf otherwise.
"""

import numpy as np


def rel_l2(a, b):
    """Relative L2 (Frobenius for matrices) error of `a` against reference `b`."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    err = np.linalg.norm(a - b)
    ref = np.linalg.norm(b)
    if ref == 0.0:
        return 0.0 if err == 0.0 else float("inf")
    return float(err / ref)
