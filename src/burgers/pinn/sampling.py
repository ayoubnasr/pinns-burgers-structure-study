"""Training and validation points for the PINN losses.

Interior (collocation) points are uniform in [-1, 1] x [0, T]; initial points are uniform in x at t = 0;
boundary points are uniform in t, half at x = -1 and half at x = +1. Optionally a fraction of the interior
points is drawn uniformly inside a window around the shock (variant V6).
Gotcha: every set is drawn from its own stream SeedSequence([seed, stream]), so the held-out validation
set never overlaps the training set's random numbers, and changing one set does not shift the others.
"""

import numpy as np

TRAIN_STREAM, VAL_STREAM = 0, 1


def interior(n, rng, T, near_fraction=0.0, window=None):
    """Return x, t arrays of n interior points; `near_fraction` of them lie in the x-window (lo, hi)."""
    n_near = int(round(near_fraction * n)) if window is not None else 0
    x = np.concatenate((rng.uniform(-1.0, 1.0, n - n_near), rng.uniform(window[0], window[1], n_near) if n_near else []))
    t = rng.uniform(0.0, T, n)
    return x, t


def initial(n, rng):
    return rng.uniform(-1.0, 1.0, n)


def boundary(n, rng, T):
    """Return x (first half -1, rest +1) and t (uniform) for n boundary points."""
    x = np.where(np.arange(n) < n // 2, -1.0, 1.0)
    return x, rng.uniform(0.0, T, n)


def point_set(cfg, seed, held_out=False, near_fraction=0.0, window=None):
    """All points for one run as numpy arrays: {'f': (x, t), 'ic': x, 'bc': (x, t)}."""
    tr = cfg["train"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, VAL_STREAM if held_out else TRAIN_STREAM]))
    n_f = tr["n_val"] if held_out else tr["n_f"]
    return {"f": interior(n_f, rng, cfg["T"], near_fraction, window),
            "ic": initial(tr["n_ic"], rng),
            "bc": boundary(tr["n_bc"], rng, cfg["T"])}
