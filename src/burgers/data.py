"""Snapshot matrices: the reference solution stored as columns X[:, j] = u(x, t_j).

A snapshot matrix has one row per grid point and one column per time, which is the form POD needs.
The interleaved split puts even-index times in the training set and odd-index times in the test set,
so test snapshots lie between training snapshots (interpolation in time, not extrapolation).
Gotcha: files are named by nu (`data_nu<nu>.npz`) inside one run directory; they are not reused across runs.
"""

from pathlib import Path

import numpy as np

from burgers.reference import solve


def build_snapshots(cfg, nu):
    """Return X[nx, nt] with column j equal to u(x, t_j) for viscosity nu."""
    x, t = cfg.grid()
    return solve(nu, x, t, dz=cfg.ref_dz, log_cutoff=cfg.ref_log_cutoff).T


def split_snapshots(X, scheme="interleaved"):
    """Split snapshot columns into (X_train, X_test)."""
    if scheme != "interleaved":
        raise ValueError(f"unknown split scheme: {scheme}")
    return X[:, 0::2], X[:, 1::2]


def data_path(run_dir, nu):
    return Path(run_dir) / f"data_nu{nu:.6g}.npz"


def save_snapshots(run_dir, nu, x, t, X):
    path = data_path(run_dir, nu)
    np.savez_compressed(path, nu=nu, x=x, t=t, X=X)
    return path


def load_snapshots(run_dir, nu):
    """Return (x, t, X) saved by `save_snapshots`."""
    with np.load(data_path(run_dir, nu)) as f:
        return f["x"], f["t"], f["X"]
