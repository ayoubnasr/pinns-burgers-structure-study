"""Proper orthogonal decomposition (POD): a linear basis for the snapshots, from the SVD X = U S V^T.

Eckart-Young: keeping the first r columns of U gives the best rank-r approximation of X, with
Frobenius error sqrt(sum_{i>=r} s_i^2), so fast singular-value decay means few modes suffice.
A moving sharp front is hard for any fixed linear basis (Kolmogorov n-width): each time shifts the
front, and a narrow front at many positions spans many nearly independent directions.
Gotcha: error on *test* snapshots is not given by s; we project them and measure it directly.
"""

import numpy as np


def pod_basis(X_train):
    """Return the POD modes U[nx, m] and singular values s[m] (descending) of the training snapshots."""
    U, s, _ = np.linalg.svd(X_train, full_matrices=False)
    return U, s


def projection_errors(U, X):
    """Relative Frobenius error ||X - U_r U_r^T X|| / ||X|| for every r = 0..m, as an array of length m+1.

    Uses the orthogonal split X = U C + E (E orthogonal to all modes), so the squared error for rank r is
    sum_{i>=r} ||C_i||^2 + ||E||^2, summed from the small end to avoid cancellation.
    """
    C = U.T @ X
    tail = np.linalg.norm(X - U @ C) ** 2
    row_sq = np.sum(C ** 2, axis=1)
    err_sq = np.concatenate((np.cumsum(row_sq[::-1])[::-1], [0.0])) + tail
    return np.sqrt(err_sq) / np.linalg.norm(X)


def projection_error(U, r, X):
    """Relative Frobenius error of projecting the columns of X onto the first r modes."""
    Ur = U[:, :r]
    return float(np.linalg.norm(X - Ur @ (Ur.T @ X)) / np.linalg.norm(X))


def modes_needed(U, X_test, tol):
    """Smallest r >= 1 whose test projection error is <= tol, or None if no available r reaches it."""
    errs = projection_errors(U, X_test)
    hits = np.nonzero(errs[1:] <= tol)[0]
    return int(hits[0] + 1) if hits.size else None


def energy(s):
    """Fraction of total energy sum(s^2) captured by the first r modes, for r = 1..m."""
    e = np.cumsum(s ** 2)
    return e / e[-1]
