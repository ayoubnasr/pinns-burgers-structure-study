import numpy as np
import pytest

from burgers.data import split_snapshots
from burgers.pod import energy, modes_needed, pod_basis, projection_error, projection_errors
from burgers.reference import solve


@pytest.fixture(scope="module")
def snapshots():
    x = np.linspace(-1.0, 1.0, 257)
    t = np.linspace(0.0, 1.0, 41)
    return split_snapshots(solve(0.01 / np.pi, x, t).T)


def test_basis_is_orthonormal(snapshots):
    U, _ = pod_basis(snapshots[0])
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-12)


def test_training_error_matches_eckart_young(snapshots):
    X_train = snapshots[0]
    U, s = pod_basis(X_train)
    norm = np.linalg.norm(X_train)
    errs = projection_errors(U, X_train)
    for r in range(s.size + 1):
        expected = np.sqrt(np.sum(s[r:] ** 2)) / norm
        assert abs(projection_error(U, r, X_train) - expected) < 1e-12
        assert abs(errs[r] - expected) < 1e-12


def test_test_errors_and_modes_needed(snapshots):
    X_train, X_test = snapshots
    U, s = pod_basis(X_train)
    errs = projection_errors(U, X_test)
    assert np.all(np.diff(errs) <= 1e-14)  # adding modes never increases the error
    for r in (1, 5, 10):
        assert abs(errs[r] - projection_error(U, r, X_test)) < 1e-12
    r = modes_needed(U, X_test, 1e-2)
    assert errs[r] <= 1e-2 < errs[r - 1]
    assert modes_needed(U, X_test, 1e-300) is None
    assert np.isclose(energy(s)[-1], 1.0) and np.all(np.diff(energy(s)) >= 0)
