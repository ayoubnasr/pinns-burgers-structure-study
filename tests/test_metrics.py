import numpy as np

from burgers.metrics import rel_l2


def test_identical_arrays_have_zero_error():
    x = np.linspace(-1, 1, 50)
    assert rel_l2(x, x) == 0.0


def test_known_value_and_scaling():
    b = np.array([3.0, 4.0])  # norm 5
    a = b + np.array([0.0, 0.5])
    assert np.isclose(rel_l2(a, b), 0.1)
    # scaling both arrays by the same factor leaves the relative error unchanged
    assert np.isclose(rel_l2(7.0 * a, 7.0 * b), rel_l2(a, b))
    # doubling the error doubles the relative error
    assert np.isclose(rel_l2(b + 2 * (a - b), b), 2 * rel_l2(a, b))


def test_zero_norm_guard():
    z = np.zeros(4)
    assert rel_l2(z, z) == 0.0
    assert rel_l2(np.ones(4), z) == float("inf")
