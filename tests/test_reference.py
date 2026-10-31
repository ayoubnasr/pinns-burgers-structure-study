import numpy as np
import pytest

from burgers.metrics import rel_l2
from burgers.reference import points_across_shock, solve, solve_fd

NU_LIST = [0.1 / np.pi, 0.01 / np.pi, 0.001 / np.pi]
X = np.linspace(-1.0, 1.0, 257)  # symmetric grid, so X[::-1] == -X
T = np.linspace(0.0, 1.0, 11)


@pytest.fixture(scope="module", params=NU_LIST)
def case(request):
    nu = request.param
    return nu, solve(nu, X, T)


def test_boundary_values_are_zero(case):
    _, u = case
    assert np.max(np.abs(u[:, [0, -1]])) < 1e-10


def test_odd_symmetry(case):
    _, u = case
    assert np.max(np.abs(u + u[:, ::-1])) < 1e-10


def test_max_abs_u_non_increasing(case):
    _, u = case
    m = np.abs(u).max(axis=1)
    assert np.all(np.diff(m) <= 1e-12)


@pytest.mark.parametrize("nu", NU_LIST)
@pytest.mark.parametrize("x0, t0", [(0.5, 0.2), (-0.6, 0.5), (0.7, 0.9)])
def test_pde_residual_small_in_smooth_region(nu, x0, t0):
    # central differences on a 3x3 stencil around (x0, t0), away from the shock at x = 0
    h = 1e-3
    u = solve(nu, x0 + h * np.array([-1.0, 0.0, 1.0]), t0 + h * np.array([-1.0, 0.0, 1.0]))
    u_t = (u[2, 1] - u[0, 1]) / (2 * h)
    u_x = (u[1, 2] - u[1, 0]) / (2 * h)
    u_xx = (u[1, 2] - 2 * u[1, 1] + u[1, 0]) / h ** 2
    c = u[1, 1]
    residual = u_t + c * u_x - nu * u_xx
    scale = abs(u_t) + abs(c * u_x) + abs(nu * u_xx)
    assert abs(residual) < 1e-4 * scale


def test_matches_finite_difference_fallback_at_largest_nu():
    # stated tolerance: 1e-3 relative L2 (the fallback is second order with dx = 2/512)
    nu = NU_LIST[0]
    x = np.linspace(-1.0, 1.0, 513)
    assert rel_l2(solve_fd(nu, x, T), solve(nu, x, T)) < 1e-3


def test_points_across_shock_on_tanh_profile():
    nu, amp = 1e-3, 0.5
    x = np.linspace(-1.0, 1.0, 4001)
    u = -amp * np.tanh(amp * x / (2 * nu))
    expected = 4 * nu / amp / (x[1] - x[0])
    assert abs(points_across_shock(u, x) - expected) < 0.02 * expected
