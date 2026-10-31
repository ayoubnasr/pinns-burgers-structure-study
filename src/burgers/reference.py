"""Reference solution of u_t + u u_x = nu u_xx, u(x,0) = -sin(pi x), u(+-1,t) = 0, via Hopf-Cole.

Hopf-Cole: u = -2 nu theta_x / theta turns Burgers into the heat equation for theta, so
u(x,t) = -int sin(pi(x-e)) F(x-e) G(e) de / int F(x-e) G(e) de with F(y) = exp(-cos(pi y)/(2 pi nu)),
G(e) = exp(-e^2/(4 nu t)). We substitute e = sqrt(4 nu t) z and use the trapezoid rule in z.
Gotcha: for small nu the exponent is ~1/nu (overflow), so we subtract its maximum (log-sum-exp) per x.
`solve_fd` is an independent finite-difference solver used only to cross-check at large nu.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Rows of x processed at once; only limits memory (nx_chunk * nz floats per temporary array).
X_CHUNK = 512


def solve(nu, x, t, dz=0.15, log_cutoff=50.0):
    """Exact (quadrature) solution u[nt, nx] on the grids x, t.

    dz: step in the scaled variable z. The integrand is smooth (entire) and grows at most like
        exp((pi t + 1) y^2) off the real axis, so the trapezoid error is ~exp(-pi^2 / (dz^2 (pi t + 1)));
        for t <= 1 and dz = 0.15 that is ~1e-46, far below round-off, for every nu.
    log_cutoff: z is truncated where the integrand is below exp(-log_cutoff) times its peak;
        the cosine term can lift the integrand by at most exp(1/(pi nu)), hence z_max below.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    z_max = np.sqrt(1.0 / (np.pi * nu) + log_cutoff)
    z = np.arange(-z_max, z_max + dz, dz)
    u = np.empty((t.size, x.size))
    for i, ti in enumerate(t):
        if ti == 0.0:
            u[i] = -np.sin(np.pi * x)
            continue
        a = np.sqrt(4.0 * nu * ti)
        for j in range(0, x.size, X_CHUNK):
            arg = np.pi * (x[j:j + X_CHUNK, None] - a * z[None, :])
            log_w = -np.cos(arg) / (2.0 * np.pi * nu) - z[None, :] ** 2
            w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
            u[i, j:j + X_CHUNK] = -(np.sin(arg) * w).sum(axis=1) / w.sum(axis=1)
    return u


def solve_fd(nu, x, t, rtol=1e-9, atol=1e-11):
    """Fallback: second-order finite differences in x (conservative flux form) + adaptive RK45 in t.

    Only meant for smooth cases (large nu) as an independent check of `solve`; returns u[nt, nx].
    """
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]

    def rhs(_, v):
        w = np.concatenate(([0.0], v, [0.0]))  # Dirichlet u = 0 at both ends
        flux = 0.5 * w ** 2
        return -(flux[2:] - flux[:-2]) / (2 * dx) + nu * (w[2:] - 2 * w[1:-1] + w[:-2]) / dx ** 2

    sol = solve_ivp(rhs, (t[0], t[-1]), -np.sin(np.pi * x[1:-1]), t_eval=t, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"finite-difference fallback failed: {sol.message}")
    u = np.zeros((t.size, x.size))
    u[:, 1:-1] = sol.y.T
    return u


def points_across_shock(u_final, x):
    """Grid points across the steepest front: (max u - min u) / max|u_x| / dx.

    For the viscous shock profile u = -U tanh(U x / (2 nu)) this width is 4 nu / U.
    """
    dx = x[1] - x[0]
    slope = np.max(np.abs(np.gradient(u_final, dx)))
    width = (u_final.max() - u_final.min()) / slope
    return float(width / dx)
