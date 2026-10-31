"""Phase 2 (not implemented): the PINN loss terms.

Planned: mean squared PDE residual u_t + u u_x - nu u_xx at collocation points (derivatives by
automatic differentiation), plus initial-condition misfit u(x, 0) + sin(pi x) and boundary misfit
u(+-1, t). Gotcha: the relative weights of the three terms are a modelling choice and must be configured.
"""


def pde_residual(model, x, t, nu):
    """Return the residual u_t + u u_x - nu u_xx at the points (x, t)."""
    raise NotImplementedError("Phase 2")


def initial_loss(model, x):
    """Mean squared misfit of u(x, 0) against -sin(pi x)."""
    raise NotImplementedError("Phase 2")


def boundary_loss(model, t):
    """Mean squared value of u(-1, t) and u(1, t), which should be zero."""
    raise NotImplementedError("Phase 2")
