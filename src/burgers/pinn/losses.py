"""PINN loss terms for u_t + u u_x = nu u_xx, u(x, 0) = -sin(pi x), u(+-1, t) = 0.

The PDE residual r = u_t + u u_x - nu u_xx is computed by automatic differentiation of the network
output with respect to its inputs (create_graph=True so the loss can itself be differentiated).
The total loss is w_pde * mean(r^2) + w_ic * mean((u(x,0) + sin(pi x))^2) + w_bc * mean(u(+-1,t)^2).
Gotcha: a model is any callable model(x, t) -> u on (n, 1) tensors, so exact functions can be tested too.
"""

import math

import torch


def as_tensor(a, requires_grad=False):
    """Column tensor (n, 1) in the default dtype (float64 in Phase 2)."""
    return torch.as_tensor(a, dtype=torch.get_default_dtype()).reshape(-1, 1).requires_grad_(requires_grad)


def pde_residual(model, x, t, nu):
    """Return the residual u_t + u u_x - nu u_xx at the points (x, t); x and t must require grad."""
    u = model(x, t)
    ones = torch.ones_like(u)
    u_x, u_t = torch.autograd.grad(u, (x, t), ones, create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def initial_loss(model, x):
    """Mean squared misfit of u(x, 0) against -sin(pi x)."""
    u = model(x, torch.zeros_like(x))
    return torch.mean((u + torch.sin(math.pi * x)) ** 2)


def boundary_loss(model, x, t):
    """Mean squared value of u at the boundary points (x = +-1), which should be zero."""
    return torch.mean(model(x, t) ** 2)


def to_tensors(points):
    """Convert a sampling.point_set dict to tensors (interior points require grad for the residual)."""
    return {"f": (as_tensor(points["f"][0], True), as_tensor(points["f"][1], True)),
            "ic": as_tensor(points["ic"]),
            "bc": (as_tensor(points["bc"][0]), as_tensor(points["bc"][1]))}


def total_loss(model, pts, nu, weights):
    """Weighted sum of the three terms; returns (total, {'pde', 'ic', 'bc'} as floats).

    A weight of 0 skips that term entirely (used for the BC term of hard-constrained models).
    """
    parts = {"pde": torch.mean(pde_residual(model, *pts["f"], nu) ** 2), "ic": initial_loss(model, pts["ic"])}
    parts["bc"] = boundary_loss(model, *pts["bc"]) if weights["bc"] else torch.zeros(())
    total = sum(weights[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
