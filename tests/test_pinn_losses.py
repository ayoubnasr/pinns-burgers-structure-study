import math

import pytest

torch = pytest.importorskip("torch")

from burgers.pinn.losses import as_tensor, boundary_loss, initial_loss, pde_residual  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_residual_of_exact_steady_shock_is_zero():
    a, nu = 0.7, 0.01
    x = as_tensor(torch.linspace(-1, 1, 201), True)
    t = as_tensor(torch.rand(201), True)
    r = pde_residual(lambda x, t: -a * torch.tanh(a * x / (2 * nu)) + 0 * t, x, t, nu)
    assert float(r.abs().max()) < 1e-10


def test_ic_and_bc_losses_vanish_for_exact_data():
    x = as_tensor(torch.linspace(-1, 1, 51))
    assert float(initial_loss(lambda x, t: -torch.sin(math.pi * x) * torch.exp(-t), x)) < 1e-28
    xb, tb = as_tensor([-1.0, 1.0]), as_tensor([0.3, 0.7])
    assert float(boundary_loss(lambda x, t: (1 - x ** 2) * t, xb, tb)) == 0.0
