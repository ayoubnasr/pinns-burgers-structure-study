"""Phase 2 (not implemented): training loop.

Planned: Adam followed by L-BFGS, with the same parameter count and iteration budget for every variant
(baseline and structure-aware) and several seeds, so differences are not explained by budget or luck.
Gotcha: the comparison is only fair if collocation points and budgets are identical across variants.
"""


def train(cfg, seed):
    """Train one PINN for the given config and seed; return the trained model and loss history."""
    raise NotImplementedError("Phase 2")
