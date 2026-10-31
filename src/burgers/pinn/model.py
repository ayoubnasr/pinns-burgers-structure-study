"""Phase 2 (not implemented): the network u_theta(x, t) used by the PINN.

Planned: a tanh multilayer perceptron, optionally preceded by a Fourier-feature embedding
[sin(2 pi B z), cos(2 pi B z)] of the input z = (x, t), which lets the network represent high
frequencies (such as a shock) more easily. Gotcha: torch is imported inside functions only, so Phase 1
runs without it.
"""


def build_mlp(width, depth, fourier_features=0, fourier_scale=1.0, seed=0):
    """Return a torch module mapping (x, t) of shape [n, 2] to u of shape [n, 1]."""
    raise NotImplementedError("Phase 2")
