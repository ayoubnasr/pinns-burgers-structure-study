"""The PINN network u(x, t): an MLP plus optional input features, hard constraints and a front term.

Inputs are (x, t), optionally x / nu (stretched coordinate) and Fourier features sin(pi k x), cos(pi k x) for
m log-spaced modes k in [1, c / nu]. With hard constraints the output is (1 - x^2) (N(x,t) - N(-x,t)) / 2,
which is exactly odd in x and exactly 0 at x = +-1. The front term adds a * tanh(b x) with learnable a, b.
Gotcha: the front term is added to N before the hard-constraint transform, so combining both keeps exactness.
"""

import math

import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def fourier_modes(m, c, nu):
    """m log-spaced mode numbers from 1 to c / nu (mode k means sin(pi k x), as in Phase 1)."""
    return torch.logspace(0.0, math.log10(max(c / nu, 1.0)), m)


def n_inputs(spec):
    return 2 + int(spec["stretched"]) + (2 * spec["fourier"]["m"] if spec["fourier"] else 0)


def param_count(spec):
    """Trainable parameters of build_model(spec, nu), computed from the layer sizes."""
    w, d = spec["width"], spec["depth"]
    return n_inputs(spec) * w + w + (d - 1) * (w * w + w) + w + 1 + (2 if spec["front"] else 0)


class PINN(nn.Module):
    """MLP on the input features, with the optional front term and hard-constraint transform."""

    def __init__(self, spec, nu):
        super().__init__()
        self.nu, self.hard, self.stretched = nu, spec["hard"], spec["stretched"]
        modes = fourier_modes(spec["fourier"]["m"], spec["fourier"]["c"], nu) if spec["fourier"] else torch.zeros(0)
        self.register_buffer("modes", modes)
        act = ACTIVATIONS[spec["activation"]]
        sizes = [n_inputs(spec)] + [spec["width"]] * spec["depth"]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), act()]
        layers.append(nn.Linear(sizes[-1], 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):  # Glorot (Xavier) normal weights, zero biases
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        if spec["steep"]:  # V5: some first-layer units see x at scale c / nu
            first = self.net[0].weight
            n_steep = int(round(spec["steep"]["steep_fraction"] * spec["width"]))
            with torch.no_grad():
                first[:n_steep, 0] = torch.randn(n_steep) * spec["steep"]["c"] / nu
        self.front = None
        if spec["front"]:
            self.front = nn.ParameterDict({
                "a": nn.Parameter(torch.tensor(float(spec["front"]["a_init"]))),
                "b": nn.Parameter(torch.tensor(spec["front"]["front_b_init_factor"] / nu))})

    def features(self, x, t):
        cols = [x, t]
        if self.stretched:
            cols.append(x / self.nu)
        if self.modes.numel():
            cols += [torch.sin(math.pi * x * self.modes), torch.cos(math.pi * x * self.modes)]
        return torch.cat(cols, dim=1)

    def raw(self, x, t):
        out = self.net(self.features(x, t))
        if self.front is not None:
            out = out + self.front["a"] * torch.tanh(self.front["b"] * x)
        return out

    def forward(self, x, t):
        if self.hard:
            return (1 - x ** 2) * (self.raw(x, t) - self.raw(-x, t)) / 2
        return self.raw(x, t)


def build_model(spec, nu, seed):
    """Build the network for a variant spec with a fixed initialization seed."""
    torch.manual_seed(seed)
    return PINN(spec, nu)


def pod_latent_model(*args, **kwargs):
    """Tier C POD-latent operator model over nu: architecture and training data are not specified."""
    raise NotImplementedError("underspecified in docs/PHASE2.md")
