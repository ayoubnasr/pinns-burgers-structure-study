import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from burgers.pinn.config2 import load_phase2_config  # noqa: E402
from burgers.pinn.models import build_model, param_count, pod_latent_model  # noqa: E402
from burgers.pinn.variants import build_spec, expand  # noqa: E402

torch.set_default_dtype(torch.float64)
CFG = load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml")
NU = 0.001 / math.pi
NAMES = expand(CFG, ["V0", "V1a", "V1b", "V2", "V3", "V4", "V5", "V6", "V7", "relu"])


def test_hard_constraint_is_exactly_odd_and_zero_at_boundary():
    model = build_model(build_spec(CFG, "V3"), NU, seed=0)
    x = torch.linspace(-1, 1, 101).reshape(-1, 1)
    t = torch.full_like(x, 0.4)
    with torch.no_grad():
        u, u_flip = model(x, t), model(-x, t)
    assert float((u + u_flip).abs().max()) < 1e-15
    assert float(u[[0, -1]].abs().max()) < 1e-15


@pytest.mark.parametrize("name", NAMES)
def test_param_count_formula_matches_model(name):
    spec = build_spec(CFG, name)
    model = build_model(spec, NU, seed=0)
    assert param_count(spec) == sum(p.numel() for p in model.parameters())


def test_feature_and_init_components():
    m = build_model(build_spec(CFG, "V1a"), NU, seed=0)
    assert math.isclose(float(m.modes[0]), 1.0) and math.isclose(float(m.modes[-1]), 1.0 / NU, rel_tol=1e-9)
    steep = build_model(build_spec(CFG, "V5"), NU, seed=0).net[0].weight[:, 0].detach()
    n = round(0.25 * CFG["net"]["width"])
    assert float(steep[:n].abs().mean()) > 100 * float(steep[n:].abs().mean())
    front = build_model(build_spec(CFG, "V4"), NU, seed=0).front
    assert math.isclose(float(front["b"]), 1.0 / NU) and math.isclose(float(front["a"]), 0.1)


def test_same_seed_same_init():
    a, b = build_model(build_spec(CFG, "V0"), NU, 7), build_model(build_spec(CFG, "V0"), NU, 7)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_pod_latent_is_a_stub():
    with pytest.raises(NotImplementedError, match="underspecified"):
        pod_latent_model()
