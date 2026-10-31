import copy
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from burgers.pinn.config2 import load_phase2_config  # noqa: E402
from burgers.pinn.train import continuation_schedule, train_run  # noqa: E402
from burgers.pinn.variants import build_spec  # noqa: E402

CFG = load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml")


def tiny(steps=50):
    cfg = copy.deepcopy(CFG)
    cfg["net"].update(width=16, depth=2)
    cfg["train"].update(adam_steps=steps, n_f=200, n_ic=50, n_bc=50, n_val=100, log_every=10)
    return cfg


def test_fifty_steps_lower_the_loss():
    cfg = tiny()
    out = train_run(cfg, build_spec(cfg, "V0"), 0.1 / math.pi, seed=0, lr=1e-3)
    assert not out["failed"] and out["steps"] == 50
    assert out["curve"][-1][1] < out["curve"][0][1]
    assert math.isfinite(out["val_loss"])


def test_divergence_marks_run_failed():
    cfg = tiny()
    cfg["train"]["divergence_bound"] = 1e-12
    out = train_run(cfg, build_spec(cfg, "V0"), 0.1 / math.pi, seed=0, lr=1e-3)
    assert out["failed"] and out["val_loss"] == float("inf") and out["steps"] == 0


def test_continuation_keeps_budget_and_ends_at_target():
    nu0, nu = 0.1 / math.pi, 0.001 / math.pi
    sched = continuation_schedule(nu, 1000, 3, nu0)
    assert sum(n for _, n in sched) == 1000 and math.isclose(sched[-1][0], nu) and math.isclose(sched[0][0], nu0)
    assert continuation_schedule(nu0, 1000, 3, nu0) == [(nu0, 1000)]


def test_continuation_variant_trains_same_steps():
    cfg = tiny(30)
    out = train_run(cfg, build_spec(cfg, "V6"), 0.001 / math.pi, seed=1, lr=1e-3, window=(-0.01, 0.01))
    assert out["steps"] == 30 and len(out["schedule"]) == 3
