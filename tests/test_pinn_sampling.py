from pathlib import Path

import numpy as np

from burgers.pinn.config2 import load_phase2_config
from burgers.pinn.sampling import point_set

CFG = load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml")


def test_deterministic_for_a_seed_and_different_across_sets():
    a, b = point_set(CFG, 3), point_set(CFG, 3)
    assert np.array_equal(a["f"][0], b["f"][0]) and np.array_equal(a["ic"], b["ic"])
    assert not np.array_equal(point_set(CFG, 4)["f"][0], a["f"][0])
    val = point_set(CFG, 3, held_out=True)
    assert val["f"][0].size == CFG["train"]["n_val"] and not np.array_equal(val["ic"], a["ic"])


def test_points_respect_the_domain():
    p = point_set(CFG, 0)
    x, t = p["f"]
    assert x.min() >= -1 and x.max() <= 1 and t.min() >= 0 and t.max() <= CFG["T"]
    assert np.all(np.abs(p["bc"][0]) == 1) and np.sum(p["bc"][0] < 0) == CFG["train"]["n_bc"] // 2


def test_near_shock_fraction_is_honored():
    window = (-0.01, 0.01)
    x, _ = point_set(CFG, 0, near_fraction=0.5, window=window)["f"]
    n = CFG["train"]["n_f"]
    inside = np.sum((x >= window[0]) & (x <= window[1]))
    # half are drawn in the window, plus the uniform points that fall there by chance (~1 % of the rest)
    assert n // 2 <= inside <= n // 2 + 0.03 * n
