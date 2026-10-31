import copy
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from burgers.pinn.config2 import load_phase2_config  # noqa: E402
from burgers.pinn.evaluate import band_shares, metrics_from_prediction, near_shock_window, reference  # noqa: E402

CFG = copy.deepcopy(load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml"))
CFG["eval"].update(nx=1025)
NU = 0.001 / math.pi


def test_zero_error_when_prediction_equals_reference():
    x, u_ref, _, _, uc_ref = reference(CFG, NU)
    m, _ = metrics_from_prediction(CFG, NU, u_ref.copy(), uc_ref.copy())
    assert m["l2_global"] == m["l2_near"] == m["l2_away"] == m["l2_spacetime"] == 0.0
    assert math.isclose(m["slope_pred"], m["slope_ref"])


def test_near_shock_window_centered_and_wide_enough():
    x, u_ref, _, _, _ = reference(CFG, NU)
    steepest = x[np.argmax(np.abs(np.gradient(u_ref, x[1] - x[0])))]
    center, half, mask = near_shock_window(x, u_ref, NU, halfwidth_factor=1e-6, min_points=8)
    assert center == steepest and abs(center) < 0.01 and mask.sum() >= 8
    _, half_big, mask_big = near_shock_window(x, u_ref, NU, halfwidth_factor=10, min_points=8)
    assert half_big >= 10 * NU and mask_big.sum() >= mask.sum()  # widened on this coarse test grid


def test_band_shares_sum_to_one_and_locate_energy():
    amp = np.zeros(101)
    amp[90] = 1.0
    s = band_shares(amp, CFG["eval"]["bands"])
    assert math.isclose(sum(s.values()), 1.0) and s["high"] == 1.0
