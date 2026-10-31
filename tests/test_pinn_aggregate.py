import json
import math
from pathlib import Path

import numpy as np

from burgers.pinn.aggregate import (bootstrap, comparison_label, evaluate_prediction, increases_as_nu_decreases,
                                    less_gain_than, load_runs, rho, smaller_gain_at_smooth, values, verdict)
from burgers.pinn.config2 import load_phase2_config

CFG = load_phase2_config(Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml")
BS = {"n": 2000, "seed": 1, "ci": 0.95}
RNG = np.random.default_rng(0)


def test_bootstrap_ci_brackets_known_ratio():
    base = RNG.lognormal(0.0, 0.2, 40)
    var = 0.5 * RNG.lognormal(0.0, 0.2, 40)
    point, lo, hi = rho(var, base, BS)
    assert lo < 0.5 < hi and lo < point < hi
    _, lo1, hi1 = bootstrap([base], lambda m: m[0], BS["n"], BS["seed"], BS["ci"])
    assert lo1 < 1.0 < hi1


def test_verdict_returns_each_label():
    assert verdict(0.3, 0.6, 0.7) == "validated"
    assert verdict(0.5, 0.9, 0.7) == "partial"
    assert verdict(0.8, 1.2, 0.7) == "not supported"
    assert verdict(1.1, 1.5, 0.7) == "harmful"


def test_comparison_predictions_on_constructed_cases():
    low, high = RNG.normal(1.0, 0.01, 5), RNG.normal(2.0, 0.01, 5)
    assert comparison_label(*increases_as_nu_decreases(low, high, BS)[1:]) == "supported"
    assert comparison_label(*increases_as_nu_decreases(high, low, BS)[1:]) == "contradicted"
    assert comparison_label(*less_gain_than(high, low, BS)[1:]) == "supported"          # V1b error larger
    same = RNG.normal(1.0, 0.1, 5)
    assert comparison_label(*less_gain_than(same, same.copy(), BS)[1:]) == "not supported"  # V1b matches V1a
    one = np.ones(5)
    # smooth case: rho = 0.9 (small gain); hard case: rho = 0.3 (large gain) -> gain smaller at smooth
    assert comparison_label(*smaller_gain_at_smooth(0.9 * one, one, 0.3 * one, one, BS)[1:]) == "supported"
    assert comparison_label(*smaller_gain_at_smooth(0.3 * one, one, 0.9 * one, one, BS)[1:]) == "contradicted"


def _run(variant, nu, seed, err, failed=False):
    return {"variant": variant, "nu": nu, "seed": seed, "failed": failed,
            "metrics": {} if failed else {"l2_near": err, "l2_global": err}}


def test_failed_runs_count_as_infinite_and_missing_data_is_not_run():
    nu = CFG["nu_list"][-1]
    runs = [_run("V0", nu, s, 1.0) for s in range(5)] + [_run("V4", nu, s, 0.2) for s in range(4)]
    runs.append(_run("V4", nu, 4, None, failed=True))
    assert np.isinf(values([r for r in runs if r["variant"] == "V4"], "l2_near")).sum() == 1
    h5 = evaluate_prediction(CFG, runs, "H5", CFG["predictions"]["H5"][0])
    # one failed run in five: ~6 % of resamples have an infinite median, so the CI upper bound is inf
    assert math.isclose(h5["value"], 0.2) and math.isinf(h5["hi"]) and h5["verdict"] == "not supported"
    h6 = evaluate_prediction(CFG, runs, "H6", CFG["predictions"]["H6"][0])
    assert h6["verdict"] == "not run"


def test_lr_search_runs_are_excluded(tmp_path):
    (tmp_path / "runs" / "V0" / "nu").mkdir(parents=True)
    (tmp_path / "lr_search" / "V0" / "nu").mkdir(parents=True)
    (tmp_path / "runs" / "V0" / "nu" / "seed0.json").write_text(json.dumps(_run("V0", 0.1, 0, 1.0)), encoding="utf-8")
    (tmp_path / "lr_search" / "V0" / "nu" / "lr0p001.json").write_text(json.dumps(_run("V0", 0.1, 0, 9.0)), encoding="utf-8")
    assert [r["metrics"]["l2_near"] for r in load_runs(tmp_path)] == [1.0]
