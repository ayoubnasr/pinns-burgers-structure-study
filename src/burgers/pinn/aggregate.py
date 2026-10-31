"""Turn finished runs into medians, bootstrap confidence intervals and one verdict per prediction.

Effect size: rho = median error of a variant / median error of V0 at the same nu. Its 95 % CI comes from a
bootstrap that resamples seeds independently for the variant and for V0. Verdicts follow docs/PHASE2.md:
validated (CI upper bound < threshold), partial (CI < 1 only), not supported (CI includes 1), harmful (CI > 1).
Gotcha: failed runs count as infinite error, never dropped; lr-search runs are not read for statistics.
"""

import json
import math
from pathlib import Path

import numpy as np

from burgers.pinn.config2 import check_against_doc

DESIGN = {"V0": "Baseline", "V1a": "Fourier features of $x$ up to $\\sim 1/\\nu$",
          "V1b": "Fourier features, range too low ($\\sim 1/(10\\nu)$)", "V2": "Stretched input $\\xi = x/\\nu$",
          "V3": "Hard constraints (odd in $x$, $u(\\pm1,t)=0$)", "V4": "Learnable front term $a\\tanh(bx)$",
          "V5": "First-layer init at scale $\\sim 1/\\nu$", "V6": "Continuation in $\\nu$ + points near the shock",
          "relu": "ReLU control", "combined": "Combined validated variants"}
GROUP_METRICS = ("l2_global", "l2_near", "l2_away", "l2_spacetime", "err_share_low", "err_share_mid",
                 "err_share_high", "slope_pred", "slope_ref")


def load_runs(root):
    """All per-seed run records under runs/ (lr-search runs live elsewhere and are excluded)."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(root) / "runs").rglob("seed*.json"))]


def select(runs, variant, nu):
    return [r for r in runs if r["variant"] == variant and math.isclose(r["nu"], nu)]


def values(runs, metric):
    """Metric per run as an array; failed runs give +inf."""
    return np.array([np.inf if r["failed"] else r["metrics"][metric] for r in runs], dtype=float)


def _median(a, axis=None):
    with np.errstate(invalid="ignore"):
        return np.median(a, axis=axis)


def bootstrap(groups, stat, n, seed, ci):
    """Point value and percentile CI of stat(medians of groups), resampling each group independently."""
    rng = np.random.default_rng(seed)
    point = stat([_median(g) for g in groups])
    meds = [_median(g[rng.integers(0, g.size, (n, g.size))], axis=1) for g in groups]
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = stat(meds)
    alpha = (1 - ci) / 2
    finite = draws[~np.isnan(draws)]
    if finite.size < draws.size / 2:  # mostly undefined (e.g. inf / inf): no interval
        return float(point), float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha, 1 - alpha], method="inverted_cdf")  # no interpolation across inf
    return float(point), float(lo), float(hi)


def _ratio(m):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(m[0]) / np.asarray(m[1])


def rho(var, base, bs):
    """rho = median(var) / median(base) with its bootstrap CI."""
    return bootstrap([var, base], _ratio, bs["n"], bs["seed"], bs["ci"])


def verdict(lo, hi, threshold):
    """Four-label verdict of docs/PHASE2.md for a ratio CI [lo, hi]."""
    if math.isnan(lo) or math.isnan(hi):
        return "not supported"
    if hi < threshold:
        return "validated"
    if hi < 1:
        return "partial"
    if lo > 1:
        return "harmful"
    return "not supported"


def comparison_label(lo, hi):
    """Label for 'stat > 1' predictions: supported (CI above 1), contradicted (CI below 1), else not supported."""
    if not math.isnan(lo) and lo > 1:
        return "supported"
    if not math.isnan(hi) and hi < 1:
        return "contradicted"
    return "not supported"


def increases_as_nu_decreases(smooth, hard, bs):
    """H1: median at the smallest nu / median at the largest nu, with CI; > 1 means it increased."""
    return bootstrap([hard, smooth], _ratio, bs["n"], bs["seed"], bs["ci"])


def less_gain_than(var, other, bs):
    """H2 clause 2: median(var) / median(other) at one nu; > 1 means var lowers the error less than other."""
    return bootstrap([var, other], _ratio, bs["n"], bs["seed"], bs["ci"])


def smaller_gain_at_smooth(var_s, base_s, var_h, base_h, bs):
    """H8: rho at the smooth nu / rho at the hard nu; > 1 means the gain is smaller in the smooth case."""
    return bootstrap([var_s, base_s, var_h, base_h], lambda m: _ratio([_ratio(m[:2]), _ratio(m[2:])]),
                     bs["n"], bs["seed"], bs["ci"])


def _nu_text(nu):
    return "all" if nu == "all" else f"{nu * math.pi:.3g}/\\pi"


def evaluate_prediction(cfg, runs, pid, c):
    """One scorecard row for one clause of a prediction."""
    bs, nus = cfg["bootstrap"], cfg["nu_list"]
    row = {"id": pid, "variant": c["variant"], "nu": c["nu"], "metric": c["metric"], "threshold": c["threshold"],
           "direction": c["direction"], "value": float("nan"), "lo": float("nan"), "hi": float("nan"), "verdict": "not run"}

    def vals(variant, nu):
        return values(select(runs, variant, nu), c["metric"])

    def have(*pairs):
        return all(any(math.isclose(nu, n) for n in nus) and vals(v, nu).size for v, nu in pairs)

    smooth, hard = max(nus), min(nus)
    d = c["direction"]
    if d in ("above", "below") and pid == "G0":
        if have((c["variant"], c["nu"])):
            row["value"] = float(_median(vals(c["variant"], c["nu"])))
            ok = row["value"] > c["threshold"] if d == "above" else row["value"] < c["threshold"]
            row["verdict"] = "pass" if ok else "fail"
    elif d == "below":
        if have((c["variant"], c["nu"]), ("V0", c["nu"])):
            row["value"], row["lo"], row["hi"] = rho(vals(c["variant"], c["nu"]), vals("V0", c["nu"]), bs)
            row["verdict"] = verdict(row["lo"], row["hi"], c["threshold"])
    elif d == "no_effect":
        if have((c["variant"], c["nu"]), ("V0", c["nu"])):
            row["value"], row["lo"], row["hi"] = rho(vals(c["variant"], c["nu"]), vals("V0", c["nu"]), bs)
            row["verdict"] = "supported" if verdict(row["lo"], row["hi"], 1.0) == "not supported" else "contradicted"
    elif d == "increases_as_nu_decreases":
        if len(nus) > 1 and have((c["variant"], smooth), (c["variant"], hard)):
            row["value"], row["lo"], row["hi"] = increases_as_nu_decreases(vals(c["variant"], smooth), vals(c["variant"], hard), bs)
            row["verdict"] = comparison_label(row["lo"], row["hi"])
    elif d == "less_gain_than":
        if have((c["variant"], c["nu"]), (c["other"], c["nu"])):
            row["value"], row["lo"], row["hi"] = less_gain_than(vals(c["variant"], c["nu"]), vals(c["other"], c["nu"]), bs)
            row["verdict"] = comparison_label(row["lo"], row["hi"])
    elif d == "smaller_gain_at_smooth":
        if len(nus) > 1 and have((c["variant"], smooth), ("V0", smooth), (c["variant"], hard), ("V0", hard)):
            row["value"], row["lo"], row["hi"] = smaller_gain_at_smooth(
                vals(c["variant"], smooth), vals("V0", smooth), vals(c["variant"], hard), vals("V0", hard), bs)
            row["verdict"] = comparison_label(row["lo"], row["hi"])
    return row


def gate_check(cfg, root):
    """G0 from the V0 runs: PASS only if every clause holds."""
    runs = load_runs(root)
    clauses = []
    for c in cfg["predictions"]["G0"]:
        row = evaluate_prediction(cfg, runs, "G0", c)
        clauses.append({"text": f"V0 median {c['metric']} at nu = {_nu_text(c['nu']).replace(chr(92), '')} "
                                f"{c['direction']} {c['threshold']:g}", "observed": row["value"], "ok": row["verdict"] == "pass"})
    return {"pass": all(c["ok"] for c in clauses), "clauses": clauses}


def median_seed(group):
    """Index of the run whose near-shock error is the median (lower median for an even count)."""
    err = values(group, "l2_near")
    order = np.argsort(err, kind="stable")
    return int(order[(len(err) - 1) // 2])


def summarize_group(cfg, root, group):
    bs = cfg["bootstrap"]
    out = {"variant": group[0]["variant"], "nu": group[0]["nu"], "n": len(group),
           "n_failed": sum(r["failed"] for r in group), "params": group[0]["params"]["params"],
           "param_rel_gap": group[0]["params"]["rel_gap"], "param_matched": group[0]["params"]["matched"],
           "exploratory": group[0]["spec"]["exploratory"], "lr": group[0]["lr"],
           "wall_seconds": float(sum(r["wall_seconds"] for r in group))}
    for m in GROUP_METRICS:
        v = values(group, m) if all(r["failed"] or m in r["metrics"] for r in group) else np.array([np.nan])
        q25, med, q75 = (np.quantile(v, [0.25, 0.5, 0.75], method="inverted_cdf") if not np.isnan(v).any()
                         else (np.nan,) * 3)
        out[m] = {"median": float(med), "q25": float(q25), "q75": float(q75)}
    near = values(group, "l2_near")
    _, lo, hi = bootstrap([near], lambda m: m[0], bs["n"], bs["seed"], bs["ci"])
    out["l2_near"].update(ci_lo=lo, ci_hi=hi)
    ms = group[median_seed(group)]
    out["median_seed"] = ms["seed"]
    from burgers.pinn.runner import run_base  # local import: runner imports torch-heavy modules
    npz = run_base(root, ms["variant"], ms["nu"], ms["seed"]).with_suffix(".npz")
    out["median_seed_npz"] = str(npz) if npz.is_file() else None
    out["median_seed_curve"] = ms.get("curve", [])
    return out


def resolve_combined(rows, candidates):
    """Variants (from `candidates`) all of whose ratio-threshold clauses are validated."""
    out = []
    for v in candidates:
        mine = [r for r in rows if r["variant"] == v and r["direction"] == "below" and r["id"] != "G0"]
        if mine and all(r["verdict"] == "validated" for r in mine):
            out.append(v)
    return out


def build_summary(cfg, root, doc_text):
    """Write results/phase2/summary.json; aborts if the config's thresholds differ from docs/PHASE2.md."""
    check_against_doc(cfg, doc_text)
    root = Path(root)
    runs = load_runs(root)
    keys = sorted({(r["variant"], r["nu"]) for r in runs}, key=lambda k: (k[0], -k[1]))
    groups = [summarize_group(cfg, root, select(runs, v, nu)) for v, nu in keys]
    rows = [evaluate_prediction(cfg, runs, pid, c) for pid, clauses in cfg["predictions"].items() for c in clauses]
    lr_runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "lr_search").rglob("*.json"))]
    candidates = [v for tier in ("A", "B") for v in cfg["tiers"].get(tier, []) if v != "V0"]
    gate = gate_check(cfg, root)
    summary = {"config_hash": cfg["hash"], "n_runs": len(runs), "n_failed": sum(r["failed"] for r in runs),
               "n_lr_search_runs": len(lr_runs), "skip_gate_used": any(r.get("skip_gate") for r in runs),
               "cost_seconds": float(sum(r["wall_seconds"] for r in runs) + sum(r["wall_seconds"] for r in lr_runs)),
               "git_hashes": sorted({r["git_hash"] + ("-dirty" if r["git_dirty"] else "") for r in runs}),
               "gate": gate, "groups": groups, "predictions": rows,
               "combined_resolution": resolve_combined(rows, candidates),
               "nu_list": cfg["nu_list"], "bootstrap": cfg["bootstrap"]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
