"""Variant registry: each variant is the baseline spec (V0) with one declared component changed.

A spec is a plain dict (width, depth, activation, fourier, stretched, hard, front, steep, continuation,
near_shock_fraction, exploratory). Except for the V7 sweep, the hidden width is then adjusted so the
parameter count is within 5 % of V0's (`match_params`), so a gain cannot come from extra capacity.
Gotcha: V7 expands into one named config per (width, depth), e.g. "V7_w128_d2", and is never matched.
"""

import copy

from burgers.pinn.models import param_count

PARAM_TOL = 0.05
COMPONENTS = ("activation", "fourier", "stretched", "hard", "front", "steep", "continuation", "near_shock_fraction")


def base_spec(cfg):
    net = cfg["net"]
    return {"name": "V0", "width": net["width"], "depth": net["depth"], "activation": net["activation"],
            "init": net["init"], "fourier": None, "stretched": False, "hard": False, "front": None,
            "steep": None, "continuation": None, "near_shock_fraction": 0.0, "exploratory": False}


def expand(cfg, names):
    """Replace V7 by its sweep configs; other names are returned unchanged."""
    out = []
    for name in names:
        if name == "V7":
            sweep = cfg["variants"]["V7"]["sweep"]
            out += [f"V7_w{w}_d{d}" for w in sweep["width"] for d in sweep["depth"]]
        else:
            out.append(name)
    return out


def declared(cfg, name, combine=()):
    """Spec keys a variant is allowed to change relative to V0."""
    if name.startswith("V7_"):
        return {"width", "depth", "exploratory"}
    if name == "combined":
        return set().union(*(declared(cfg, n) for n in combine)) if combine else set()
    return set(cfg["variants"][name])


def build_spec(cfg, name, combine=()):
    """Spec dict for a variant name (unmatched). `combine` lists the variants merged into 'combined'."""
    spec = base_spec(cfg)
    spec["name"] = name
    if name.startswith("V7_"):
        w, d = name[3:].split("_")
        spec.update(width=int(w[1:]), depth=int(d[1:]), exploratory=True)
    elif name == "combined":
        for other in combine:
            spec.update(copy.deepcopy(cfg["variants"][other]))
        spec["combine"] = list(combine)
    elif name != "pod_latent":
        spec.update(copy.deepcopy(cfg["variants"][name]))
    return spec


def match_params(spec, target):
    """Adjust the hidden width (smallest change) so param_count is within 5 % of `target`.

    Returns (spec, info) where info has the count, relative gap and flags. V7 configs are exempt.
    """
    spec = copy.deepcopy(spec)
    w0 = spec["width"]
    if spec["exploratory"]:
        n = param_count(spec)
        return spec, {"params": n, "rel_gap": n / target - 1, "matched": False, "exempt": True}
    gaps = {w: param_count({**spec, "width": w}) / target - 1 for w in range(1, 4 * w0 + 1)}
    within = [w for w, g in gaps.items() if abs(g) <= PARAM_TOL]
    best = min(within, key=lambda w: (abs(w - w0), abs(gaps[w]))) if within else min(gaps, key=lambda w: abs(gaps[w]))
    spec["width"] = best
    return spec, {"params": param_count(spec), "rel_gap": gaps[best], "matched": bool(within), "exempt": False}


def resolve(cfg, name, combine=()):
    """Build and parameter-match a variant against V0; returns (spec, match_info)."""
    return match_params(build_spec(cfg, name, combine), param_count(base_spec(cfg)))
