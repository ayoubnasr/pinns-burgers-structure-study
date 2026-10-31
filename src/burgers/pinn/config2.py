"""Load and validate configs/phase2.yaml into a plain dict, and check it against docs/PHASE2.md.

Viscosities may be written as "0.01/pi" and are evaluated to floats here, so Phase 1 and Phase 2
use the same numbers. The prediction thresholds in the config must equal the ones written in the
pre-registration file; `check_against_doc` compares the two and raises if they differ.
Gotcha: the config hash is taken over the parsed dict, so comments and spacing do not change it.
"""

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import yaml

KNOWN_DIRECTIONS = {"above", "below", "increases_as_nu_decreases", "less_gain_than", "no_effect",
                    "smaller_gain_at_smooth"}


def parse_nu(value):
    """Turn 0.01, "0.01" or "0.01/pi" into a float; "all" is returned unchanged."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "all":
        return text
    if text.endswith("/pi"):
        return float(text[:-3]) / math.pi
    return float(text)


def config_hash(cfg):
    blob = json.dumps({k: v for k, v in cfg.items() if k != "hash"}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _check(condition, message):
    if not condition:
        raise ValueError(f"invalid phase2 config: {message}")


def validate(cfg):
    """Check structure and cross-references; returns cfg with its hash refreshed."""
    for key in ("nu_list", "T", "net", "train", "seeds", "bootstrap", "eval", "variants", "predictions", "tiers", "output"):
        _check(key in cfg, f"missing key '{key}'")
    tr = cfg["train"]
    _check(tr["adam_steps"] >= 1 and len(tr["lr_grid"]) >= 1, "need adam_steps >= 1 and a non-empty lr_grid")
    _check(tr["lr_schedule"] == "cosine", "only the cosine schedule is implemented")
    _check(cfg["dtype"] == "float64", "only float64 is supported")
    _check(len(cfg["seeds"]) >= 1 and all(nu > 0 for nu in cfg["nu_list"]), "need seeds and positive viscosities")
    edges = sorted(cfg["eval"]["bands"].values())
    _check(edges[0][0] == 0 and edges[-1][1] == 1 and all(a[1] == b[0] for a, b in zip(edges, edges[1:])),
           "bands must tile [0, 1]")
    for tier, names in cfg["tiers"].items():
        for name in names:
            _check(name in cfg["variants"], f"tier {tier} lists unknown variant '{name}'")
    for pid, clauses in cfg["predictions"].items():
        for c in clauses:
            _check(c["variant"] in cfg["variants"], f"{pid} uses unknown variant '{c['variant']}'")
            _check(c["direction"] in KNOWN_DIRECTIONS, f"{pid} has unknown direction '{c['direction']}'")
            _check(c["nu"] == "all" or any(math.isclose(c["nu"], nu) for nu in cfg["nu_list"]),
                   f"{pid} uses a nu that is not in nu_list")
    cfg["hash"] = config_hash(cfg)
    return cfg


def load_phase2_config(path="configs/phase2.yaml"):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return from_dict(cfg)


def from_dict(d):
    """Parse viscosities and validate a raw config dict (a deep copy is made)."""
    cfg = copy.deepcopy(d)
    cfg["nu_list"] = [parse_nu(v) for v in cfg["nu_list"]]
    for clauses in cfg["predictions"].values():
        for c in clauses:
            c["nu"] = parse_nu(c["nu"])
    return validate(cfg)


def doc_predictions(md_text):
    """Map each prediction ID in docs/PHASE2.md (G0, H1, ...) to the sorted thresholds written in its line.

    rho thresholds are read from "$\\rho \\le X$"; for G0 (no rho) the percentages are read as fractions.
    """
    found = {}
    for line in md_text.splitlines():
        m = re.match(r"- \*\*(G0|H\d+)\b", line.strip())
        if not m:
            continue
        pid = m.group(1)
        rhos = [float(v) for v in re.findall(r"\\rho \\le ([0-9.]+)", line)]
        if pid == "G0":
            rhos = [float(v) / 100 for v in re.findall(r"([0-9.]+)\\?%", line)]
        found[pid] = sorted(set(rhos))
    return found


def check_against_doc(cfg, md_text):
    """Raise ValueError if prediction IDs or thresholds differ between the config and docs/PHASE2.md."""
    doc = doc_predictions(md_text)
    _check(set(doc) == set(cfg["predictions"]),
           f"prediction IDs differ from docs/PHASE2.md: doc {sorted(doc)}, config {sorted(cfg['predictions'])}")
    for pid, clauses in cfg["predictions"].items():
        mine = sorted({c["threshold"] for c in clauses if c["threshold"] is not None})
        _check(len(mine) == len(doc[pid]) and all(math.isclose(a, b) for a, b in zip(mine, doc[pid])),
               f"{pid} thresholds {mine} do not match docs/PHASE2.md {doc[pid]}")
