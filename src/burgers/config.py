"""Load and validate the YAML run configuration into a flat, read-only dataclass.

Every number the pipeline uses (grid, viscosities, tolerances, fit window) comes from here,
so no physical or numerical constant is hard-coded in the analysis modules.
Gotcha: the raw YAML text is kept in `Config.raw_text` so each run can store an exact copy.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

SPLIT_SCHEMES = ("interleaved",)


@dataclass(frozen=True)
class Config:
    x_min: float
    x_max: float
    t_max: float
    nx: int
    nt: int
    nu_list: tuple
    seed: int
    split_scheme: str
    ref_dz: float
    ref_log_cutoff: float
    min_points_across_shock: float
    pod_tolerances: tuple
    pod_energy_ranks: tuple
    spectrum_times: tuple
    fit_k_range: tuple
    spectrum_floor: float
    output_root: str
    raw_text: str = ""

    def grid(self):
        """Return the space grid x[nx] and time grid t[nt] (both uniform, endpoints included)."""
        x = np.linspace(self.x_min, self.x_max, self.nx)
        t = np.linspace(0.0, self.t_max, self.nt)
        return x, t


def _check(condition, message):
    if not condition:
        raise ValueError(f"invalid config: {message}")


def load_config(path):
    """Read a YAML file and return a validated Config."""
    text = Path(path).read_text(encoding="utf-8")
    d = yaml.safe_load(text)
    p, ref, pod, spec = d["problem"], d["reference"], d["pod"], d["spectrum"]
    cfg = Config(
        x_min=float(p["x_min"]),
        x_max=float(p["x_max"]),
        t_max=float(p["t_max"]),
        nx=int(p["nx"]),
        nt=int(p["nt"]),
        nu_list=tuple(float(v) for v in p["nu_list"]),
        seed=int(d["seed"]),
        split_scheme=str(d["split"]["scheme"]),
        ref_dz=float(ref["dz"]),
        ref_log_cutoff=float(ref["log_cutoff"]),
        min_points_across_shock=float(d["resolution"]["min_points_across_shock"]),
        pod_tolerances=tuple(float(v) for v in pod["tolerances"]),
        pod_energy_ranks=tuple(int(v) for v in pod["energy_ranks"]),
        spectrum_times=tuple(float(v) for v in spec["times"]),
        fit_k_range=tuple(int(v) for v in spec["fit_k_range"]),
        spectrum_floor=float(spec["floor"]),
        output_root=str(d["output"]["root"]),
        raw_text=text,
    )
    _check(cfg.x_min < cfg.x_max and cfg.t_max > 0, "domain must be non-empty")
    _check(cfg.nx >= 8 and cfg.nt >= 3, "need nx >= 8 and nt >= 3")
    _check(len(cfg.nu_list) > 0 and all(nu > 0 for nu in cfg.nu_list), "viscosities must be positive")
    _check(cfg.split_scheme in SPLIT_SCHEMES, f"split.scheme must be one of {SPLIT_SCHEMES}")
    _check(cfg.ref_dz > 0 and cfg.ref_log_cutoff > 0, "reference dz and log_cutoff must be positive")
    _check(all(0 < tol < 1 for tol in cfg.pod_tolerances), "pod tolerances must lie in (0, 1)")
    _check(all(r >= 1 for r in cfg.pod_energy_ranks), "pod energy ranks must be >= 1")
    _check(all(0 < s <= cfg.t_max for s in cfg.spectrum_times), "spectrum times must lie in (0, t_max]")
    _check(len(cfg.fit_k_range) == 2 and 1 <= cfg.fit_k_range[0] < cfg.fit_k_range[1], "bad fit_k_range")
    _check(0 < cfg.spectrum_floor < 1, "spectrum floor must lie in (0, 1)")
    return cfg
