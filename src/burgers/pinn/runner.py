"""Run queue for Phase 2: learning-rate search, seeds, resume, parallel workers and the pre-registration guard.

For each (variant, nu), seed 0 is trained at every learning rate in lr_grid; the lr with the lowest held-out
PDE + IC + BC loss is kept (that run becomes seed 0, the others stay under lr_search/ and are excluded from
statistics), then seeds 1.. run with the chosen lr. The reference error is never used for selection.
Gotcha: training refuses to start unless the tag `phase2-predictions` exists and docs/PHASE2.md is unchanged
since that tag, so the predictions provably precede the runs.
"""

import json
import math
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from burgers import io
from burgers.pinn.evaluate import evaluate_run, near_shock_window, reference
from burgers.pinn.train import setup_torch, train_run
from burgers.pinn.variants import expand, resolve

TAG, DOC = "phase2-predictions", "docs/PHASE2.md"


class GuardError(RuntimeError):
    pass


def check_guard(repo="."):
    """Raise GuardError unless the tag exists and docs/PHASE2.md matches the tagged version."""
    tag = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}"], cwd=repo, capture_output=True)
    if tag.returncode != 0:
        raise GuardError(f"git tag '{TAG}' is missing: commit {DOC} and tag it before any training")
    diff = subprocess.run(["git", "diff", "--quiet", TAG, "--", DOC], cwd=repo, capture_output=True)
    if diff.returncode != 0:
        raise GuardError(f"{DOC} differs from the version tagged '{TAG}': training is blocked")


def nu_tag(nu):
    return f"nu{nu * math.pi:.3g}pi"


def run_base(root, variant, nu, seed):
    return Path(root) / "runs" / variant / nu_tag(nu) / f"seed{seed}"


def lr_base(root, variant, nu, lr):
    return Path(root) / "lr_search" / variant / nu_tag(nu) / ("lr" + f"{lr:g}".replace(".", "p"))  # no dot: with_suffix() would cut it


def git_state():
    h = io.git_hash()
    return h.replace("-dirty", ""), h.endswith("-dirty")


def shock_window(cfg, nu):
    x, u_ref, *_ = reference(cfg, nu)
    e = cfg["eval"]["near_shock"]
    center, half, _ = near_shock_window(x, u_ref, nu, e["halfwidth_factor"], e["min_points"])
    return center - half, center + half


def execute(job):
    """Train and evaluate one run, write <base>.json (+ .npz); returns a one-line progress message."""
    setup_torch()
    cfg, variant, nu, seed, lr = job["cfg"], job["variant"], job["nu"], job["seed"], job["lr"]
    base = Path(job["base"])
    base.parent.mkdir(parents=True, exist_ok=True)
    spec, match = resolve(cfg, variant, job.get("combine", ()))
    commit, dirty = git_state()
    meta = {"variant": variant, "nu": nu, "seed": seed, "lr": lr, "lr_search": job["lr_search"],
            "skip_gate": job.get("skip_gate", False), "config_hash": cfg["hash"], "git_hash": commit, "git_dirty": dirty,
            "spec": spec, "params": match, "failed": False, "reason": "", "metrics": {}}
    start = time.perf_counter()
    try:
        window = shock_window(cfg, nu) if spec["near_shock_fraction"] else None
        res = train_run(cfg, spec, nu, seed, lr, window)
        meta.update({k: res[k] for k in ("failed", "reason", "steps", "curve", "final_loss", "val_loss", "val_parts",
                                         "steps_to_target", "schedule", "train_seconds")})
        if not res["failed"]:
            meta["metrics"], arrays = evaluate_run(res["model"], cfg, nu)
            np.savez_compressed(base.with_suffix(".npz"), **arrays)
    except Exception as exc:  # one broken run must not stop the queue; it is recorded as failed
        meta.update(failed=True, reason=f"{type(exc).__name__}: {exc}", val_loss=float("inf"))
    meta["wall_seconds"] = time.perf_counter() - start
    base.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    status = "FAILED " + meta["reason"] if meta["failed"] else f"l2_near={meta['metrics']['l2_near']:.3g}"
    return f"{variant} {nu_tag(nu)} seed{seed} lr={lr:g}{' (lr search)' if job['lr_search'] else ''}: " \
           f"{status} [{meta['wall_seconds']:.0f} s]"


def _run_jobs(job_list, jobs, log):
    if not job_list:
        return
    if jobs <= 1:
        for job in job_list:
            log(execute(job))
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for fut in as_completed([pool.submit(execute, job) for job in job_list]):
            log(fut.result())


def select_lr(cfg, root, variant, nu):
    """Copy the lr-search run with the lowest validation loss to seed 0; return the chosen lr."""
    vals = {}
    for lr in cfg["train"]["lr_grid"]:
        meta = json.loads(lr_base(root, variant, nu, lr).with_suffix(".json").read_text(encoding="utf-8"))
        vals[lr] = meta["val_loss"] if not meta["failed"] else float("inf")
    best = min(vals, key=lambda lr: (vals[lr], -lr))  # ties (e.g. all failed) go to the larger lr
    src, dst = lr_base(root, variant, nu, best), run_base(root, variant, nu, cfg["seeds"][0])
    if not dst.with_suffix(".json").exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
        meta.update(lr_search=False, selected_from_lr_search=True, lr_candidates=vals)
        dst.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
        if src.with_suffix(".npz").exists():
            shutil.copyfile(src.with_suffix(".npz"), dst.with_suffix(".npz"))
    return best


def run_variants(cfg, names, jobs=1, resume=False, skip_gate=False, combine=(), log=print):
    """lr search, then the remaining seeds, for every (variant, nu); skips runs whose output exists."""
    root = Path(cfg["output"]["root"])
    names = expand(cfg, names)
    common = {"cfg": cfg, "skip_gate": skip_gate, "combine": tuple(combine)}
    lr_jobs = [dict(common, variant=v, nu=nu, seed=cfg["seeds"][0], lr=lr, lr_search=True, base=str(lr_base(root, v, nu, lr)))
               for v in names for nu in cfg["nu_list"] for lr in cfg["train"]["lr_grid"]]
    todo = [j for j in lr_jobs if not Path(j["base"]).with_suffix(".json").exists()]
    if len(todo) < len(lr_jobs) and not resume:
        raise FileExistsError(f"outputs already exist under {root}; pass --resume to skip finished runs")
    _run_jobs(todo, jobs, log)
    seed_jobs = []
    for v in names:
        for nu in cfg["nu_list"]:
            lr = select_lr(cfg, root, v, nu)
            seed_jobs += [dict(common, variant=v, nu=nu, seed=s, lr=lr, lr_search=False, base=str(run_base(root, v, nu, s)))
                          for s in cfg["seeds"][1:]]
    _run_jobs([j for j in seed_jobs if not Path(j["base"]).with_suffix(".json").exists()], jobs, log)


def estimate(cfg, jobs, steps=200, log=print):
    """Time `steps` baseline steps per nu, extrapolate to adam_steps and print totals for the gate and tiers."""
    setup_torch()
    probe = json.loads(json.dumps(cfg))
    probe["train"]["adam_steps"] = steps
    per_run = {}
    spec, _ = resolve(cfg, "V0")
    for nu in cfg["nu_list"]:
        res = train_run(probe, spec, nu, 0, cfg["train"]["lr_grid"][0])
        t0 = time.perf_counter()
        evaluate_run(res["model"], cfg, nu)
        per_step = res["train_seconds"] / steps
        per_run[nu] = per_step * cfg["train"]["adam_steps"] + (time.perf_counter() - t0)
        log(f"{nu_tag(nu)}: {per_step * 1e3:.1f} ms/step -> about {per_run[nu] / 60:.1f} min per run")
    runs_per_group = len(cfg["train"]["lr_grid"]) + len(cfg["seeds"]) - 1
    group_cost = sum(per_run.values()) * runs_per_group
    plan = {"gate": ["V0"], **{f"tier {k}": [n for n in expand(cfg, v) if n not in ("V0", "pod_latent")]
                               for k, v in cfg["tiers"].items()}}
    for label, names in plan.items():
        cpu = group_cost * len(names)
        log(f"{label}: {len(names)} configs x {len(cfg['nu_list'])} nu x {runs_per_group} runs "
            f"= {len(names) * len(cfg['nu_list']) * runs_per_group} runs, about {cpu / 3600:.1f} CPU-h, "
            f"{cpu / 3600 / jobs:.1f} h wall-clock with --jobs {jobs}")
    log("Estimates use the baseline's step time; wider/deeper V7 configs and hard constraints (V3, two forward "
        "passes) cost more. Tier A excludes V0 (run by the gate); tier C counts only 'combined' (pod_latent is a stub).")
