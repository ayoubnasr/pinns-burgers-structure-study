"""One PINN training run: Adam with cosine learning-rate decay for a fixed number of steps.

Every variant uses the same optimizer, steps, point counts and loss weights; only its spec differs.
Continuation (V6) splits the same step budget into stages with nu decreasing geometrically from the
largest configured nu to the target, so the last stage trains at the target nu.
Gotcha: no early stopping or checkpoint selection; a NaN loss or one above `divergence_bound` stops the
run and marks it failed, and the run is still recorded (it counts as an infinite error later).
"""

import math
import time

import numpy as np
import torch

from burgers.pinn.losses import to_tensors, total_loss
from burgers.pinn.models import build_model
from burgers.pinn.sampling import point_set


def setup_torch():
    """Float64, one thread per worker and deterministic kernels, as the protocol requires."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def continuation_schedule(nu_target, steps, stages, nu_start):
    """List of (nu, n_steps) stages; a single stage when there is nothing to continue from."""
    if not stages or stages <= 1 or nu_target >= nu_start:
        return [(nu_target, steps)]
    nus = np.geomspace(nu_start, nu_target, stages)
    counts = [steps // stages] * stages
    counts[-1] += steps - sum(counts)
    return [(float(n), c) for n, c in zip(nus, counts)]


def train_run(cfg, spec, nu, seed, lr, window=None):
    """Train one model; returns a dict with the model, loss curve, validation loss and failure flag."""
    setup_torch()
    tr = cfg["train"]
    start = time.perf_counter()
    model = build_model(spec, nu, seed)
    near = spec["near_shock_fraction"] if window is not None else 0.0
    pts = to_tensors(point_set(cfg, seed, near_fraction=near, window=window))
    val = to_tensors(point_set(cfg, seed, held_out=True))
    weights = dict(tr["weights"], bc=0.0 if spec["hard"] else tr["weights"]["bc"])
    stages = (spec["continuation"] or {}).get("stages", 1)
    schedule = continuation_schedule(nu, tr["adam_steps"], stages, max(cfg["nu_list"]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["adam_steps"])
    curve, steps_to_target, failed, reason, step = [], None, False, "", 0
    for stage_nu, n_steps in schedule:
        for _ in range(n_steps):
            opt.zero_grad()
            loss, _ = total_loss(model, pts, stage_nu, weights)
            value = float(loss.detach())
            if not math.isfinite(value) or value > tr["divergence_bound"]:
                failed, reason = True, f"loss {value:.3g} at step {step}"
                break
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if steps_to_target is None and value <= tr["target_loss"]:
                steps_to_target = step
            if step % tr["log_every"] == 0 or step == tr["adam_steps"]:
                curve.append([step, value])
        if failed:
            break
    val_loss, val_parts = (float("inf"), {}) if failed else total_loss(model, val, nu, tr["weights"])
    return {"model": model, "failed": failed, "reason": reason, "steps": step, "curve": curve,
            "final_loss": curve[-1][1] if curve else float("nan"), "val_loss": float(val_loss),
            "val_parts": val_parts, "steps_to_target": steps_to_target, "lr": lr,
            "schedule": schedule, "train_seconds": time.perf_counter() - start}
