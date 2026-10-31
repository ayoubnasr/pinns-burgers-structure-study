"""Metrics for one trained PINN at t = T, against the Phase 1 Hopf-Cole reference.

Relative L2 errors (burgers.metrics.rel_l2) globally, in a window around the shock and outside it; the
Fourier amplitudes (burgers.spectrum) of the error and of the prediction; the share of error energy
sum |e_k|^2 in low/mid/high bands of k / k_max; and the fitted log-log slope of the prediction's spectrum.
Gotcha: the near-shock window is centered on the reference's steepest point (found numerically) with
half-width halfwidth_factor * nu, widened so it always contains at least min_points grid points.
"""

from functools import lru_cache

import numpy as np
import torch

from burgers.metrics import rel_l2
from burgers.pinn.losses import as_tensor
from burgers.reference import solve
from burgers.spectrum import fit_slope, fourier_amplitudes


@lru_cache(maxsize=8)
def _reference(nu, T, nx, cnx, cnt):
    x = np.linspace(-1.0, 1.0, nx)
    xc, tc = np.linspace(-1.0, 1.0, cnx), np.linspace(0.0, T, cnt)
    return x, solve(nu, x, np.array([T]))[0], xc, tc, solve(nu, xc, tc)


def reference(cfg, nu):
    """(x, u_ref at T, coarse x, coarse t, coarse u_ref[nt, nx]); cached per process."""
    e = cfg["eval"]
    return _reference(float(nu), float(cfg["T"]), e["nx"], e["coarse"]["nx"], e["coarse"]["nt"])


def near_shock_window(x, u_ref, nu, halfwidth_factor, min_points):
    """Return (center, halfwidth, mask) of the window around the steepest point of u_ref."""
    dx = x[1] - x[0]
    center = x[np.argmax(np.abs(np.gradient(u_ref, dx)))]
    half = max(halfwidth_factor * nu, np.ceil((min_points - 1) / 2) * dx * (1 + 1e-9))
    return float(center), float(half), np.abs(x - center) <= half


def band_shares(amp, bands):
    """Fraction of sum(amp^2) in each band of k / k_max (last band includes k_max)."""
    k = np.arange(amp.size) / (amp.size - 1)
    energy = amp ** 2
    total = energy.sum()
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (k >= lo) & ((k < hi) | (hi == 1.0))
        out[name] = float(energy[mask].sum() / total) if total > 0 else 0.0
    return out


def predict(model, x, t):
    """Model output as a numpy array, for 1-D x and t of equal length."""
    with torch.no_grad():
        return model(as_tensor(x), as_tensor(t)).numpy().ravel()


def metrics_from_prediction(cfg, nu, u_pred, uc_pred=None):
    """All metrics for a prediction u_pred on the eval grid at T (and optionally on the coarse grid)."""
    e = cfg["eval"]
    x, u_ref, _, _, uc_ref = reference(cfg, nu)
    _, _, mask = near_shock_window(x, u_ref, nu, e["near_shock"]["halfwidth_factor"], e["near_shock"]["min_points"])
    k, err_amp = fourier_amplitudes(u_pred - u_ref, x)
    _, pred_amp = fourier_amplitudes(u_pred, x)
    _, ref_amp = fourier_amplitudes(u_ref, x)
    shares = band_shares(err_amp, e["bands"])
    m = {"l2_global": rel_l2(u_pred, u_ref), "l2_near": rel_l2(u_pred[mask], u_ref[mask]),
         "l2_away": rel_l2(u_pred[~mask], u_ref[~mask]),
         **{f"err_share_{name}": v for name, v in shares.items()},
         "slope_pred": fit_slope(k, pred_amp, e["fit_k_range"], e["floor"])["slope"],
         "slope_ref": fit_slope(k, ref_amp, e["fit_k_range"], e["floor"])["slope"],
         "near_points": int(mask.sum())}
    if uc_pred is not None:
        m["l2_spacetime"] = rel_l2(uc_pred, uc_ref)
    return m, {"x": x, "u_pred": u_pred, "u_ref": u_ref, "err_amp": err_amp, "pred_amp": pred_amp, "ref_amp": ref_amp}


def evaluate_run(model, cfg, nu):
    """Metrics dict and arrays (for the run's .npz) of a trained model."""
    x, _, xc, tc, _ = reference(cfg, nu)
    u_pred = predict(model, x, np.full_like(x, cfg["T"]))
    X, Tg = np.meshgrid(xc, tc)
    uc_pred = predict(model, X.ravel(), Tg.ravel()).reshape(X.shape)
    m, arrays = metrics_from_prediction(cfg, nu, u_pred, uc_pred)
    arrays.update(xc=xc, tc=tc, uc_pred=uc_pred)
    return m, arrays
