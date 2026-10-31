"""Fourier analysis of one snapshot u(x, t) on [-1, 1].

Since u(-1) = u(1) = 0 and u is odd, its 2-periodic extension is continuous, so the FFT of the samples
(duplicate endpoint x = 1 dropped) gives the sine-series coefficients |u_hat_k| of mode sin(pi k x).
Why a shock decays slowly: integrating by parts, a jump makes |u_hat_k| ~ 1/k (slope -1 in log-log);
a smooth front of width delta looks like a jump for k << 1/delta and decays fast only beyond that.
Gotcha: the fitted log-log slope is an *estimate* over a finite window, not a law; amplitudes at
round-off level are excluded from the fit.
"""

import numpy as np


def fourier_amplitudes(u, x):
    """Return mode numbers k = 0..N/2 and one-sided amplitudes |u_hat_k| (series-coefficient scale)."""
    samples = np.asarray(u, dtype=float)[:-1]  # drop x = x_max, the duplicate of x_min after extension
    n = samples.size
    amp = np.abs(np.fft.rfft(samples)) / n
    amp[1:] *= 2.0  # positive and negative frequencies combined
    if n % 2 == 0:
        amp[-1] /= 2.0  # the Nyquist mode has no partner
    return np.arange(amp.size), amp


def fit_slope(k, amp, k_range, floor):
    """Least-squares fit of log10|u_hat_k| = slope * log10 k + c over k in k_range.

    Points with amp <= floor * max(amp) are treated as round-off and left out. Returns a dict with the
    slope, intercept, R^2 and the number and range of points used (slope is NaN if fewer than 3).
    """
    k = np.asarray(k)
    mask = (k >= k_range[0]) & (k <= k_range[1]) & (amp > floor * amp.max())
    n = int(mask.sum())
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "n_points": n, "k_min": None, "k_max": None}
    lx, ly = np.log10(k[mask]), np.log10(amp[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((ly - ly.mean()) ** 2)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "n_points": n, "k_min": int(k[mask].min()), "k_max": int(k[mask].max())}
