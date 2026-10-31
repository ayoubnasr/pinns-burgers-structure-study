import numpy as np

from burgers.spectrum import fit_slope, fourier_amplitudes

X = np.linspace(-1.0, 1.0, 2049)  # 2048 periodic points
K_RANGE = (10, 100)
FLOOR = 1e-12


def test_sawtooth_has_slope_minus_one():
    # f(x) = x jumps at the period boundary; exact sine coefficients 2 / (pi k)
    k, amp = fourier_amplitudes(X, X)
    assert np.allclose(amp[1:20], 2 / (np.pi * k[1:20]), rtol=1e-3)
    fit = fit_slope(k, amp, K_RANGE, FLOOR)
    assert abs(fit["slope"] + 1.0) < 0.02 and fit["r2"] > 0.999


def test_triangle_wave_has_slope_minus_two():
    # |x| has a kink: odd-k coefficients 4 / (pi^2 k^2); even-k ones vanish and are dropped by the floor
    k, amp = fourier_amplitudes(np.abs(X), X)
    fit = fit_slope(k, amp, K_RANGE, FLOOR)
    assert abs(fit["slope"] + 2.0) < 0.05
    assert fit["n_points"] == 45  # the odd k in [10, 100]


def test_analytic_signal_decays_faster_than_any_jump():
    k, amp = fourier_amplitudes(np.sin(np.pi * X) / (1.5 - np.cos(np.pi * X)), X)
    fit = fit_slope(k, amp, K_RANGE, FLOOR)
    assert fit["slope"] < -5.0


def test_too_few_points_gives_nan():
    k, amp = fourier_amplitudes(np.sin(np.pi * X), X)
    assert np.isnan(fit_slope(k, amp, K_RANGE, FLOOR)["slope"])
