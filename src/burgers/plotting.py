"""All figures, in one consistent style (saved as PNG at 200 dpi and as SVG).

Each viscosity keeps the same color and line style in every figure (fixed order, never by rank), and
every legend entry names its nu. All quantities are dimensionless, which the axis labels state.
Gotcha: amplitudes at round-off level are clipped to a display floor so log axes stay readable.
"""

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (blue, orange, aqua)
STYLES = ["-", "--", ":"]
DPI = 200
DISPLAY_FLOOR = 1e-17  # below double-precision round-off relative to O(1) amplitudes

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5, "lines.linewidth": 1.5,
    "legend.frameon": False,
})


def nu_label(nu):
    return f"ν = {nu * np.pi:.3g}/π"


def _style(i):
    return {"color": COLORS[i % len(COLORS)], "linestyle": STYLES[i % len(STYLES)]}


def _save(fig, run_dir, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(run_dir / f"{name}.{ext}", dpi=DPI)
    plt.close(fig)


def plot_spectrum(run_dir, nu_list, times, spectra, fits, k_range):
    """spectra[(nu, t)] = (k, amp); fits[(nu, t)] = fit dict. One panel per time, last time = post-shock."""
    fig, axes = plt.subplots(1, len(times), figsize=(3.4 * len(times), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, times):
        for i, nu in enumerate(nu_list):
            k, amp = spectra[(nu, t)]
            ax.loglog(k[1:], np.maximum(amp[1:], DISPLAY_FLOOR), label=nu_label(nu), **_style(i))
            f = fits[(nu, t)]
            if np.isfinite(f["slope"]):
                kk = np.array([f["k_min"], f["k_max"]], dtype=float)
                ax.loglog(kk, 10 ** f["intercept"] * kk ** f["slope"], color="black", linewidth=0.8)
        ax.axvspan(*k_range, color="#eeeeee", zorder=0)
        ax.set_title(f"t = {t:g}")
        ax.set_xlabel("mode number k (wavenumber πk, dimensionless)")
    axes[0].set_ylabel("|û_k| (dimensionless)")
    axes[0].legend(title="black: log-log fit in shaded window", title_fontsize=7)
    _save(fig, run_dir, "fig_spectrum")


def plot_singular_values(run_dir, nu_list, svals):
    """svals[nu] = singular values s (descending); plotted as s_i / s_1 on a log y-axis."""
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for i, nu in enumerate(nu_list):
        s = svals[nu]
        ax.semilogy(np.arange(1, s.size + 1), np.maximum(s / s[0], DISPLAY_FLOOR), label=nu_label(nu), **_style(i))
    ax.set_xlabel("mode index i")
    ax.set_ylabel("normalized singular value s_i / s_1")
    ax.legend()
    _save(fig, run_dir, "fig_singular_values")


def plot_modes_vs_error(run_dir, nu_list, errors, tolerances):
    """errors[nu][r] = relative test projection error with r modes (r = 0..m)."""
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for i, nu in enumerate(nu_list):
        e = errors[nu]
        ax.semilogy(np.arange(1, e.size), np.maximum(e[1:], DISPLAY_FLOOR), label=nu_label(nu), **_style(i))
    for tol in tolerances:
        ax.axhline(tol, color="#555555", linewidth=0.8)
        ax.annotate(f"tol = {tol:g}", (1, tol), textcoords="offset points", xytext=(2, 2), fontsize=7)
    ax.set_xlabel("number of POD modes r")
    ax.set_ylabel("relative test projection error")
    ax.legend()
    _save(fig, run_dir, "fig_modes_vs_error")
