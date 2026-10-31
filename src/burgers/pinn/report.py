"""Phase 2 figures, scorecard and README block, all generated from results/phase2/summary.json.

The scorecard lists, per prediction clause, the design choice, the observed effect rho = median error of the
variant / median error of V0 with its 95 % bootstrap CI, and the verdict. Figures use the median-seed run of
each group (the seed whose near-shock error is the median), never the best seed.
Gotcha: before any run exists the README block only says that Phase 2 has not been run; the block between
<!-- PHASE2:START --> and <!-- PHASE2:END --> is replaced idempotently and nothing else is touched.
"""

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from burgers.cli import _replace_block  # noqa: E402
from burgers.pinn.aggregate import DESIGN  # noqa: E402

# one fixed color per variant in every figure (categorical slots in fixed order; neutrals for V0 and V7)
COLORS = {"V0": "#555555", "V1a": "#2a78d6", "V1b": "#eb6834", "V2": "#1baf7a", "V3": "#eda100", "V4": "#e87ba4",
          "V5": "#008300", "V6": "#6250d6", "relu": "#e34948", "combined": "#111111"}
V7_COLOR = "#b0b0b0"
NOT_RUN = "Phase 2 has not been run. No PINN has been trained yet, so there are no Phase 2 results."
SCOPE = ("Scope: one PDE (Burgers), one baseline and small networks at three viscosities; a validated rule "
         "means \"helped here\", not \"general\".")


def _color(v):
    return V7_COLOR if v.startswith("V7_") else COLORS.get(v, "#000000")


def _nu_tex(nu):
    return f"$\\nu = {nu * math.pi:.3g}/\\pi$"


def _nu_file(nu):
    return f"nu{nu * math.pi:.3g}pi"


def _label(v):
    return "V7 sweep (exploratory)" if v.startswith("V7_") else v


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    ax.legend(seen.values(), seen.keys(), fontsize=7, frameon=False)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_error_vs_nu(summary, figs):
    groups = summary["groups"]
    variants = list(dict.fromkeys(g["variant"] for g in groups))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, v in enumerate(variants):
        for g in (g for g in groups if g["variant"] == v):
            med, lo, hi = g["l2_near"]["median"], g["l2_near"]["ci_lo"], g["l2_near"]["ci_hi"]
            if not np.isfinite(med):
                continue
            x = g["nu"] * (1 + 0.04 * (i - len(variants) / 2))  # small offset so dots do not overlap
            yerr = [[med - lo], [hi - med]] if np.isfinite(lo) and np.isfinite(hi) else None
            ax.errorbar(x, med, yerr=yerr, fmt="o", ms=5, color=_color(v), label=_label(v), capsize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("viscosity $\\nu$")
    ax.set_ylabel("near-shock relative $L^2$ error at $T$ (median, 95% CI)")
    _legend(ax)
    _save(fig, figs / "error_vs_nu.png")


def _groups_at(summary, nu):
    return [g for g in summary["groups"] if math.isclose(g["nu"], nu) and g["median_seed_npz"]]


def fig_spectra(summary, figs, nu):
    fig, ax = plt.subplots(figsize=(6, 4))
    ref_drawn = False
    for g in _groups_at(summary, nu):
        with np.load(g["median_seed_npz"]) as f:
            k = np.arange(f["err_amp"].size)
            ax.loglog(k[1:], np.maximum(f["err_amp"][1:], 1e-17), color=_color(g["variant"]), lw=1, label=_label(g["variant"]))
            if not ref_drawn:
                ax.loglog(k[1:], np.maximum(f["ref_amp"][1:], 1e-17), color="black", lw=1.5, ls="--", label="reference $u$")
                ref_drawn = True
    ax.set_xlabel("mode number $k$ (wavenumber $\\pi k$)")
    ax.set_ylabel("amplitude of (prediction $-$ reference)")
    ax.set_title(f"Error spectra at $T$, {_nu_tex(nu)}", fontsize=9)
    _legend(ax)
    _save(fig, figs / f"spectra_{_nu_file(nu)}.png")


def fig_profiles(summary, figs, nu, halfwidth):
    fig, ax = plt.subplots(figsize=(6, 4))
    ref_drawn = False
    for g in _groups_at(summary, nu):
        with np.load(g["median_seed_npz"]) as f:
            x, keep = f["x"], np.abs(f["x"]) <= halfwidth
            ax.plot(x[keep], f["u_pred"][keep], color=_color(g["variant"]), lw=1, label=_label(g["variant"]))
            if not ref_drawn:
                ax.plot(x[keep], f["u_ref"][keep], color="black", lw=1.5, ls="--", label="reference")
                ref_drawn = True
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x, T)$, median-seed run")
    ax.set_title(f"Solution near the shock, {_nu_tex(nu)}", fontsize=9)
    _legend(ax)
    _save(fig, figs / f"profiles_{_nu_file(nu)}.png")


def fig_training_curves(summary, figs):
    nus = summary["nu_list"]
    fig, axes = plt.subplots(1, len(nus), figsize=(4 * len(nus), 3.4), squeeze=False)
    for ax, nu in zip(axes[0], nus):
        for g in (g for g in summary["groups"] if math.isclose(g["nu"], nu) and g["median_seed_curve"]):
            c = np.array(g["median_seed_curve"])
            ax.semilogy(c[:, 0], c[:, 1], color=_color(g["variant"]), lw=1, label=_label(g["variant"]))
        ax.set_title(_nu_tex(nu), fontsize=9)
        ax.set_xlabel("Adam step")
    axes[0][0].set_ylabel("training loss (median-seed run)")
    _legend(axes[0][-1])
    _save(fig, figs / "training_curves.png")


def _fmt(v):
    return "-" if v is None or not np.isfinite(v) else f"{v:.3g}"


def readme_block(summary, fig_prefix):
    """Markdown for the README: scorecard, figures, then one line per failed or inconclusive prediction."""
    if not summary or summary["n_runs"] == 0:
        return NOT_RUN
    rows = summary["predictions"]
    out = ["| Design choice | Prediction | Clause | Observed $\\rho$ (95% CI) | Verdict |", "|---|---|---|---|---|"]
    for r in rows:
        nu = "all $\\nu$" if r["nu"] == "all" else _nu_tex(r["nu"])
        clause = f"{r['metric'].replace('_', ' ')}, {nu}, {r['direction'].replace('_', ' ')}"
        if r["threshold"] is not None:
            clause += f" {r['threshold']:g}"
        if r["id"] == "G0":
            obs = f"median {_fmt(r['value'])}"
        else:
            obs = f"{_fmt(r['value'])} [{_fmt(r['lo'])}, {_fmt(r['hi'])}]"
        out.append(f"| {DESIGN.get(r['variant'], r['variant'])} | {r['id']} | {clause} | {obs} | **{r['verdict']}** |")
    gate = summary["gate"]
    out += ["", f"Gate G0: **{'PASS' if gate['pass'] else 'FAIL'}**. Runs: {summary['n_runs']} "
            f"({summary['n_failed']} failed, counted as infinite error), plus {summary['n_lr_search_runs']} "
            f"learning-rate search runs; total compute {summary['cost_seconds'] / 3600:.2f} CPU-h."
            + (" Some runs were started with --skip-gate." if summary["skip_gate_used"] else ""), "", SCOPE, ""]
    figs = ["error_vs_nu.png", "training_curves.png"] + [f"{kind}_{_nu_file(nu)}.png" for nu in summary["nu_list"]
                                                          for kind in ("spectra", "profiles")]
    out += [f"![{name[:-4]}]({fig_prefix}figs/{name})" for name in figs] + [""]
    weak = [r for r in rows if r["verdict"] not in ("validated", "supported", "pass")]
    for r in weak:
        out.append(f"- {r['id']} ({r['variant']}, {r['metric'].replace('_', ' ')}): **{r['verdict']}**.")
    return "\n".join(out).rstrip()


def write_report(cfg, root, readme=None):
    """Make the figures (if results exist) and, with `readme`, regenerate its PHASE2 block."""
    root = Path(root)
    path = root / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if summary and summary["n_runs"]:
        figs = root / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        fig_error_vs_nu(summary, figs)
        fig_training_curves(summary, figs)
        for nu in summary["nu_list"]:
            fig_spectra(summary, figs, nu)
            fig_profiles(summary, figs, nu, 5 * cfg["eval"]["near_shock"]["halfwidth_factor"] * nu)
    if readme:
        readme = Path(readme)
        prefix = Path(os.path.relpath(root, readme.parent)).as_posix() + "/"
        text = readme.read_text(encoding="utf-8")
        readme.write_text(_replace_block(text, "PHASE2", readme_block(summary, prefix)), encoding="utf-8")
