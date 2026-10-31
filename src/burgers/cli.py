"""Command-line entry point: python -m burgers.cli data|spectrum|pod|report|all --config <yaml>.

`data` creates a new run folder and stores the reference snapshots; `spectrum`, `pod` and `report`
work on the latest run folder; `all` runs the four steps in one new run folder.
`report` turns summary.json into markdown (and, with --readme, rewrites the README's generated blocks),
so every number shown comes from the code, and checks each prediction from docs/PREDICTIONS.md.
Gotcha: viscosities are always reported from largest to smallest, whatever their order in the config.
"""

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np

from burgers import io, plotting
from burgers.config import load_config
from burgers.data import build_snapshots, load_snapshots, save_snapshots, split_snapshots
from burgers.pod import modes_needed, pod_basis, projection_errors
from burgers.reference import points_across_shock
from burgers.spectrum import fit_slope, fourier_amplitudes


def _nus(cfg):
    return sorted(cfg.nu_list, reverse=True)


def _nu_text(nu):
    return f"{nu * np.pi:.3g}/pi"


def step_data(cfg, run_dir):
    start = time.perf_counter()
    x, t = cfg.grid()
    resolution = []
    for nu in _nus(cfg):
        X = build_snapshots(cfg, nu)
        save_snapshots(run_dir, nu, x, t, X)
        pts = points_across_shock(X[:, -1], x)
        resolution.append({"nu": nu, "points_across_shock": pts,
                           "required": cfg.min_points_across_shock, "ok": pts >= cfg.min_points_across_shock})
    io.update_summary(run_dir, {"resolution": resolution, "data_seconds": time.perf_counter() - start})


def step_spectrum(cfg, run_dir):
    nus, spectra, fits, rows = _nus(cfg), {}, {}, []
    for nu in nus:
        x, t, X = load_snapshots(run_dir, nu)
        for ts in cfg.spectrum_times:
            j = int(np.argmin(np.abs(t - ts)))
            if abs(t[j] - ts) > 1e-9 * cfg.t_max:
                raise ValueError(f"spectrum time {ts} is not on the time grid")
            k, amp = fourier_amplitudes(X[:, j], x)
            fit = fit_slope(k, amp, cfg.fit_k_range, cfg.spectrum_floor)
            spectra[(nu, ts)], fits[(nu, ts)] = (k, amp), fit
            rows.append({"nu": nu, "t": ts, **fit})
    plotting.plot_spectrum(run_dir, nus, cfg.spectrum_times, spectra, fits, cfg.fit_k_range)
    io.update_summary(run_dir, {"spectrum": {"fit_k_range": list(cfg.fit_k_range), "floor": cfg.spectrum_floor,
                                             "fits": rows}})


def step_pod(cfg, run_dir):
    nus, svals, errors, rows = _nus(cfg), {}, {}, []
    for nu in nus:
        _, _, X = load_snapshots(run_dir, nu)
        X_train, X_test = split_snapshots(X, cfg.split_scheme)
        U, s = pod_basis(X_train)
        svals[nu], errors[nu] = s, projection_errors(U, X_test)
        ranks = [r for r in cfg.pod_energy_ranks if r <= s.size]
        rows.append({
            "nu": nu, "n_train": X_train.shape[1], "n_test": X_test.shape[1], "n_modes_available": int(s.size),
            "modes_needed": {str(tol): modes_needed(U, X_test, tol) for tol in cfg.pod_tolerances},
            # 1 - energy, summed directly from the tail so values near round-off are not lost to cancellation
            "missing_energy_at_rank": {str(r): float(np.sum(s[r:] ** 2) / np.sum(s ** 2)) for r in ranks},
            "normalized_s_at_rank": {str(r): float(s[r - 1] / s[0]) for r in ranks},
        })
    plotting.plot_singular_values(run_dir, nus, svals)
    plotting.plot_modes_vs_error(run_dir, nus, errors, cfg.pod_tolerances)
    lines = ["| nu | " + " | ".join(f"modes for tol {tol:g}" for tol in cfg.pod_tolerances) + " |",
             "|---|" + "---|" * len(cfg.pod_tolerances)]
    for row in rows:
        cells = [_modes_text(row["modes_needed"][str(tol)], row["n_modes_available"]) for tol in cfg.pod_tolerances]
        lines.append(f"| {_nu_text(row['nu'])} | " + " | ".join(cells) + " |")
    (run_dir / "table_modes_needed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    io.update_summary(run_dir, {"pod": {"tolerances": list(cfg.pod_tolerances), "split": cfg.split_scheme,
                                        "results": rows}})


def _modes_text(r, available):
    return str(r) if r is not None else f"> {available} (not reached)"


def _strictly_increasing(values):
    return all(a < b for a, b in zip(values, values[1:]))


def check_predictions(summary):
    """Compare summary numbers with predictions P1-P3; each verdict is supported/contradicted/undetermined."""
    verdicts = {}
    fits = summary["spectrum"]["fits"]
    t_post = max(f["t"] for f in fits)
    slopes = [f["slope"] for f in sorted(fits, key=lambda f: -f["nu"]) if f["t"] == t_post]
    if any(not np.isfinite(s) for s in slopes):
        verdicts["P1"] = ("undetermined", f"a slope at t = {t_post:g} could not be fitted")
    else:
        ok = _strictly_increasing(slopes)
        verdicts["P1"] = ("supported" if ok else "contradicted",
                          f"slopes at t = {t_post:g} from largest to smallest nu: " + ", ".join(f"{s:.2f}" for s in slopes))
    rows = summary["pod"]["results"]  # largest nu first
    ranks = list(rows[0]["missing_energy_at_rank"])
    failed = [f"{label} at rank {k}" for k in ranks for key, label in
              (("missing_energy_at_rank", "missing energy"), ("normalized_s_at_rank", "s_r/s_1"))
              if not _strictly_increasing([r[key][k] for r in rows])]
    verdicts["P2"] = ("contradicted" if failed else "supported",
                      ("not increasing as nu decreases: " + ", ".join(failed) + ". " if failed else "") +
                      "at ranks " + ", ".join(ranks) + " (largest to smallest nu): missing energy " +
                      "; ".join("/".join(f"{r['missing_energy_at_rank'][k]:.2e}" for k in ranks) for r in rows) +
                      "; s_r/s_1 " +
                      "; ".join("/".join(f"{r['normalized_s_at_rank'][k]:.2e}" for k in ranks) for r in rows))
    parts, status = [], "supported"
    for tol in summary["pod"]["tolerances"]:
        needed = [r["modes_needed"][str(tol)] for r in rows]
        parts.append(f"tol {tol:g}: " + ", ".join("not reached" if m is None else str(m) for m in needed))
        for a, b in zip(needed, needed[1:]):
            if a is not None and b is not None and not a < b:
                status = "contradicted"
            elif a is None and b is not None:
                status = "contradicted"
            elif a is None and b is None and status == "supported":
                status = "undetermined"
    verdicts["P3"] = (status, "modes needed from largest to smallest nu: " + "; ".join(parts))
    return verdicts


def setup_markdown(cfg):
    x, t = cfg.grid()
    return "\n".join([
        "| Item | Value |", "|---|---|",
        "| Equation | u_t + u u_x = nu u_xx |",
        f"| Domain | x in [{cfg.x_min:g}, {cfg.x_max:g}], t in [0, {cfg.t_max:g}] |",
        "| Initial / boundary | u(x,0) = -sin(pi x), u(-1,t) = u(1,t) = 0 |",
        "| Viscosities | " + ", ".join(_nu_text(nu) for nu in _nus(cfg)) + " |",
        f"| Grid | nx = {cfg.nx} (dx = {x[1] - x[0]:.3g}), nt = {cfg.nt} (dt = {t[1] - t[0]:.3g}) |",
        f"| Train / test split | {cfg.split_scheme}: even-index times train, odd-index times test |",
        "| POD tolerances | " + ", ".join(f"{tol:g}" for tol in cfg.pod_tolerances) + " |",
        "| Spectrum times | " + ", ".join(f"{s:g}" for s in cfg.spectrum_times) +
        f" (fit window k in [{cfg.fit_k_range[0]}, {cfg.fit_k_range[1]}]) |",
    ])


def results_markdown(summary, fig_prefix):
    out = [f"Run `{summary['run_id']}` (git `{summary['git_hash']}`); "
           f"reference data computed in {summary['data_seconds']:.0f} s"
           + (f", whole pipeline in {summary['total_seconds']:.0f} s." if "total_seconds" in summary else "."), ""]
    out += ["**Shock resolution at t = t_max** (grid points across the steepest front):", "",
            "| nu | points across shock | required | ok |", "|---|---|---|---|"]
    out += [f"| {_nu_text(r['nu'])} | {r['points_across_shock']:.1f} | {r['required']:g} | {'yes' if r['ok'] else 'NO'} |"
            for r in summary["resolution"]]
    sp = summary["spectrum"]
    out += ["", f"**Fourier decay.** Fitted log-log slope of |u_hat_k| over k in {sp['fit_k_range']} "
            f"(amplitudes below {sp['floor']:g} x max excluded). These are estimates over a finite window, not decay laws.", "",
            "| nu | t | slope | R^2 | points used | k range used |", "|---|---|---|---|---|---|"]
    for f in sp["fits"]:
        slope = f"{f['slope']:.2f}" if np.isfinite(f["slope"]) else "not fitted"
        r2 = f"{f['r2']:.3f}" if np.isfinite(f["r2"]) else "-"
        k_used = f"{f['k_min']}-{f['k_max']}" if f["k_min"] is not None else "-"
        out.append(f"| {_nu_text(f['nu'])} | {f['t']:g} | {slope} | {r2} | {f['n_points']} | {k_used} |")
    out += ["", f"![Fourier spectrum]({fig_prefix}fig_spectrum.png)", ""]
    pod = summary["pod"]
    tols = pod["tolerances"]
    out += [f"**POD.** Basis from {pod['results'][0]['n_train']} training snapshots, error measured on "
            f"{pod['results'][0]['n_test']} test snapshots ({pod['split']} split).", "",
            "| nu | " + " | ".join(f"modes for tol {tol:g}" for tol in tols) + " | " +
            " | ".join(f"missing energy, {k} modes" for k in pod["results"][0]["missing_energy_at_rank"]) + " |",
            "|---|" + "---|" * (len(tols) + len(pod["results"][0]["missing_energy_at_rank"]))]
    for r in pod["results"]:
        out.append(f"| {_nu_text(r['nu'])} | " +
                   " | ".join(_modes_text(r["modes_needed"][str(tol)], r["n_modes_available"]) for tol in tols) + " | " +
                   " | ".join(f"{v:.2e}" for v in r["missing_energy_at_rank"].values()) + " |")
    out += ["", f"![Singular values]({fig_prefix}fig_singular_values.png)",
            f"![Modes vs error]({fig_prefix}fig_modes_vs_error.png)", "",
            "**Observed vs predicted** (predictions in [docs/PREDICTIONS.md](docs/PREDICTIONS.md), written before these results):", "",
            "| Prediction | Verdict | Evidence |", "|---|---|---|"]
    names = {"P1": "P1 Fourier decay slower for smaller nu", "P2": "P2 singular values decay slower for smaller nu",
             "P3": "P3 more modes needed for smaller nu"}
    for key, (verdict, evidence) in check_predictions(summary).items():
        out.append(f"| {names[key]} | **{verdict}** | {evidence} |")
    return "\n".join(out)


def _replace_block(text, name, body):
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"README has no {name}:START/END markers")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def step_report(cfg, run_dir, readme=None):
    summary = io.read_summary(run_dir)
    (run_dir / "report.md").write_text(setup_markdown(cfg) + "\n\n" + results_markdown(summary, "") + "\n",
                                       encoding="utf-8")
    if readme:
        readme = Path(readme)
        prefix = Path(os.path.relpath(run_dir, readme.parent)).as_posix() + "/"
        text = readme.read_text(encoding="utf-8")
        text = _replace_block(text, "SETUP", setup_markdown(cfg))
        text = _replace_block(text, "RESULTS", results_markdown(summary, prefix))
        readme.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="burgers", description="Burgers structure study, Phase 1.")
    parser.add_argument("step", choices=["data", "spectrum", "pod", "report", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--readme", default=None, help="README to update with generated setup/results blocks")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    start = time.perf_counter()
    run_dir = io.new_run_dir(cfg) if args.step in ("data", "all") else io.latest_run_dir(cfg.output_root)
    if args.step in ("data", "all"):
        step_data(cfg, run_dir)
    if args.step in ("spectrum", "all"):
        step_spectrum(cfg, run_dir)
    if args.step in ("pod", "all"):
        step_pod(cfg, run_dir)
    if args.step == "all":
        io.update_summary(run_dir, {"total_seconds": time.perf_counter() - start})
    if args.step in ("report", "all"):
        step_report(cfg, run_dir, args.readme)
    print(f"{args.step}: done in {time.perf_counter() - start:.1f} s -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
