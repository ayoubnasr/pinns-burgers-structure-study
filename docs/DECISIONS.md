# Decisions

Judgment calls made while building the study, one line each with the reason.

- **Repo folder name.** The repo lives in `pinns-pipeline/` (the folder provided), not `burgers-structure-study/`; the package and layout are otherwise as specified.
- **Grid size nx = 8193 (not 2048).** With nx = 2048 the smallest viscosity (0.001/pi) had only ~2.8 grid points across the shock at t = 1 (spec requires >= 5). nx = 8193 gives ~7.4 points and a 8192-point periodic grid. The check is recomputed every run and stored in `summary.json`.
- **Shock width measure.** Points across the shock = (max u - min u) / max|u_x| / dx at t = t_max; for the viscous profile -U tanh(U x / (2 nu)) this equals 4 nu / (U dx).
- **Reference method.** Hopf-Cole quadrature with the substitution e = sqrt(4 nu t) z, trapezoid rule in z, and log-sum-exp scaling per x, which keeps it stable at the smallest nu, so no fallback is needed for production runs.
- **Quadrature step dz = 0.15, cutoff 50.** The trapezoid error bound ~exp(-pi^2 / (dz^2 (pi t + 1))) is far below round-off; measured max difference vs dz = 0.05 at nu = 0.001/pi was 4e-14. dz = 0.15 instead of 0.05 makes `make all` fit in the 5-minute budget.
- **Fallback solver.** A second-order finite-difference method-of-lines solver (`solve_fd`) is kept only as an independent cross-check at the largest nu (tolerance 1e-3 relative L2 on 513 points).
- **Extra config keys.** `reference.dz`, `reference.log_cutoff`, `resolution.min_points_across_shock`, `pod.energy_ranks` and `spectrum.floor` were added so that no numerical constant is hard-coded.
- **Seed.** Phase 1 contains no randomness; `seed` is stored in the config and summary for Phase 2.
- **make on Windows.** No `make` binary on the author's Windows machine; the Makefile is verified with `mingw32-make` (GNU make). It sets `PYTHONPATH=src`, so the targets also work before `make install`.
- **Fit window k in [10, 100].** k >= 10 skips the few large-scale modes set by the sin(pi x) initial shape; k <= 100 stays below the viscous cutoff k ~ 1/(pi delta) of the smallest nu (delta = front width from the resolution check), so for that nu the window sees the jump-like regime. For larger nu the window lies past their cutoff, so their "slope" only summarizes a fast, non-power-law decay.
- **Spectrum round-off floor 1e-12.** Amplitudes below 1e-12 x max are double-precision noise and are excluded from the fit; the number and range of points actually used is reported next to every slope.
- **Split.** Interleaved (even times train, odd times test), as specified: tests interpolation in time only.
- **POD energy measured as missing energy.** The first run computed captured energy as cumsum(s^2)/sum(s^2), which rounds to exactly 1.0 at rank 20 for two viscosities and produced a false tie. It is now computed directly as sum(s[r:]^2)/sum(s^2). Only the measurement changed; the P2 criterion in PREDICTIONS.md did not. Values below ~1e-13 relative are at round-off level.
- **Verdicts follow PREDICTIONS.md literally.** Ties count as failures (P3), and P2 requires strict ordering of both missing energy and s_r/s_1 at every configured rank. No criterion was loosened after seeing results.

## Phase 2

- **Steps and points.** `phase2 time` measured ~190-310 ms/step for the baseline with n_f = 4000 (about 16-26 min per 5000-step run). Defaults were lowered to adam_steps = 3000, n_f = 2000, which measured 35-153 ms/step on repeated timings (about 2-8 min per run; the machine's timing varied between runs).
- **Other train defaults.** n_ic = n_bc = 256, n_val = 2000, log_every = 100, divergence_bound = 1e6, target_loss = 1e-3 (the "steps to reach a fixed loss" level). Validation points are uniform for every variant (no near-shock concentration), so lr selection compares like with like.
- **Init.** "Glorot" = Xavier-normal weights with zero biases. V5 redraws the x-weight of the first round(0.25 width) units from N(0, (c/nu)^2), c = 1.
- **Fourier features.** V1a/V1b use m = 16 log-spaced modes (k from 1 to c/nu), concatenated with (x, t).
- **Front term with hard constraints.** a tanh(bx) is added before the hard-constraint transform, so a combination with V3 stays exactly odd and zero at x = +-1.
- **Continuation (V6).** 3 stages, nu geometric from the largest configured nu (0.1/pi) to the target, equal step split; at the target 0.1/pi it is a single stage. Near-shock points use the same window as the metric.
- **Parameter matching.** Hidden width only, smallest change landing within 5 %; every variant except V7 matched (largest gap V1a, +3.2 %). No exceptions.
- **Bands.** As configured: low [0, 0.1), mid [0.1, 0.4), high [0.4, 1.0] of k / k_max, k_max = Nyquist mode of the nx = 8193 eval grid.
- **Prediction clauses not fully specified in docs/PHASE2.md.** H3 clause 2 and H6 use near-shock error; H4 is checked at each nu separately; H1 compares the largest and smallest nu (its "wrong if"); comparison clauses (H1, H2 clause 2, H8) use a bootstrap CI of a ratio of medians: supported if the CI is above 1, contradicted if below, otherwise not supported ("V1b matches V1a").
- **Combined (Tier C).** "auto" = variants whose ratio-threshold clauses are all validated (V1b has none, so it is never included).
- **lr ties.** If validation losses tie (e.g. all failed), the larger lr is kept.
- **Superseded stubs.** `pinn/model.py`, `configs/pinn_*.yaml` and the `pinn` extras group were replaced by the Phase 2 files and the `phase2` extras group.
- **Make targets.** Tier/gate targets ignore the command's exit status (`-`) so the log tail is always printed; read the tail for errors.
- **Figures.** Spectra and profiles are drawn for every nu, not only the smallest. Prediction nu values absent from a reduced nu_list (smoke/tests) are reported "not run" instead of rejected.
