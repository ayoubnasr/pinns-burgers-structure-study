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
