# Phase 2: do the design choices work?

**Status: planned, not run.** This file is committed and tagged `phase2-predictions` before any
PINN is trained. Results will be compared against it in the README, including contradictions.

## Question

Phase 1 showed that a sharper Burgers shock is harder to represent: its spectrum flattens toward
$1/k$ and a linear basis needs more modes. The README lists architecture rules that follow from
that. Phase 2 asks, for each rule: **does it improve a PINN, by how much, and where in the spectrum?**

## Design

**Problem.** Same as Phase 1: $\nu \in \{0.1/\pi,\ 0.01/\pi,\ 0.001/\pi\}$, reference solution from
Hopf–Cole. $\nu = 0.1/\pi$ is the smooth control; $0.001/\pi$ is the hard case.

**Baseline.** Tanh MLP, 4 hidden layers of 64 units, inputs $(x,t)$, Glorot init. Loss is PDE
residual + initial condition + boundary condition with fixed weights. Adam, fixed number of steps,
fixed collocation set. Every variant gets the same optimizer, steps, collocation points and loss weights.

**One change at a time.** Each variant changes exactly one row of the README table. Parameter
counts are matched to the baseline within 5%, so a gain cannot come from extra capacity.

| ID | Variant | Rule tested |
|---|---|---|
| V0 | Baseline | none |
| V1a | Fourier-feature encoding of $x$, frequencies spaced across scales up to $\sim 1/\nu$ | encoding sized from the measured spectrum |
| V1b | Same encoding, range too low (up to $\sim 1/(10\nu)$) | control: is the *sizing* what matters, or just having features? |
| V2 | Extra input $\xi = x/\nu$ | stretched coordinate |
| V3 | Hard constraints: exact oddness in $x$ and exact $u(\pm1,t)=0$ | hard-coded structure |
| V4 | Baseline plus a learnable front term $a\tanh(bx)$, $b$ initialized $\sim 1/\nu$ | localized front unit |
| V5 | First-layer weights initialized at scale $\sim 1/\nu$ | activation/initialization rule |
| V6 | Continuation in $\nu$ and residual points concentrated near the shock | training schedule |
| V7 | Width $\in\{32,64,128\}$, depth $\in\{2,4,6\}$ | **no prediction**; exploratory |

## Metrics (all at $T = 1$ unless stated)

- Relative $L^2$ error against the reference: global, near the shock ($|x| \le 10\nu$-scaled window, fixed in config), and away from the shock.
- Error spectrum: Fourier amplitudes of (prediction − reference), reported in low, mid and high wavenumber bands, plus the fitted slope of the prediction's own spectrum versus the reference's.
- Cost: wall-clock time and steps to reach a fixed training loss.
- Failed or diverged runs are counted and reported, never dropped.

## Protocol

- **Seeds:** at least 5 per configuration. Report the median and the median-seed plot, not the best seed.
- **Selection without leakage:** learning rate is chosen from $\{10^{-3}, 3\times10^{-4}\}$ per configuration using the PDE-residual loss on held-out points. Reference error is never used for selection.
- **Gate first:** run V0 on all three $\nu$ before any variant. If the gate fails, stop (see G0).
- **Effect size:** for each variant, $\rho$ = median error of the variant divided by median error of V0, with a 95% bootstrap confidence interval over seeds.
- **Verdict per prediction:** *validated* (CI upper bound below the stated threshold), *partial* (CI below 1 but not meeting the threshold), *not supported* (CI includes 1), *harmful* (CI above 1).

## Tiers (so the plan can run within a real budget)

- **Tier A:** G0, V0, V1a, V1b, V2, V3, all three $\nu$, 5 seeds.
- **Tier B:** V4, V5, V6, V7, plus a ReLU sanity control.
- **Tier C:** all helpful variants combined (does the gain add up?), and a POD-latent operator model over $\nu$.

Before launching any tier, time one run and report the total estimate. If it exceeds the budget, reduce steps or collocation points; drop configurations before dropping seeds below 5.

## Pre-registered predictions (thresholds are defaults; review before tagging)

- **G0 (the baseline fails where expected).** Median near-shock error of V0 at $\nu = 0.001/\pi$ exceeds 10%, and global error at $\nu = 0.1/\pi$ is below 1%. *Wrong if:* either fails. Then there is no spectral failure to fix at this budget, and every later result is read in that light.
- **H1.** For V0, the share of error energy in the high-wavenumber band increases as $\nu$ decreases. *Wrong if:* it does not increase from $0.1/\pi$ to $0.001/\pi$.
- **H2.** V1a lowers near-shock error at $\nu = 0.001/\pi$ by at least 30% ($\rho \le 0.7$), and V1b lowers it less than V1a. *Wrong if:* the CI for V1a includes 0.7 or V1b matches V1a.
- **H3.** V2 gives $\rho \le 0.7$ at $\nu = 0.001/\pi$ and has no significant effect at $0.1/\pi$.
- **H4.** V3 lowers global error at all $\nu$ by at least 20% ($\rho \le 0.8$).
- **H5.** V4 gives $\rho \le 0.7$ near the shock at $\nu = 0.001/\pi$.
- **H6.** V5 gives $\rho \le 0.8$ at $\nu = 0.001/\pi$.
- **H7.** V6 gives $\rho \le 0.8$ near the shock at $\nu = 0.001/\pi$.
- **H8.** Gains from the smooth case ($0.1/\pi$) are smaller than gains at $0.001/\pi$ for H2, H3 and H5, since the baseline already works there.
- **No prediction:** width and depth (V7), and the Tier C combination.

## Reporting

All outputs are generated from `results/phase2/summary.json`:
1. Dot plot of error versus $\nu$ per variant with confidence intervals.
2. Error spectra at $\nu = 0.001/\pi$ for V0 and each variant, with the reference spectrum.
3. Median-seed solution profiles near the shock.
4. Training curves.
5. A **scorecard**: design choice, prediction, observed $\rho$ with CI, verdict. This table answers how valid each choice was.

Negative and inconclusive results go in the README with the same prominence as positive ones.

## What this will not show

One PDE, one baseline, small networks, three viscosities. A validated rule means "helped here," not "general." Whether the same reasoning transfers to other PDEs is a separate question.