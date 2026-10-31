# Predictions (written before any analysis was run)

This file is committed and tagged `predictions` before any reference solution, spectrum or POD
analysis has been computed. Results are compared against it in the README, including contradictions.

*Note: the "Why I predict this" paragraphs were added after the tag, to document the reasoning
behind each prediction. The predictions and their "Wrong if" criteria are unchanged from the
tagged version (only the math notation was typeset).*

Problem: $u_t + u\,u_x = \nu\,u_{xx}$ on $x \in [-1, 1]$, $t \in [0, 1]$, with $u(x,0) = -\sin(\pi x)$,
$u(\pm 1, t) = 0$, and $\nu \in \{0.1/\pi,\ 0.01/\pi,\ 0.001/\pi\}$.

## Motivation

My research question is how the structure of a PDE shapes a learning method. It began as a puzzle
about generalization: deep networks can memorize pure noise, yet they generalize. The loss-landscape
and sharp-minima literature I read points to the optimizer's own preferences as the explanation, and
one of those preferences is spectral bias: networks fit smooth, low-frequency structure first and
sharp detail last. For a PDE whose solution develops a shock, that preference works against the
network. This argument rests on a premise that can be tested without any network: a sharper front
must make the solution harder to represent. P1 to P3 test that premise on the exact solution and on
a linear basis.

## P1. Fourier decay

At the post-shock time $T = 1$, the Fourier amplitudes $|\hat{u}_k|$ of $u(x, T)$ (odd periodic
extension) decay more slowly with $k$ as $\nu$ decreases. Concretely: the fitted log-log slope over
the configured wavenumber window is less negative (closer to the $-1$ of a jump) for smaller $\nu$.

**Why I predict this:** a jump discontinuity has Fourier coefficients that fall off like $1/k$, while
a smooth function's fall off much faster. Viscosity sets the thickness of the shock layer, roughly
$\nu$ wide. Wavenumbers below about $1/\nu$ see something close to a jump, so the spectrum follows
$1/k$; beyond that the layer is resolved and the decay turns exponential. A smaller $\nu$ pushes
that crossover to higher $k$, so a fixed fitting window sits more in the $1/k$ regime and the
fitted slope should move toward $-1$. This is the spectral content that spectral bias says a
network learns last.

**Wrong if:** the fitted slopes at $T = 1$ are not ordered
$\text{slope}(0.1/\pi) < \text{slope}(0.01/\pi) < \text{slope}(0.001/\pi)$
(i.e. a smaller $\nu$ gives an equal or steeper decay).

## P2. POD singular-value decay

The normalized singular values $s_i / s_1$ of the training snapshot matrix decay more slowly as $\nu$
decreases.

**Why I predict this:** POD is an SVD of the snapshot matrix $A$, and by the Eckart–Young theorem the
best rank-$r$ approximation error is set by the trailing singular values:

$$
\min_{\operatorname{rank}(B)\le r} \lVert A - B \rVert_F^2 \;=\; \sum_{i>r} \sigma_i^2 .
$$

So the decay of $\sigma_i$ measures how well a linear basis can compress the solution family. A
smooth family lives in a few fixed patterns. A front that steepens over time does not: its shape
changes qualitatively across the snapshots, so a fixed set of smooth modes needs more terms to
follow it. This is the Kolmogorov $n$-width problem for transport-dominated PDEs, and viscosity is
what softens it. I therefore expect slower singular-value decay at smaller $\nu$.

**Wrong if:** for some index range the smaller-$\nu$ curve lies clearly below the larger-$\nu$ curve,
or the energy captured by a fixed number of modes is not lower for smaller $\nu$.

## P3. Modes needed

For each fixed relative test-projection tolerance ($10^{-2}$ and $10^{-3}$), the number of POD modes $r$
needed increases as $\nu$ decreases.

**Why I predict this:** this is the practical form of P2. The number of modes needed for a given
error is read from the tail of the singular values, so a heavier tail at smaller $\nu$ should cost
more modes at a fixed tolerance. The criterion is strict (strictly increasing, ties count as
failing), so the prediction can fail.

**Wrong if:** for either tolerance, `modes_needed` is not strictly increasing from $\nu = 0.1/\pi$
to $\nu = 0.01/\pi$ to $\nu = 0.001/\pi$. Ties count as the prediction failing.