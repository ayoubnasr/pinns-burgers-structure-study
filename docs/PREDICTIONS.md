# Predictions (written before any analysis was run)

This file is committed and tagged `predictions` before any reference solution, spectrum or POD
analysis has been computed. Results are compared against it in the README, including contradictions.

Problem: `u_t + u u_x = nu u_xx` on `x in [-1, 1]`, `t in [0, 1]`, `u(x,0) = -sin(pi x)`,
`u(+-1, t) = 0`, with `nu in {0.1/pi, 0.01/pi, 0.001/pi}`.

## P1. Fourier decay

At the post-shock time `T = 1`, the Fourier amplitudes `|u_hat_k|` of `u(x, T)` (odd periodic
extension) decay more slowly with `k` as `nu` decreases. Concretely: the fitted log-log slope over
the configured wavenumber window is less negative (closer to the `-1` of a jump) for smaller `nu`.

**Wrong if:** the fitted slopes at `T = 1` are not ordered `slope(0.1/pi) < slope(0.01/pi) < slope(0.001/pi)`
(i.e. a smaller `nu` gives an equal or steeper decay).

## P2. POD singular-value decay

The normalized singular values `s_i / s_1` of the training snapshot matrix decay more slowly as `nu`
decreases.

**Wrong if:** for some index range the smaller-`nu` curve lies clearly below the larger-`nu` curve,
or the energy captured by a fixed number of modes is not lower for smaller `nu`.

## P3. Modes needed

For each fixed relative test-projection tolerance (1e-2 and 1e-3), the number of POD modes `r`
needed increases as `nu` decreases.

**Wrong if:** for either tolerance, `modes_needed` is not strictly increasing from `nu = 0.1/pi`
to `nu = 0.01/pi` to `nu = 0.001/pi`. Ties count as the prediction failing.

## Scope reminder

These predictions concern the *solution* and a *linear reduced basis* only. Nothing here predicts
or tests the behaviour of neural networks (spectral bias, PINN accuracy); that is Phase 2.
