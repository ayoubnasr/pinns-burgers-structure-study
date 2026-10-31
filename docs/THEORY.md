# Theory (plain language)

**Why a shock gives slowly decaying Fourier coefficients.** A Fourier coefficient measures how much of a
wave sin(pi k x) is in u. Integrating by parts once trades a factor 1/k for a derivative of u. A smooth function
can do this again and again, so its coefficients fall faster than any power of 1/k (for analytic u, exponentially).
A jump stops the process after the first step: the boundary term from the jump stays, so coefficients fall only
like 1/k, a slope of -1 on a log-log plot. A viscous shock of width delta is smooth, but only on the scale delta:
for k well below 1/delta it looks like a jump (slope near -1), and only beyond that does the decay speed up.
Smaller nu means a thinner shock, so the jump-like range reaches higher k.

**What singular values mean (Eckart-Young).** Put the snapshots u(x, t_j) as columns of a matrix X and take
its SVD, X = U S V^T. The columns of U (POD modes) are an orthonormal basis ordered by importance. Eckart-Young
says the best possible rank-r approximation of X, in the Frobenius norm, keeps the first r modes, and its error
is exactly sqrt(s_{r+1}^2 + s_{r+2}^2 + ...). So the decay of the singular values *is* the decay of the best
achievable error of any r-dimensional linear basis, for these snapshots.

**Why a moving sharp front needs many linear modes (Kolmogorov n-width).** The n-width of a set of functions is
the error of the best n-dimensional linear space for that set. Translates of a smooth bump are captured well by a
few modes, but translates of a narrow front are not: two copies shifted by more than the front width are nearly
orthogonal, so each new position adds a new direction. For a jump moving across the domain the n-width decays only
like n^(-1/2). In Burgers the shock forms, sits at x = 0 and sharpens; the more the solution looks like such a
front, the more modes POD needs.

**What Phase 1 shows and does not show.** It measures two properties of the reference solution itself: how fast
its Fourier coefficients decay, and how well a linear basis built from training snapshots reconstructs held-out
snapshots. It does not train or test any neural network, so it says nothing about spectral bias or PINN accuracy;
that is Phase 2. Slopes are fits over one wavenumber window (estimates, not laws), and POD is tested only on
interpolation in time for one initial condition.
