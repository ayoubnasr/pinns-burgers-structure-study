"""Burgers structure study.

Phase 1 asks whether shock formation in Burgers' equation (u_t + u u_x = nu u_xx)
makes the solution harder to represent, measured by Fourier coefficient decay and
POD singular-value decay as the viscosity nu decreases.
Phase 1 depends only on numpy/scipy/matplotlib/pyyaml; Phase 2 (burgers.pinn) imports torch lazily.
"""

__version__ = "0.1.0"
