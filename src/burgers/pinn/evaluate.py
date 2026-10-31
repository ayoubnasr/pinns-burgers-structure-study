"""Phase 2 (not implemented): compare a trained PINN with the Phase 1 reference solution.

Planned: relative L2 error over the whole grid, the same error restricted to a band around the shock,
and the Fourier spectrum (burgers.spectrum) of the prediction and of the error at the spectrum times.
Gotcha: evaluate on the exact grid of the reference data so no interpolation error is mixed in.
"""


def evaluate(model, run_dir, nu):
    """Return a dict of error metrics and spectra for one model against the stored reference data."""
    raise NotImplementedError("Phase 2")
