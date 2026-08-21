"""Configurable coefficient laws for nonlinear diffusion."""

from collections.abc import Callable

import numpy as np


def regularized_coefficient(u, epsilon=0.0):
    """Evaluate normalized ``epsilon + u/(1+u)`` with value 0.5 at u=1."""
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("regularization epsilon must be finite and nonnegative.")

    values = np.asarray(u, dtype=float)
    denominator = 1.0 + values
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Nonlinear state contains nonfinite values.")
    if np.any(np.abs(denominator) <= 1e-12):
        raise FloatingPointError("Nonlinear coefficient is singular near u=-1.")

    scale = 0.5 / (0.5 + epsilon)
    coefficient = scale * (epsilon + values / denominator)
    if not np.all(np.isfinite(coefficient)):
        raise FloatingPointError("Nonlinear coefficient contains nonfinite values.")
    return coefficient


def constant_coefficient(u, value=0.5):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("constant coefficient must be finite and positive.")
    return np.full_like(np.asarray(u, dtype=float), value, dtype=float)


def build_coefficient(settings) -> Callable:
    """Return the coefficient callable selected by configuration."""
    if settings.model == "regularized":
        epsilon = settings.regularization_epsilon

        def coefficient(u, _context=None):
            return regularized_coefficient(u, epsilon)

        return coefficient
    if settings.model == "constant":
        value = settings.constant_value

        def coefficient(u, _context=None):
            return constant_coefficient(u, value)

        return coefficient
    raise ValueError(f"Unknown nonlinear coefficient model '{settings.model}'.")


def ellipticity_ratio(epsilon):
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative.")
    return epsilon / (0.5 + epsilon)
