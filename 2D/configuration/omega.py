"""Optimal-relaxation estimates for 2D solver diagnostics and initialization."""

import numpy as np

from .models import Config
from .runtime import Grid


def compute_omega_opt(settings: Config, grid: Grid) -> float:
    """Return the optimal-omega estimate for the configured method and grid."""
    nx_intervals = grid.Nx + 1
    ny_intervals = grid.Ny + 1
    c_max = (
        nx_intervals * np.cos(np.pi / nx_intervals)
        + ny_intervals * np.cos(np.pi / ny_intervals)
    ) / (nx_intervals + ny_intervals)
    denominator = 1.0 + np.sqrt(max(0.0, 1.0 - c_max**2))

    if settings.run.method.startswith("p_"):
        return 2.0 / (settings.run.cfl * denominator)   # SORp optimal-omega formula
    if settings.run.method.startswith("h_"):
        return 1.0 / denominator    # SORh optimal-omega formula
    return settings.run.omega
