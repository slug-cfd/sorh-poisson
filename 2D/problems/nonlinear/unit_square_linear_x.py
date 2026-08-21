"""Unit-square nonlinear diffusion with u=x Dirichlet data."""

import numpy as np

from .types import NonlinearCaseDefinition


def zero_rhs(x, y, _context=None):
    shape = np.broadcast_shapes(np.shape(x), np.shape(y))
    return np.zeros(shape, dtype=float)


def linear_in_x_boundary(x, y, _context=None):
    x_values, y_values = np.broadcast_arrays(x, y)
    return np.asarray(x_values, dtype=float) + np.zeros_like(y_values, dtype=float)


CASE = NonlinearCaseDefinition(
    name="unit_square_linear_x",
    display_name="Unit-Square Nonlinear Diffusion",
    domain={"xa": 0.0, "xb": 1.0, "ya": 0.0, "yb": 1.0},
    rhs=zero_rhs,
    boundary=linear_in_x_boundary,
)
