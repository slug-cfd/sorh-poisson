"""Smoother polar-star Poisson test.

This reuses the polar-star exact solution and RHS with a larger epsilon so
the transition layers are better resolved on moderate grids such as 128x128.
"""

from .polar_star import exact, rhs
from .types import PoissonTestProblem


PROBLEM = PoissonTestProblem(
    name="polar_star_smooth",
    domain={
        "xa": 0.0,
        "xb": 0.5,
        "ya": 0.5,
        "yb": 1.0,
    },
    constants={
        "epsilon": 0.015,
    },
    rhs=rhs,
    exact=exact,
)
