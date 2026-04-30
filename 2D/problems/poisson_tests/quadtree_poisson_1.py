import numpy as np

from .types import PoissonTestProblem


def exact(x, y, config):
    """Equation (4.2): u_exact(x, y) = sin(x) + sin(y)."""
    return np.sin(x) + np.sin(y)


def rhs(x, y, config):
    """Equation (4.1): Delta u(x, y) = -(sin(x) + sin(y))."""
    return -(np.sin(x) + np.sin(y))


PROBLEM = PoissonTestProblem(
    name="quadtree_poisson_1",
    domain={
        "xa": -10.0,
        "xb": 10.0,
        "ya": -10.0,
        "yb": 10.0,
    },
    rhs=rhs,
    exact=exact,
)
