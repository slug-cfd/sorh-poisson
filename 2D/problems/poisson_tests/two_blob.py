import math

from .types import PoissonTestProblem


def A(x, y, R):
    c = 3.0 * (R ** 2)
    return (((x - (5.0 * R)) ** 2) + (y ** 2)) / c


def B(x, y, R):
    c = 3.0 * (R ** 2)
    return (((x + (5.0 * R)) ** 2) + (y ** 2)) / c


def rho(x, y, rho0, R):
    c = rho0 / 3.0
    a = A(x, y, R)
    b = B(x, y, R)

    left = (2 - a) * ((1 + a) ** (-5.0 / 2.0))
    right = (2 - b) * ((1 + b) ** (-5.0 / 2.0))

    return c * (left + right)


def rhs(x, y, config):
    return rho(x, y, config.rho0, config.R)


def exact(x, y, config):
    c = -4.0 * math.pi * config.G * config.rho0 * (config.R ** 2)
    left = (1 + A(x, y, config.R)) ** (-0.5)
    right = (1 + B(x, y, config.R)) ** (-0.5)
    return c * (left + right)


# Legacy callable signatures used by older config files.
def f(x, y, rho0, R):
    return rho(x, y, rho0, R)


def phi_exact(x, y, rho0, R, G):
    c = -4.0 * math.pi * G * rho0 * (R ** 2)
    left = (1 + A(x, y, R)) ** (-0.5)
    right = (1 + B(x, y, R)) ** (-0.5)
    return c * (left + right)


PROBLEM = PoissonTestProblem(
    name="two_blob",
    domain={
        "xa": -20.0,
        "xb": 20.0,
        "ya": -20.0,
        "yb": 20.0,
    },
    constants={
        "R": 1.0,
        "rho0": 3.0,
        "G": 0.25 / math.pi,
    },
    rhs=rhs,
    exact=exact,
)
