"""Smooth globally directional Poisson test.

The exact solution is an elongated Gaussian ridge in rotated coordinates.
It has a clear global directional bias, but the ridge width is broad enough
to be resolved on a 64x64 grid.
"""

import numpy as np

from .types import PoissonTestProblem


DEFAULT_THETA = 0.25 * np.pi
DEFAULT_SIGMA = 0.035
DEFAULT_K = 0.2 * np.pi
DEFAULT_AMPLITUDE = 62.50


def _constants(config):
    return config.problem_instance.constants


def _rotated_coordinates(x, y, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    s = cos_t * x + sin_t * y
    n = -sin_t * x + cos_t * y
    return s, n


def exact(x, y, config):
    constants = _constants(config)
    theta = constants.get("ridge_theta", DEFAULT_THETA)
    sigma = constants.get("sigma", DEFAULT_SIGMA)
    k = constants.get("k", DEFAULT_K)
    amplitude = constants.get("amplitude", DEFAULT_AMPLITUDE)

    s, n = _rotated_coordinates(x, y, theta)
    ridge = np.exp(-((n / sigma) ** 2))
    modulation = 1.0 + amplitude * np.cos(k * s)
    return ridge * modulation


def rhs(x, y, config):
    constants = _constants(config)
    theta = constants.get("ridge_theta", DEFAULT_THETA)
    sigma = constants.get("sigma", DEFAULT_SIGMA)
    k = constants.get("k", DEFAULT_K)
    amplitude = constants.get("amplitude", DEFAULT_AMPLITUDE)

    s, n = _rotated_coordinates(x, y, theta)
    ridge = np.exp(-((n / sigma) ** 2))
    cos_ks = np.cos(k * s)
    modulation = 1.0 + amplitude * cos_ks

    normal_part = ((4.0 * n * n) / (sigma ** 4)) - (2.0 / (sigma ** 2))
    along_part = -amplitude * (k ** 2) * cos_ks
    return ridge * ((normal_part * modulation) + along_part)


PROBLEM = PoissonTestProblem(
    name="directional_ridge",
    domain={
        "xa": -1.0,
        "xb": 1.0,
        "ya": -1.0,
        "yb": 1.0,
    },
    constants={
        "ridge_theta": DEFAULT_THETA,
        "sigma": DEFAULT_SIGMA,
        "k": DEFAULT_K,
        "amplitude": DEFAULT_AMPLITUDE,
    },
    rhs=rhs,
    exact=exact,
)
