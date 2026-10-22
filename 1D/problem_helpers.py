import numpy as np


MIXED_MODE_PROFILE_OPTIONS = {"low", "mid", "packet", "mixed"}


def mixed_mode_components(x):
    """Return the canonical components of the linear error-propagation profile."""
    x = np.asarray(x)
    node_numbers = np.arange(x.size)

    components = {
        "low": np.sin(np.pi * x),
        "mid": 0.2 * np.sin(16.0 * np.pi * x),
        "packet": (
            0.5
            * np.exp(-((x - 0.28) / 0.035) ** 2)
            * (-1.0) ** node_numbers
        ),
    }

    # The experiment uses homogeneous Dirichlet data.  Enforce those values
    # exactly instead of retaining roundoff or the Gaussian's tiny endpoint tails.
    for values in components.values():
        values[[0, -1]] = 0.0

    return components


def mixed_mode_profile(x, profile):
    components = mixed_mode_components(x)
    if profile == "mixed":
        return sum(components.values(), start=np.zeros_like(np.asarray(x)))
    return components[profile].copy()


def fourth_order_laplacian_dimensionless(U, i):
    return (
        -U[i - 2]
        + 16.0 * U[i - 1]
        - 30.0 * U[i]
        + 16.0 * U[i + 1]
        - U[i + 2]
    ) / 12.0


def exact_diagnostic(U, config):
    if config.problem in {"linear", "linear_mixed_mode"}:
        return U - config.U_exact

    return (U - np.log1p(U)) - (1.0 - np.log(2.0)) * config.x


def equation_residual(U, config):
    """Return the discrete steady-equation residual when one is defined."""
    if config.problem != "linear_mixed_mode":
        return None

    result = np.zeros_like(U)
    dx2 = config.dx**2

    if config.uses_fourth_order_laplacian:
        for i in range(config.ib, config.ie + 1):
            result[i] = -fourth_order_laplacian_dimensionless(U, i) / dx2
    else:
        for i in range(config.ib, config.ie + 1):
            result[i] = -(U[i - 1] - 2.0 * U[i] + U[i + 1]) / dx2

    return result
