"""Minimal linear SORp sweep using the centered fourth-order Laplacian."""

from problem_helpers import fourth_order_laplacian_dimensionless


def get_params(config):
    v = config.linear_v
    dt = config.dt_factor * config.cfl * (config.dx**2) / (2.0 * v)
    r = v * dt / (config.dx**2)
    return {"v": v, "dt": dt, "r": r}


def full_sweep(U, params, config):
    r = params["r"]
    omega = config.omega
    U_new = U.copy()

    for i in range(config.ib, config.ie + 1):
        laplacian_fourth = fourth_order_laplacian_dimensionless(U_new, i)
        U_new[i] += omega * r * laplacian_fourth

    return U_new
