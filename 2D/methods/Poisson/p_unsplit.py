import boundary_conditions as bc
from methods.poisson.poisson_helpers import *


def get_params(config):
    # get parameters from config class
    cfl, dx, dy = config.cfl, config.dx, config.dy

    # pseudo-time step
    dt_factor = 1.0
    dt_default = cfl / (2 * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt_default = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    dt = dt_factor * dt_default

    params = {
        "dt": dt,
    }
    return params


def _update_point(U, U_old, i, j, dt, omega, f_all, config):
    candidate = U[i, j] + dt * (
        laplacian_2D(U, i, j, config, config.laplacian_stencil)
        - f_all[i, j]
    )
    U[i, j] = U_old[i, j] + omega * (candidate - U_old[i, j])


def full_sweep(U, params, boundaries, config):
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    omega, f_all = config.omega, config.f_all
    dt = params["dt"]

    U_old = U.copy()
    gs_ordering = config.p_unsplit_config.gs_ordering

    if gs_ordering == "lexicographic":
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                _update_point(U, U_old, i, j, dt, omega, f_all, config)
    elif gs_ordering == "red_black":
        for color in (0, 1):
            for i in range(ib, ie + 1):
                for j in range(jb, je + 1):
                    if (i + j) % 2 == color:
                        _update_point(
                            U, U_old, i, j, dt, omega, f_all, config
                        )
    else:
        raise ValueError(f"Unknown p_unsplit gs_ordering='{gs_ordering}'.")

    # update BCs
    bc.apply_bcs_poisson_2d(U, config, boundaries)

    return U
