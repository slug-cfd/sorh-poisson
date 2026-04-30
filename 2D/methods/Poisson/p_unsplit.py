import BC_2D as bc
from methods.Poisson.poisson_helpers import *


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


def full_sweep(U, params, boundaries, config):
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    omega, f_all = config.omega, config.f_all
    dt = params["dt"]

    U_old = U.copy()

    # Lexicographic GS - simple sequential updates
    # for i in range(ib, ie + 1):
    #     for j in range(jb, je + 1):
    #         U[i, j] = (midpoint * U[i, j]
    #                    + rx * (U[i - 1, j] + U[i + 1, j])
    #                    + ry * (U[i, j - 1] + U[i, j + 1]) - (dt * f_all[i, j]))
    #
    #         # SOR update
    #         U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

    # Red-Black GS - checkerboard update pattern
    # -> helps to remove direction dependence and asymmetry issues

    # # RED sweep
    # for i in range(ib, ie + 1):
    #     for j in range(jb, je + 1):
    #         if (i + j) % 2 == 0:
    #             U[i, j] = midpoint * U[i, j] \
    #                       + rx * (U[i - 1, j] + U[i + 1, j]) \
    #                       + ry * (U[i, j - 1] + U[i, j + 1]) - (dt_mod * f_all[i, j])
    #
    #             # SOR update
    #             U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))
    #
    # # BLACK sweep
    # for i in range(ib, ie + 1):
    #     for j in range(jb, je + 1):
    #         if (i + j) % 2 == 1:
    #             U[i, j] = midpoint * U[i, j] \
    #                       + rx * (U[i - 1, j] + U[i + 1, j]) \
    #                       + ry * (U[i, j - 1] + U[i, j + 1]) - (dt_mod * f_all[i, j])
    #
    #             # SOR update
    #             U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

    # Red-Black GS - checkerboard update pattern
    # -> helps to remove direction dependence and asymmetry issues
    # sweep over INTERNAL indices only: [ib, ie]

    # RED sweep
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 0:
                U[i, j] = U[i, j] + (dt * (centered_2D_laplacian(U, i, j, config) - f_all[i, j]))

                # SOR update
                U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

    # BLACK sweep
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 1:
                U[i, j] = U[i, j] + (dt * (centered_2D_laplacian(U, i, j, config) - f_all[i, j]))

                # SOR update
                U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

    # update BCs
    bc.apply_bcs_poisson_2d(U, config, boundaries)

    return U
