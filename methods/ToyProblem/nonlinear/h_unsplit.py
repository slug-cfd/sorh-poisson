import numpy as np
from methods.ToyProblem.nonlinear import nl_helpers as nlh
import BC_2D as bc


def get_params(config):
    params = {}
    return params


def full_sweep(U, params, config):
    # get parameters from config class and param dict
    omega, ib, ie, jb, je = config.omega, config.ib, config.ie, config.jb, config.je
    cfl, dx, dy = config.cfl, config.dx, config.dy

    vx = nlh.get_vx(U, config)
    vy = nlh.get_vy(U, config)

    # use a single dt per timestep
    dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
    rx = dt * dy / dx
    ry = dt * dx / dy

    U_old = U.copy()
    U_new = U.copy()

    # only update interior cells: [ib, ie]
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            # get values for v
            vxm = vx[i - 1, j - 1, 0]
            vxp = vx[i - 1, j - 1, 1]
            vym = vy[i - 1, j - 1, 0]
            vyp = vy[i - 1, j - 1, 1]

            rxvp = rx * vxp
            rxvm = rx * vxm
            ryvp = ry * vyp
            ryvm = ry * vym

            midpoint = rxvp + rxvm + ryvp + ryvm
            c = 1.0 / (dx + dy)

            U_new[i, j] = U[i, j] + c * (dy * U_new[i - 1, j] + dx * U_new[i, j - 1]
                                         + (-midpoint * U[i, j]) + ((rxvm - dy) * U[i - 1, j])
                                         + ((ryvm - dx) * U[i, j - 1]) + (rxvp * U[i + 1, j])
                                         + (ryvp * U[i, j + 1]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, config)
    return U_new
