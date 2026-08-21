"""Directional sweep kernels shared by nonlinear splitting methods."""

import boundary_conditions as bc


def x_sweep(U, vx, rx, config):
    omega = config.omega
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    U_old = U.copy()
    U_new = U.copy()

    for j in range(jb, je + 1):
        for i in range(ib, ie + 1):
            vm = vx[i - ib, j - jb, 0]
            vp = vx[i - ib, j - jb, 1]
            U_new[i, j] = (
                U_new[i - 1, j]
                + (-1 + rx * vm) * U[i - 1, j]
                + (1 - rx * (vm + vp)) * U[i, j]
                + rx * vp * U[i + 1, j]
            )
            U_new[i, j] = U_old[i, j] + omega * (
                U_new[i, j] - U_old[i, j]
            )

    bc.update_BCs(U_new, config)
    return U_new


def y_sweep(U, vy, ry, config):
    omega = config.omega
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    U_old = U.copy()
    U_new = U.copy()

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            vm = vy[i - ib, j - jb, 0]
            vp = vy[i - ib, j - jb, 1]
            U_new[i, j] = (
                U_new[i, j - 1]
                + (-1 + ry * vm) * U[i, j - 1]
                + (1 - ry * (vm + vp)) * U[i, j]
                + ry * vp * U[i, j + 1]
            )
            U_new[i, j] = U_old[i, j] + omega * (
                U_new[i, j] - U_old[i, j]
            )

    bc.update_BCs(U_new, config)
    return U_new
