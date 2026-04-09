import numpy as np
import BC_2D as bc


def get_vx(U, params):
    Nx, Ny = params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    vx = np.zeros((Nx, Ny, 2), dtype=np.float64)

    for j in range(jb, je + 1):
        for i in range(ib, ie + 1):
            # update values for v
            um = 0.5 * (U[i - 1, j] + U[i, j])
            up = 0.5 * (U[i, j] + U[i + 1, j])
            vm = um / (1.0 + um)
            vp = up / (1.0 + up)
            vx[i - 1, j - 1, 0] = vm
            vx[i - 1, j - 1, 1] = vp

    return vx


def get_vy(U, params):
    Nx, Ny = params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    vy = np.zeros((Nx, Ny, 2), dtype=np.float64)

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            # update values for v
            um = 0.5 * (U[i, j - 1] + U[i, j])
            up = 0.5 * (U[i, j] + U[i, j + 1])
            vm = um / (1.0 + um)
            vp = up / (1.0 + up)
            vy[i - 1, j - 1, 0] = vm
            vy[i - 1, j - 1, 1] = vp

    return vy


def x_sweep(U, vx, rx, config):
    Nx, Ny, omega = config.Nx, config.Ny, config.omega
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je

    U_old = U.copy()
    U_new = U.copy()

    for j in range(jb, je+1):  # y-loop: no y-dependence for x-sweep
        for i in range(ib, ie+1):  # x-loop: 1D hyperbolic update rule
            # get values for v
            vm = vx[i-1, j-1, 0]
            vp = vx[i-1, j-1, 1]

            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i - 1, j] + ((-1 + rx * vm) * U[i - 1, j])
                           + ((1 - rx * (vm + vp)) * U[i, j] + (rx * vp) * U[i + 1, j]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, config)
    return U_new


def y_sweep(U, vy, ry, config):
    Nx, Ny, omega = config.Nx, config.Ny, config.omega
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je

    U_old = U.copy()
    U_new = U.copy()

    for i in range(ib, ie+1):  # x-loop: no x-dependence for y-sweep
        for j in range(jb, je+1):  # y-loop: 1D hyperbolic update rule
            # get values for v
            vm = vy[i-1, j-1, 0]
            vp = vy[i-1, j-1, 1]

            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i, j - 1] + ((-1 + ry * vm) * U[i, j - 1])
                           + ((1 - ry * (vm + vp)) * U[i, j] + (ry * vp) * U[i, j + 1]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, config)
    return U_new
