# sor_hyperbolic_2D.py
# finite difference for the semi-implicit SOR method (2D)
# solving a hyperbolic PDE, nonlinear, various BC types

import numpy as np
import BC_2D as bc


def get_vx(U, Nx, Ny):
    vx = np.zeros((Nx-1, Ny-1, 2), dtype=np.float64)

    for j in range(1, Ny):  # y-loop: no y-dependence for x-sweep
        for i in range(1, Nx):  # x-loop: 1D hyperbolic update rule
            # update values for v
            um = 0.5 * (U[i - 1, j] + U[i, j])
            up = 0.5 * (U[i, j] + U[i + 1, j])
            vm = um / (1.0 + um)
            vp = up / (1.0 + up)
            vx[i-1, j-1, 0] = vm
            vx[i-1, j-1, 1] = vp

    return vx


def get_vy(U, Nx, Ny):
    vy = np.zeros((Nx - 1, Ny - 1, 2), dtype=np.float64)

    for i in range(1, Nx):  # x-loop: no x-dependence for y-sweep
        for j in range(1, Ny):  # y-loop: 1D hyperbolic update rule
            # update values for v
            um = 0.5 * (U[i, j - 1] + U[i, j])
            up = 0.5 * (U[i, j] + U[i, j + 1])
            vm = um / (1.0 + um)
            vp = up / (1.0 + up)
            vy[i-1, j-1, 0] = vm
            vy[i-1, j-1, 1] = vp

    return vy


def unsplit_update(U, dx, dy, vx, vy, rx, ry, x, Nx, Ny, bc_type, omega):
    U_old = U.copy()
    U_new = U.copy()

    # rx = dt*dy / dx
    # ry = dt*dx / dy

    for i in range(1, Nx):  # x-loop: no x-dependence for y-sweep
        for j in range(1, Ny):  # y-loop: 1D hyperbolic update rule
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

    bc.update_BCs(U_new, x, bc_type)
    return U_new


def unsplit_solve(U, x, dx, dy, Nx, Ny, cfl, bc_type, omega, iter_max):
    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:
        U_init = U.copy()

        vx = get_vx(U, Nx, Ny)
        vy = get_vy(U, Nx, Ny)

        # use a single dt per timestep
        dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
        rx = dt*dy / dx
        ry = dt*dx / dy

        # update U^n -> U^n+1
        U = unsplit_update(U, dx, dy, vx, vy, rx, ry, x, Nx, Ny, bc_type, omega)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[1:Nx, 1:Ny], ord="fro") / np.sqrt((Nx - 1) * (Ny - 1))
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    return U, t, iter_count, res, err_conv


def solve(Nx=16, Ny=16, cfl=0.8, iter_max=10000, omega=1.0):
    dx = 1.0 / Nx
    dy = 1.0 / Ny

    x = np.linspace(0, 1, Nx + 1)
    # y = np.linspace(0, 1, Ny+1)

    U = np.zeros((Nx + 1, Ny + 1))

    # set initial BCs
    bc_type = 0
    bc.update_BCs(U, x, bc_type)

    U, t, iter_count, res, err_conv = unsplit_solve(U, x, dx, dy, Nx, Ny,
                                                    cfl, bc_type, omega, iter_max)

    return U, t, iter_count, res, err_conv
