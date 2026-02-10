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


def x_sweep(U, vx, rx, x, Nx, Ny, bc_type, omega):
    U_old = U.copy()
    U_new = U.copy()

    for j in range(1, Ny):  # y-loop: no y-dependence for x-sweep
        for i in range(1, Nx):  # x-loop: 1D hyperbolic update rule
            # get values for v
            vm = vx[i-1, j-1, 0]
            vp = vx[i-1, j-1, 1]

            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i - 1, j] + ((-1 + rx * vm) * U[i - 1, j])
                           + ((1 - rx * (vm + vp)) * U[i, j] + (rx * vp) * U[i + 1, j]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, x, bc_type)
    return U_new


def y_sweep(U, vy, ry, x, Nx, Ny, bc_type, omega):
    U_old = U.copy()
    U_new = U.copy()

    for i in range(1, Nx):  # x-loop: no x-dependence for y-sweep
        for j in range(1, Ny):  # y-loop: 1D hyperbolic update rule
            # get values for v
            vm = vy[i-1, j-1, 0]
            vp = vy[i-1, j-1, 1]

            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i, j - 1] + ((-1 + ry * vm) * U[i, j - 1])
                           + ((1 - ry * (vm + vp)) * U[i, j] + (ry * vp) * U[i, j + 1]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, x, bc_type)
    return U_new


def DS1(U, x, dx, dy, Nx, Ny, cfl, bc_type, omega, iter_max):
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
        rx = dt / dx
        ry = dt / dy

        # 1. Solve u_tx = vu_xx: U^n -> U^(n+1/2)
        U = x_sweep(U, vx, rx, x, Nx, Ny, bc_type, omega)

        # 2. Solve u_ty = vu_yy: U^(n+1/2) -> U^(n+1)
        U = y_sweep(U, vy, ry, x, Nx, Ny, bc_type, omega)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[1:Nx, 1:Ny], ord="fro") / np.sqrt((Nx - 1) * (Ny - 1))
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    return U, t, iter_count, res, err_conv


def DS2(U, x, dx, dy, Nx, Ny, cfl, bc_type, omega, iter_max):
    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2


    ### Step 1: Initial 1/4-step ###
    vx = get_vx(U, Nx, Ny)

    dt = 0.5 * cfl * (dx / np.abs(vx).max())
    rx = dt / dx

    # initial x-sweep: U^n -> U^(n+1/4)
    U = x_sweep(U, vx, rx, x, Nx, Ny, bc_type, omega)
    t += dt


    ### Step 2: main iteration loop (full time steps) ###
    while res > threshold and iter_count < iter_max:
        U_init = U.copy()

        vx = get_vx(U, Nx, Ny)
        vy = get_vy(U, Nx, Ny)

        dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
        ry = dt / dy
        rx = dt / dx

        # full y-sweep: U^(n+1/4) -> U^(n+3/4)
        U = y_sweep(U, vy, ry, x, Nx, Ny, bc_type, omega)

        # full x-sweep: U^(n+3/4) -> U^(n+5/4)
        U = x_sweep(U, vx, rx, x, Nx, Ny, bc_type, omega)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[1:Nx, 1:Ny], ord="fro") / np.sqrt((Nx - 1) * (Ny - 1))
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1


    ### Step 3: Final 3/4-step ###
    vx = get_vx(U, Nx, Ny)
    vy = get_vy(U, Nx, Ny)

    dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])

    ry = dt / dy
    rx = 0.5 * dt / dx  # half-step

    # final y-sweep: U^(n+1/4) -> U^(n+3/4)
    U = y_sweep(U, vy, ry, x, Nx, Ny, bc_type, omega)

    # final x-sweep: U^(n+3/4) -> U^(n+1)
    U = x_sweep(U, vx, rx, x, Nx, Ny, bc_type, omega)

    # since dt is different between step 1 and step 3, we
    # estimate the total dt value between both iterations as
    # 0.5 * (dt_init + dt_final)
    t += 0.5 * dt

    # extra iteration to account for step 1 and step 3
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

    ds_order = 1

    match ds_order:
        case 1:     # 1st-order dimensional splitting (DS1)
            U, t, iter_count, res, err_conv = DS1(U, x, dx, dy, Nx, Ny, cfl, bc_type, omega, iter_max)
        case 2:     # 2nd-order dimensional splitting (DS2)
            U, t, iter_count, res, err_conv = DS2(U, x, dx, dy, Nx, Ny, cfl, bc_type, omega, iter_max)
        case _:
            return -1

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res, err_conv
