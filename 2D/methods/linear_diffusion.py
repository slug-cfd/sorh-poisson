"""Reference iterative methods for constant-coefficient linear diffusion."""

import numpy as np
import boundary_conditions as bc

def jacobi_sweep(U, params, coeffs):
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    rx, ry, midpoint = coeffs

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            U[i, j] = midpoint * U[i, j] \
                      + rx * (U[i + 1, j] + U[i - 1, j]) \
                      + ry * (U[i, j + 1] + U[i, j - 1])

def sor_gs_sweep(U, U_old, params, coeffs, sor=True):
    ib, ie, jb, je, omega = params.ib, params.ie, params.jb, params.je, params.omega
    rx, ry, midpoint = coeffs

    # RED sweep
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 0:
                U[i, j] = midpoint * U[i, j] \
                          + rx * (U[i + 1, j] + U[i - 1, j]) \
                          + ry * (U[i, j + 1] + U[i, j - 1])

                # SOR update
                if sor:
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

    # BLACK sweep
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 1:
                U[i, j] = midpoint * U[i, j] \
                          + rx * (U[i + 1, j] + U[i - 1, j]) \
                          + ry * (U[i, j + 1] + U[i, j - 1])

                # SOR update
                if sor:
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

def parabolic(params, iter_method="sor"):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    iter_max = params.iter_max

    v = 0.5     # linear -> const. v

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * v * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    rx = v * dt / (dx ** 2)
    ry = v * dt / (dy ** 2)
    midpoint = 1.0 - (2.0 * rx) - (2.0 * ry)
    coeffs = (rx, ry, midpoint)

    U = params.U0.copy()
    bc.update_BCs(U, params)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:
        #  store the old solution to measure the steadiness
        U_old = U.copy()

        if iter_method == "sor":
            sor_gs_sweep(U, U_old, params, coeffs, sor=True)
        elif iter_method == "gs":
            sor_gs_sweep(U, U_old, params, coeffs, sor=False)
        elif iter_method == "jacobi":
            jacobi_sweep(U, params, coeffs)
        else:
            print("Invalid iteration method")
            return 0

        # update BCs
        bc.update_BCs(U, params)

        # calculate error
        err = U - U_old

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    # pointwise error (linear -> exact solution = x)
    # err_exact = np.abs(x - U)

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res, err_conv
