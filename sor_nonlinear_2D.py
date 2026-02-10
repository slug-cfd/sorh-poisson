# sor_nonlinear_2D.py
# finite difference for the semi-implicit SOR method (2D)
# solving a parabolic PDE, nonlinear, various BC types

import numpy as np
import BC_2D as bc

def solve(Nx=16, Ny=16, cfl=0.8, iter_max=10000, omega=1.0):
    dx = 1 / Nx
    dy = 1 / Ny

    v = 0.5     # nonlinear -> maximum value for stability condition

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * v * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    rx = dt / (dx ** 2)
    ry = dt / (dy ** 2)

    x = np.linspace(0, 1, Nx+1)
    # y = np.linspace(0, 1, Ny+1)

    U = np.zeros((Nx + 1, Ny + 1))

    bc_type = 0
    bc.update_BCs(U, x, bc_type)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:

        #  store the old solution to measure the steadiness
        Uold = U.copy()

        # RED sweep
        for i in range(1, Nx):
            for j in range(1, Ny):
                if (i + j) % 2 == 0:
                    # update values for v
                    uxm = 0.5 * (U[i - 1, j] + U[i, j])
                    uxp = 0.5 * (U[i, j] + U[i + 1, j])
                    uym = 0.5 * (U[i, j - 1] + U[i, j])
                    uyp = 0.5 * (U[i, j] + U[i, j + 1])

                    vxm = uxm / (1.0 + uxm)
                    vxp = uxp / (1.0 + uxp)
                    vym = uym / (1.0 + uym)
                    vyp = uyp / (1.0 + uyp)

                    x_rv = rx * (vxp + vxm)
                    y_rv = ry * (vyp + vym)
                    midpoint = 1.0 - x_rv - y_rv

                    U[i, j] = midpoint * U[i, j] \
                                  + rx * (vxm * U[i - 1, j] + vxp * U[i + 1, j]) \
                                  + ry * (vym * U[i, j - 1] + vyp * U[i, j + 1])

                # SOR update
                U[i, j] = Uold[i, j] + (omega * (U[i, j] - Uold[i, j]))

        # BLACK sweep
        for i in range(1, Nx):
            for j in range(1, Ny):
                if (i + j) % 2 == 1:
                    # update values for v
                    uxm = 0.5 * (U[i - 1, j] + U[i, j])
                    uxp = 0.5 * (U[i, j] + U[i + 1, j])
                    uym = 0.5 * (U[i, j - 1] + U[i, j])
                    uyp = 0.5 * (U[i, j] + U[i, j + 1])

                    vxm = uxm / (1.0 + uxm)
                    vxp = uxp / (1.0 + uxp)
                    vym = uym / (1.0 + uym)
                    vyp = uyp / (1.0 + uyp)

                    x_rv = rx * (vxp + vxm)
                    y_rv = ry * (vyp + vym)
                    midpoint = 1.0 - x_rv - y_rv

                    U[i, j] = midpoint * U[i, j] \
                                  + rx * (vxm * U[i - 1, j] + vxp * U[i + 1, j]) \
                                  + ry * (vym * U[i, j - 1] + vyp * U[i, j + 1])

                # SOR update
                U[i, j] = Uold[i, j] + (omega * (U[i, j] - Uold[i, j]))


        bc.update_BCs(U, x, bc_type)

        # calculate error
        err = U - Uold

        res = np.linalg.norm(err[1:Nx, 1:Ny], ord="fro") / np.sqrt((Nx-1)*(Ny-1))
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1


    # pointwise error (linear -> exact solution = x)
    # err_exact = np.abs(x - U)

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res, err_conv
