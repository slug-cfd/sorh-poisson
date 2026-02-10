# jacobi_linear_2D.py
# finite difference for the explicit Jacobi method (2D)
# solving a linear parabolic pde with constant v

import numpy as np

def solve(Nx=16, Ny=16, cfl=0.8, iter_max=10000, omega=1.0):
    dx = 1 / Nx
    dy = 1 / Ny

    v = 0.5     # linear -> v(u) = const (universal speed)

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * v * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    rx = v * dt / (dx ** 2)
    ry = v * dt / (dy ** 2)

    grid = np.zeros((3, 3))

    midpoint = 1.0 - (2.0 * rx) - (2.0 * ry)
    grid[1, 1] = midpoint

    # middle row
    grid[1, 0] = rx
    grid[1, 2] = rx

    # middle column
    grid[0, 1] = ry
    grid[2, 1] = ry

    x = np.linspace(0, 1, Nx+1)
    y = np.linspace(0, 1, Ny+1)

    U = np.zeros((Nx+1, Ny+1))

    # BCs: wrap-around
    # U[0, 0] = 0.0
    # U[:, -1] = 1.0
    # U[-1, :] = 1.0

    # BC: U(x, y) = x
    # x=0 -> u=0
    U[0, :] = 0.0
    # x=1 -> u=1
    U[-1, :] = 1.0
    # y=0 -> u=x
    U[:, 0] = x
    # y=1 -> u=x
    U[:, -1] = x

    # a = 1.0
    # b = 2.0
    # c = 1.0
    # d = 2.0
    #
    # # BC: spatial oscillations
    # U[:, 0] = a * np.sin(b * np.pi * x)
    # U[:, -1] = 0.0
    # U[0, :] = c * np.sin(d * np.pi * x)
    # U[-1, :] = 0.0

    # BC: combination
    # U[:, 0] = a * np.sin(b * np.pi * x)
    # U[:, -1] = 0.0
    # U[0, :] = -np.sqrt(x)
    # U[-1, :] = np.sqrt(x)

    # print("initial U")
    # print(U)

    # U_new = U.copy()

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)
    # exact = x[1:-1]     # only internal grid pts

    t = 0

    # tmax = 0.4
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    # err = np.zeros((Nx-1, Ny-1))

    while res > threshold and iter_count < iter_max:

        #  store the old solution to measure the steadiness
        Uold = U.copy()

        # Lexicographic GS
        # for i in range(1, Nx):     # internal grid pts
        #     for j in range(1, Ny):
        #         # midpoint + middle row + middle column
        #         # GS update: no U_new
        #         U[i, j] = (grid[1, 1] * U[i, j]
        #                        + grid[1, 0] * U[i+1, j] + grid[1, 2] * U[i-1, j]
        #                        + grid[0, 1] * U[i, j-1] + grid[2, 1] * U[i, j+1])

        # RED sweep
        for i in range(1, Nx):
            for j in range(1, Ny):
                if (i + j) % 2 == 0:
                    U[i, j] = midpoint * U[i, j] \
                              + rx * (U[i + 1, j] + U[i - 1, j]) \
                              + ry * (U[i, j + 1] + U[i, j - 1])

                # SOR update
                U[i, j] = Uold[i, j] + (omega * (U[i, j] - Uold[i, j]))


        # BLACK sweep
        for i in range(1, Nx):
            for j in range(1, Ny):
                if (i + j) % 2 == 1:
                    U[i, j] = midpoint * U[i, j] \
                              + rx * (U[i + 1, j] + U[i - 1, j]) \
                              + ry * (U[i, j + 1] + U[i, j - 1])

                # SOR update
                U[i, j] = Uold[i, j] + (omega * (U[i, j] - Uold[i, j]))


        # update BCs

        # U_new[0, 0] = 0.0
        # U_new[:, -1] = 1.0
        # U_new[-1, :] = 1.0

        # x=0 -> u=0
        U[0, :] = 0.0
        # x=1 -> u=1
        U[-1, :] = 1.0
        # y=0 -> u=x
        U[:, 0] = x
        # y=1 -> u=x
        U[:, -1] = x

        # U[:, 0] = a * np.sin(b * np.pi * x)
        # U[:, -1] = 0.0
        # U[0, :] = c * np.sin(d * np.pi * x)
        # U[-1, :] = 0.0

        # U[:, 0] = a * np.sin(b * np.pi * x)
        # U[:, -1] = 0.0
        # U[0, :] = -np.sqrt(x)
        # U[-1, :] = np.sqrt(x)

        # U = U_new.copy()

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
    return U, t, iter_count, res
