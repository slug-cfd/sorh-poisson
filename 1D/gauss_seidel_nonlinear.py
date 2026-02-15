# finite difference for the semi-implicit Gauss-Seidel method
# solving a linear pde with constant v

import numpy as np
import matplotlib.pyplot as plt

def solve(N=32, cfl=0.8, iter_max=3000):
    dx = 1 / N

    v = 0.5     # initial / maximum

    dt = cfl * ((dx ** 2) / (2*v))
    r = dt / (dx ** 2)  # const. factor for method update rule (excluding dynamic v)

    # a = r
    # b = 1 - (2 * r)
    # c = r

    U = np.zeros(N+1)

    # BCs
    U[0] = 0
    # U[-1] = 1 is better for code performance than U[-1] = 0.
    # if U[-1] = 0 is used, the error check in line 72
    # needs to include the boundary points too.
    U[-1] = 1
    # U_new = U

    # x = np.zeros(N+1)
    x = np.linspace(0, 1, N+1)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)
    # exact = x[1:-1]     # only internal grid pts

    t = 0

    # tmax = 0.4
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    err = np.zeros(N-1)

    while res > threshold and iter_count < iter_max:

        #  store the old solution to measure the steadiness
        Uold = U.copy()

        for i in range(1, N):     # internal grid pts
            # update values for v
            um = 0.5 * (U[i - 1] + U[i])
            up = 0.5 * (U[i] + U[i + 1])
            vm = um / (1 + um)
            vp = up / (1 + up)

            U[i] = (r * vm * U[i - 1]) + ((1 - r * (vm + vp)) * U[i]) + (r * vp * U[i + 1])

            # should update dt since v is non-constant
            v_max = vm if vm > vm else vp
            if np.abs(v_max) > 10e-3:
                dt_temp = cfl * ((dx ** 2) / (2 * v_max))
                if dt_temp < dt:
                    dt = dt_temp


            # U[i] = (r * U[i-1]) + ((1-2*r) * U[i]) + (r * U[i+1])
            # U_new[i] = (a * U[i - 1]) + (b * U[i]) + (c * U[i + 1])   # using a, b, c

        # update BCs
        U[0] = 0
        U[-1] = 1

        # calculate error
        err = U - Uold

        res = np.linalg.norm(err, 2) / np.sqrt(N-1)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    # pointwise error (nonlinear)
    # err_exact = np.abs(x - U)
    err_exact = np.abs((1 - np.log(2)) * x - (U - np.log(1 + U)))

    return U, t, iter_count, res, err_conv, err_exact
