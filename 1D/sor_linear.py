# SOR overrelaxing parameter omega
# omega = 1.75  # 2-2*pi*dx;

# U_new(i) = c*U(i-1) + a*U(i) + b*U(i+1);
# U(i) = Uold(i) + w*(U_new(i)-Uold(i));

# finite difference for the semi-implicit SOR method
# solving a linear pde with constant v

import numpy as np
import matplotlib.pyplot as plt

def solve(N=32, cfl=0.8, iter_max=3000, omega=1.0):
    dx = 1 / N

    v = 0.5     # linear -> v(u) = const
    dt = cfl * ((dx ** 2) / (2*v))
    r = v * dt / (dx ** 2)  # const. factor for method update rule

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
    U_new = U

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
            U[i] = (r * U[i-1]) + ((1-2*r) * U[i]) + (r * U[i+1])
            # U_new[i] = (a * U[i - 1]) + (b * U[i]) + (c * U[i + 1])   # using a, b, c

            # SOR update
            U[i] = Uold[i] + (omega * (U_new[i] - Uold[i]))

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

    # pointwise error (linear -> exact solution = x)
    err_exact = np.abs(x - U)

    return U, t, iter_count, res, err_conv, err_exact
