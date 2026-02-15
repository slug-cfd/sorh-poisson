# SOR overrelaxing parameter omega
# omega = 1.75  # 2-2*pi*dx;

# U_new(i) = c*U(i-1) + a*U(i) + b*U(i+1);
# U(i) = Uold(i) + w*(U_new(i)-Uold(i));

# finite difference for the semi-implicit SOR method
# solving a hyperbolic pde

import numpy as np
import matplotlib.pyplot as plt

def solve(N=32, cfl=0.8, iter_max=3000, omega=1.0):
    dx = 1 / N

    v = 0.5     # initial / maximum

    # dt = cfl * ((dx ** 2) / (2*v))
    dt = cfl * (dx / v)

    # r = dt / (dx ** 2)  # const. factor for method update rule (excluding dynamic v)
    r = dt / dx         # new dt condition for hyperbolic system

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
    U_new = U.copy()

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
    res_limit = 1e4

    while threshold < res < res_limit and iter_count < iter_max:

        #  store the old solution to measure the steadiness
        Uold = U.copy()
        U_new = U.copy()

        for i in range(1, N):     # internal grid pts
            # update values for v
            um = 0.5 * (U[i - 1] + U[i])
            up = 0.5 * (U[i] + U[i + 1])
            vm = um / (1 + um)
            vp = up / (1 + up)

            # original nonlinear mode
            # U[i] = (r * vm * U[i - 1]) + ((1 - r * (vm + vp)) * U[i]) + (r * vp * U[i + 1])

            # new hyperbolic formulation
            U_new[i] = (U_new[i - 1] + ((-1 + r * vm) * U[i - 1])
                        + ((1 - r * (vm + vp)) * U[i] + (r * vp) * U[i + 1]))

            # should update dt since v is non-constant
            v_max = vm if vm > vp else vp
            if np.abs(v_max) > 10e-3:
                dt_temp = cfl * (dx / v_max)
                if dt_temp < dt:
                    dt = dt_temp

            # linear modes
            # U[i] = (r * U[i-1]) + ((1-2*r) * U[i]) + (r * U[i+1])
            # U_new[i] = (a * U[i - 1]) + (b * U[i]) + (c * U[i + 1])   # using a, b, c

            # SOR update
            U_new[i] = Uold[i] + (omega * (U_new[i] - Uold[i]))

        # update BCs
        U_new[0] = 0.0
        U_new[-1] = 1.0

        U[:] = U_new

        # calculate error
        err = U - Uold

        res = np.linalg.norm(err, 2) / np.sqrt(N-1)
        # print(f"res (iter {iter_count}):", res)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    # pointwise error (nonlinear)
    # err_exact = np.abs(x - U)
    err_exact = np.abs((1 - np.log(2)) * x - (U - np.log(1 + U)))

    # safeguard against weird results
    # solver may halt at low iterations for unstable cases
    if res > threshold:
        iter_count = iter_max

    return U, t, iter_count, res, err_conv, err_exact
