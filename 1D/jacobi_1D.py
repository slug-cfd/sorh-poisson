### jacobi_1D.py ###
# finite difference for the explicit Jacobi method
# options for linear and nonlinear v(u)

import numpy as np

def solve(N=32, cfl=0.8, iter_max=10000, linear=True):
    dx = 1.0 / N

    # if linear     -> v(u) = 0.5 (const)
    # if nonlinear  -> v(u) varies, but v(u) = 0.5 can be used for dt calculation
    v = 0.5

    dt = cfl * ((dx ** 2) / (2*v))
    r = v * dt / (dx ** 2)  # const. factor for method update rule

    # a = r
    # b = 1 - (2 * r)
    # c = r

    U = np.zeros(N+1)

    # BCs
    U[0] = 0
    U[-1] = 1
    U_new = U

    x = np.linspace(0, 1, N+1)

    # error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:

        #  store the old solution to measure the steadiness
        Uold = U.copy()

        if linear:
            for i in range(1, N):     # internal grid pts
                U_new[i] = (r * U[i-1]) + ((1-2*r) * U[i]) + (r * U[i+1])
                # U_new[i] = (a * U[i - 1]) + (b * U[i]) + (c * U[i + 1])   # using a, b, c

        else:
            for i in range(1, N):  # internal grid pts
                # update values for v
                um = 0.5 * (U[i - 1] + U[i])
                up = 0.5 * (U[i] + U[i + 1])
                vm = um / (1 + um)
                vp = up / (1 + up)

                U_new[i] = (r * vm * U[i - 1]) + ((1 - r * (vm + vp)) * U[i]) + (r * vp * U[i + 1])

                # should update dt since v is non-constant
                v_max = vm if vm > vm else vp
                if np.abs(v_max) > 10e-3:
                    dt_temp = cfl * ((dx ** 2) / (2 * v_max))
                    if dt_temp < dt:
                        dt = dt_temp

        # update BCs
        U_new[0] = 0
        U_new[-1] = 1
        U = U_new.copy()

        # calculate error
        err = U - Uold

        res = np.linalg.norm(err, 2) / np.sqrt(N-1)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1


    # get pointwise error
    if linear:  # linear -> exact solution = x
        err_exact = np.abs(x - U)
    else:       # nonlinear
        err_exact = np.abs((1-np.log(2))*x - (U - np.log(1+U)))


    return U, t, iter_count, res, err_conv, err_exact
