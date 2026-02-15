import numpy as np
import BC_2D as bc

def parabolic(params, phi_exact, get_all_f):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega, iter_max = params.omega, params.iter_max

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, phi_exact=phi_exact)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    rx = dt / (dx ** 2)
    ry = dt / (dy ** 2)
    midpoint = 1.0 - (2.0 * rx) - (2.0 * ry)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:
        #  store the old solution to measure the steadiness
        U_old = U.copy()

        # RED sweep
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                if (i + j) % 2 == 0:
                    U[i, j] = midpoint * U[i, j] \
                                  + rx * (U[i - 1, j] + U[i + 1, j]) \
                                  + ry * (U[i, j - 1] + U[i, j + 1]) - (dt * f_all[i, j])

                # SOR update
                U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # BLACK sweep
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                if (i + j) % 2 == 1:
                    U[i, j] = midpoint * U[i, j] \
                                  + rx * (U[i - 1, j] + U[i + 1, j]) \
                                  + ry * (U[i, j - 1] + U[i, j + 1]) - (dt * f_all[i, j])

                # SOR update
                U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # update BCs
        bc.apply_bcs_poisson_2d(U, params, phi_exact=phi_exact)

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

def x_sweep(U, rx, dt, params, phi_exact, f_all):
    Nx, Ny, omega = params.Nx, params.Ny, params.omega
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    U_old = U.copy()
    U_new = U.copy()

    midpoint = 1.0 - (2.0 * rx)

    for j in range(jb, je + 1):  # y-loop: no y-dependence for x-sweep
        for i in range(ib, ie + 1):  # x-loop: 1D hyperbolic update rule
            # 1D hyperbolic discretization
            new = (rx * U[i-1, j]) + (midpoint * U[i, j]) + (rx * U[i+1, j]) - (dt * f_all[i, j])

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (new - U_old[i, j]))

    bc.apply_bcs_poisson_2d(U_new, params, phi_exact=phi_exact)
    return U_new

def y_sweep(U, ry, dt, params, phi_exact, f_all):
    Nx, Ny, omega = params.Nx, params.Ny, params.omega
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    U_old = U.copy()
    U_new = U.copy()

    midpoint = 1.0 - (2.0 * ry)

    for i in range(ib, ie + 1):  # x-loop: no x-dependence for y-sweep
        for j in range(jb, je + 1):  # y-loop: 1D hyperbolic update rule
            # 1D hyperbolic discretization
            new = (ry * U[i, j-1]) + (midpoint * U[i, j]) + (ry * U[i, j+1]) - (dt * f_all[i, j])

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (new - U_old[i, j]))

    bc.apply_bcs_poisson_2d(U_new, params, phi_exact=phi_exact)
    return U_new

def DS1(params, phi_exact, get_all_f):
    # get parameters as needed
    Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    cfl, iter_max = params.cfl, params.iter_max

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, phi_exact=phi_exact)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:
        U_init = U.copy()

        # use a single dt per timestep
        dt = cfl * np.min([dx, dy])
        rx = dt / dx
        ry = dt / dy

        # 1. Solve u_tx = vu_xx: U^n -> U^(n+1/2)
        U = x_sweep(U, rx, dt, params, phi_exact, f_all)

        # 2. Solve u_ty = vu_yy: U^(n+1/2) -> U^(n+1)
        U = y_sweep(U, ry, dt, params, phi_exact, f_all)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    return U, t, iter_count, res, err_conv

def DS2(params, phi_exact, get_all_f):
    # get parameters as needed
    Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    cfl, iter_max = params.cfl, params.iter_max

    # error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, phi_exact=phi_exact)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    ### Step 1: Initial 1/4-step ###
    dt = 0.5 * cfl * dx
    rx = dt / dx

    # initial x-sweep: U^n -> U^(n+1/4)
    U = x_sweep(U, rx, dt, params, phi_exact, f_all)
    t += dt

    ### Step 2: main iteration loop (full time steps) ###
    while res > threshold and iter_count < iter_max:
        U_init = U.copy()

        dt = cfl * np.min([dx, dy])
        ry = dt / dy
        rx = dt / dx

        # full y-sweep: U^(n+1/4) -> U^(n+3/4)
        U = y_sweep(U, ry, dt, params, phi_exact, f_all)

        # full x-sweep: U^(n+3/4) -> U^(n+5/4)
        U = x_sweep(U, rx, dt, params, phi_exact, f_all)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    ### Step 3: Final 3/4-step ###
    dt = cfl * np.min([dx, dy])

    ry = dt / dy
    rx = 0.5 * dt / dx  # half-step

    # final y-sweep: U^(n+1/4) -> U^(n+3/4)
    U = y_sweep(U, ry, dt, params, phi_exact, f_all)

    # final x-sweep: U^(n+3/4) -> U^(n+1)
    U = x_sweep(U, rx, dt, params, phi_exact, f_all)

    # since dt is different between step 1 and step 3, we
    # estimate the total dt value between both iterations as
    # 0.5 * (dt_init + dt_final)
    t += 0.5 * dt

    # extra iteration to account for step 1 and step 3
    iter_count += 1

    return U, t, iter_count, res, err_conv

def unsplit_update(U, params, update_coeffs, phi_exact, f_all):
    U_old = U.copy()
    U_new = U.copy()

    Nx, Ny, dx, dy, omega = params.Nx, params.Ny, params.dx, params.dy, params.omega
    rx, ry, midpoint, c1, c2 = update_coeffs
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    # only update interior cells: [ib, ie]
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            U_new[i, j] = U[i, j] + c1 * (dy * U_new[i - 1, j] + dx * U_new[i, j - 1]
                                          + (-midpoint * U[i, j]) + ((rx - dy) * U[i - 1, j])
                                          + ((ry - dx) * U[i, j - 1]) + (rx * U[i + 1, j])
                                          + (ry * U[i, j + 1]) - (c2 * f_all[i, j]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.apply_bcs_poisson_2d(U_new, params, phi_exact=phi_exact)
    return U_new

def unsplit(params, phi_exact, get_all_f):
    #  error array (one value per iteration)
    err_conv = np.zeros(params.iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, phi_exact=phi_exact)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    # these parameters are always constant, so pre-calculate them
    dt = cfl * np.min([dx, dy])  # |v| = 1
    rx = dt * dy / dx
    ry = dt * dx / dy

    midpoint = (2.0 * rx) + (2.0 * ry)
    c1 = 1.0 / (dx + dy)
    c2 = dt * dx * dy

    update_coeffs = (rx, ry, midpoint, c1, c2)

    while res > threshold and iter_count < params.iter_max:
        U_init = U.copy()

        # update U^n -> U^n+1
        U = unsplit_update(U, params, update_coeffs, phi_exact, f_all)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    return U, t, iter_count, res, err_conv

