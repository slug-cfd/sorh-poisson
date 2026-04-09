import BC_2D as bc
from methods.Poisson.poisson_helpers import *

# DS1/2 solvers
# case "DS1":
#     U, t, iter_count, res, err_conv = pm.DS1(params, phi_exact, get_all_f_from_xy)
# case "DS2":
#     U, t, iter_count, res, err_conv = pm.DS2(params, phi_exact, get_all_f_from_xy)


def x_sweep(U, rx, dt, params, f_all, boundaries):
    Nx, Ny, omega = params.Nx, params.Ny, params.omega
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    U_old = U.copy()
    U_new = U.copy()

    midpoint = 1.0 - (2.0 * rx)

    for j in range(jb, je + 1):  # y-loop: no y-dependence for x-sweep
        for i in range(ib, ie + 1):  # x-loop: 1D hyperbolic update rule
            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i - 1, j] + ((rx - 1.0) * U[i - 1, j]) + (midpoint * U[i, j])
                           + (rx * U[i + 1, j]) - (dt * f_all[i, j]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.apply_bcs_poisson_2d(U, params, boundaries)
    return U_new


def y_sweep(U, ry, dt, params, f_all, boundaries):
    Nx, Ny, omega = params.Nx, params.Ny, params.omega
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    U_old = U.copy()
    U_new = U.copy()

    midpoint = 1.0 - (2.0 * ry)

    for i in range(ib, ie + 1):  # x-loop: no x-dependence for y-sweep
        for j in range(jb, je + 1):  # y-loop: 1D hyperbolic update rule
            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i, j - 1] + ((ry - 1.0) * U[i, j - 1]) + (midpoint * U[i, j])
                           + (ry * U[i, j + 1]) - (dt * f_all[i, j]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.apply_bcs_poisson_2d(U, params, boundaries)
    return U_new


def DS1(params, phi_exact, get_all_f):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    norm, iter_max = params.norm, params.iter_max

    # error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7  # error/residual tolerance
    res_max = 1e7  # upper bound to avoid blow-up in unstable cases
    res_diff = 1e2  # difference in norm of the solution between current and previous iteration

    # we need a separate variable for res_exact because we are still
    # using res_diff as the criterion to exit the iteration loop,
    # even though we are returning and plotting res_exact.
    res_exact = 1e4
    res_prev = 0.0

    boundaries = bc.get_bcs_poisson_2d(params, phi_exact)

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, boundaries)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    # dt will be fixed; no nonlinear terms
    dt = cfl * np.min([dx, dy])
    rx = dt / dx ** 2
    ry = dt / dy ** 2

    metric = "exact"

    x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
    X, Y = np.meshgrid(x, y, indexing='ij')

    U_exact = phi_exact(X, Y, rho0, R, G)  # exact solution over full grid

    while res_diff > threshold and iter_count < iter_max and res_exact < res_max:
        U_old = U.copy()

        # 1. Solve u_tx = u_xx - f: U^n -> U^(n+1/2)
        U = x_sweep(U, rx, dt, params, f_all, boundaries)

        # 2. Solve u_ty = u_yy - f: U^(n+1/2) -> U^(n+1)
        U = y_sweep(U, ry, dt, params, f_all, boundaries)

        # difference between laplacian(U) and f
        # this can also be used as the convergence condition, but is more costly
        # l_all = get_all_laplace(U, params)
        # err_diff = l_all - f_all

        # difference between subsequent iterations
        err_diff = U - U_old

        if metric == "exact":
            err_exact = U - U_exact
            res_exact = np.linalg.norm(err_exact[ib:ie + 1, jb:je + 1], ord=norm)
            res_diff = np.abs(res_exact - res_prev)
            res_prev = res_exact
        else:
            # take the norm and average
            res_diff = np.linalg.norm(err_diff[ib:ie + 1, jb:je + 1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_diff /= np.sqrt(Nx * Ny)

        err_conv[iter_count] = res_exact

        #  update clock and iteration count
        t += dt
        iter_count += 1

    return U, t, iter_count, res_exact, err_conv


def DS2(params, phi_exact, get_all_f):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    norm, iter_max = params.norm, params.iter_max

    # error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7  # error/residual tolerance
    res_max = 1e7  # upper bound to avoid blow-up in unstable cases
    res_diff = 1e2  # difference in norm of the solution between current and previous iteration

    # we need a separate variable for res_exact because we are still
    # using res_diff as the criterion to exit the iteration loop,
    # even though we are returning and plotting res_exact.
    res_exact = 1e4
    res_prev = 0.0

    boundaries = bc.get_bcs_poisson_2d(params, phi_exact)

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, boundaries)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    ### Step 1: Initial 1/4-step ###
    dt = 0.5 * cfl * dx
    rx = dt / dx

    # initial x-sweep: U^n -> U^(n+1/4)
    U = x_sweep(U, rx, dt, params, f_all, boundaries)
    t += dt

    ### Step 2: main iteration loop (full time steps) ###
    dt = cfl * np.min([dx, dy])
    ry = dt / dy
    rx = dt / dx

    metric = "exact"

    x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
    X, Y = np.meshgrid(x, y, indexing='ij')

    U_exact = phi_exact(X, Y, rho0, R, G)  # exact solution over full grid

    while res_diff > threshold and iter_count < iter_max and res_exact < res_max:
        U_old = U.copy()

        # full y-sweep: U^(n+1/4) -> U^(n+3/4)
        U = y_sweep(U, ry, dt, params, f_all, boundaries)

        # full x-sweep: U^(n+3/4) -> U^(n+5/4)
        U = x_sweep(U, rx, dt, params, f_all, boundaries)

        # difference between laplacian(U) and f
        # this can also be used as the convergence condition, but is more costly
        # l_all = get_all_laplace(U, params)
        # err_diff = l_all - f_all

        # difference between subsequent iterations
        err_diff = U - U_old

        if metric == "exact":
            err_exact = U - U_exact
            res_exact = np.linalg.norm(err_exact[ib:ie + 1, jb:je + 1], ord=norm)
            res_diff = np.abs(res_exact - res_prev)
            res_prev = res_exact
        else:
            # take the norm and average
            res_diff = np.linalg.norm(err_diff[ib:ie + 1, jb:je + 1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_diff /= np.sqrt(Nx * Ny)

        err_conv[iter_count] = res_exact

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    ### Step 3: Final 3/4-step ###
    dt = cfl * np.min([dx, dy])

    ry = dt / dy
    rx = 0.5 * dt / dx  # half-step

    # final y-sweep: U^(n+1/4) -> U^(n+3/4)
    U = y_sweep(U, ry, dt, params, f_all, boundaries)

    # final x-sweep: U^(n+3/4) -> U^(n+1)
    U = x_sweep(U, rx, dt, params, f_all, boundaries)

    # since dt is different between step 1 and step 3, we
    # estimate the total dt value between both iterations as
    # 0.5 * (dt_init + dt_final)
    t += 0.5 * dt

    # extra iteration to account for step 1 and step 3
    iter_count += 1

    return U, t, iter_count, res_exact, err_conv
