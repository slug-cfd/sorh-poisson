import scipy.sparse as sp
import scipy.sparse.linalg as spla
import BC_2D as bc
from methods.Poisson.poisson_helpers import *


def parabolic_update(U, params, solve_x, solve_y, dt, f_all, boundaries):
    Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega = params.omega

    phi_xL, phi_xR, phi_yB, phi_yT = boundaries
    U_old = U.copy()

    # 1st step (implicit in x, explicit in y)
    U_new = U.copy()

    h_in = np.zeros(Nx-4)
    for j in range(jb, je+1):
        f_j = f_all[ib:ie+1, j]
        h_j = np.concatenate(([phi_xL[j] / (dx ** 2)], h_in, [phi_xR[j] / (dx ** 2)]))
        Dyy_j = (U[ib:ie+1, j-1] - (2.0 * U[ib:ie+1, j]) + U[ib:ie+1, j+1]) / dy**2
        rhs = U[ib:ie+1, j] + (0.5 * dt * (Dyy_j - f_j + h_j))
        U_new[ib:ie+1, j] = solve_x(rhs)

    # 2nd step (implicit in y, explicit in x)
    U = U_new.copy()

    h_in = np.zeros(Ny-4)
    for i in range(ib, ie+1):
        f_i = f_all[i, jb:je+1]
        h_i = np.concatenate(([phi_yB[i] / (dy ** 2)], h_in, [phi_yT[i] / (dy ** 2)]))
        Dxx_i = (U[i-1, jb:je+1] - (2.0 * U[i, jb:je+1]) + U[i+1, jb:je+1]) / dx**2
        rhs = U[i, jb:je+1] + (0.5 * dt * (Dxx_i - f_i + h_i))
        U_new[i, jb:je+1] = solve_y(rhs)

    # SOR update
    U_new = ((1.0 - omega) * U_old) + (omega * U_new)

    return U_new


def parabolic(params, phi_exact, get_all_f):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega, iter_max, norm = params.omega, params.iter_max, params.norm

    boundaries = bc.get_bcs_poisson_2d(params, phi_exact)

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, boundaries)

    # experiment on dt
    dt_factor = 1.0
    dt = dt_factor * cfl * np.min([dx, dy])

    # set up matrices for implicit-explicit systems
    # Sparse identities (CSR by default)
    Ix = sp.identity(Nx-2, format="csr")
    Iy = sp.identity(Ny-2, format="csr")

    # Sparse tridiagonals
    Tx = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Nx-2, Nx-2), format="csr")
    Ty = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Ny-2, Ny-2), format="csr")

    cx = (0.5 * dt / dx ** 2)
    cy = (0.5 * dt / dy ** 2)

    Dxx = cx * Tx
    Dyy = cy * Ty

    Ixm = Ix - Dxx  # still CSR
    Iym = Iy - Dyy

    solve_x = spla.factorized(Ixm.tocsc())
    solve_y = spla.factorized(Iym.tocsc())

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7    # error/residual tolerance
    res_max   = 1e7     # upper bound to avoid blow-up in unstable cases
    res_diff  = 1e2     # difference in norm of the solution between current and previous iteration

    # we need a separate variable for res_exact because we are still
    # using res_diff as the criterion to exit the iteration loop,
    # even though we are returning and plotting res_exact.
    res_exact = 1e4
    res_prev = 0.0

    metric = "exact"

    x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
    X, Y = np.meshgrid(x, y, indexing='ij')

    # print("using exact solution")
    U_exact = phi_exact(X, Y, rho0, R, G)

    while res_diff > threshold and iter_count < iter_max and res_exact < res_max:
        #  store the old solution to measure the steadiness
        U_old = U.copy()

        U_new = parabolic_update(U, params, solve_x, solve_y, dt, f_all, boundaries)
        U = U_new.copy()

        # update BCs
        # bc.apply_bcs_poisson_2d(U, params, boundaries)

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

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res_exact, err_conv


def hyperbolic_update_leapfrog(U, U_old_half, params, solve_x, solve_y, dt, f_all, boundaries):
    Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega = params.omega

    U_old = U.copy()
    phi_xL, phi_xR, phi_yB, phi_yT = boundaries

    # ----------------------------
    # 1) x-implicit, y-explicit
    # ----------------------------
    U_new_half = U.copy()

    for j in range(jb, je + 1):
        f_j = f_all[ib:ie + 1, j]

        # Explicit pieces (computed from full U arrays)
        Dx_j = (U[ib:ie + 1, j] - U[ib - 1:ie, j]) / dx
        Dy_j = (U[ib:ie + 1, j] - U[ib:ie + 1, j - 1]) / dy
        Dyy_j = (U[ib:ie + 1, j - 1] - 2.0 * U[ib:ie + 1, j] + U[ib:ie + 1, j + 1]) / (dy ** 2)
        Dy_j_half = (U_old_half[ib:ie + 1, j] - U_old_half[ib:ie + 1, j - 1]) / dy

        # Base RHS (leapfrog)
        rhs = (Dx_j + Dy_j) + (0.5 * dt) * (Dyy_j - f_j) - Dy_j_half

        # ---- BC corrections for (Dx - 0.5*dt*Dxx) on the LHS moved to RHS ----
        # Dx backward diff needs left boundary at first interior point
        rhs[0] += phi_xL[j] / dx

        # Dxx needs both boundaries at first/last interior point, scaled by 0.5*dt
        rhs[0]  += 0.5 * dt * (phi_xL[j] / (dx ** 2))
        rhs[-1] += 0.5 * dt * (phi_xR[j] / (dx ** 2))

        # Solve line system
        U_new_half[ib:ie+1, j] = solve_x(rhs)

    # SOR update
    # U_new_half[ib:ie+1, je:je+1] = (((1.0 - omega) * U_old_half[ib:ie+1, je:je+1])
    #                                 + (omega * U_new_half[ib:ie+1, je:je+1]))
    U_new_half = (((1.0 - omega) * U_old_half) + (omega * U_new_half))
    bc.apply_bcs_poisson_2d(U_new_half, params, boundaries)

    # ----------------------------
    # 2) y-implicit, x-explicit
    # ----------------------------
    U_new = U_new_half.copy()

    for i in range(ib, ie + 1):
        f_i = f_all[i, jb:je + 1]

        # Explicit pieces (computed from full U arrays)
        Dx_i = (U_new_half[i, jb:je + 1] - U_new_half[i - 1, jb:je + 1]) / dx
        Dy_i = (U_new_half[i, jb:je + 1] - U_new_half[i, jb - 1:je]) / dy
        Dxx_i = (U_new_half[i - 1, jb:je + 1] - 2.0 * U_new_half[i, jb:je + 1]
                 + U_new_half[i + 1, jb:je + 1]) / (dx ** 2)
        Dx_i_half = (U[i, jb:je + 1] - U[i - 1, jb:je + 1]) / dx

        # Base RHS (leapfrog)
        rhs = (Dx_i + Dy_i) + (0.5 * dt) * (Dxx_i - f_i) - Dx_i_half

        # ---- BC corrections for (Dy - 0.5*dt*Dyy) on the LHS moved to RHS ----
        # Dy backward diff needs bottom boundary at first interior point
        rhs[0] += phi_yB[i] / dy

        # Dyy needs both boundaries at first/last interior point, scaled by 0.5*dt
        rhs[0]  += 0.5 * dt * (phi_yB[i] / (dy ** 2))
        rhs[-1] += 0.5 * dt * (phi_yT[i] / (dy ** 2))

        # Solve line system
        U_new[i, jb:je+1] = solve_y(rhs)

    # SOR update
    # U_new[ib:ie+1, je:je+1] = (((1.0 - omega) * U_old[ib:ie+1, je:je+1])
    #                            + (omega * U_new[ib:ie+1, je:je+1]))
    U_new = (((1.0 - omega) * U_old) + (omega * U_new))
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new_half, U_new


def hyperbolic_update_bdf1(U, Unm1, params, solve_x, solve_y, dt, f_all, boundaries):
    Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega = params.omega

    U_old = U.copy()
    phi_xL, phi_xR, phi_yB, phi_yT = boundaries

    # ----------------------------
    # 1) x-implicit, y-explicit
    # ----------------------------
    U_new_half = U.copy()

    for j in range(jb, je + 1):
        f_j = f_all[ib:ie + 1, j]

        # Explicit pieces (computed from full U arrays)
        Dx_j = (U[ib:ie + 1, j] - U[ib - 1:ie, j]) / dx
        Dy_j = (U[ib:ie + 1, j] - U[ib:ie + 1, j - 1]) / dy
        Dyy_j = (U[ib:ie + 1, j - 1] - 2.0 * U[ib:ie + 1, j] + U[ib:ie + 1, j + 1]) / (dy ** 2)
        Dy_j_old = (Unm1[ib:ie + 1, j] - Unm1[ib:ie + 1, j - 1]) / dy

        # Base RHS (BDF1)
        rhs = (Dx_j - Dy_j) + (0.5 * dt) * (Dyy_j - f_j) + Dy_j_old

        # ---- BC corrections for (Dx - 0.5*dt*Dxx) on the LHS moved to RHS ----
        # Dx backward diff needs left boundary at first interior point
        rhs[0] += phi_xL[j] / dx

        # Dxx needs both boundaries at first/last interior point, scaled by 0.5*dt
        rhs[0]  += 0.5 * dt * (phi_xL[j] / (dx ** 2))
        rhs[-1] += 0.5 * dt * (phi_xR[j] / (dx ** 2))

        # Solve line system
        U_new_half[ib:ie+1, j] = solve_x(rhs)


    # SOR update
    # U_new_half = (((1.0 - omega) * U_old) + (omega * U_new_half))
    # bc.apply_bcs_poisson_2d(U, params, boundaries)

    # ----------------------------
    # 2) y-implicit, x-explicit
    # ----------------------------
    U_new = U_new_half.copy()

    for i in range(ib, ie + 1):
        f_i = f_all[i, jb:je + 1]

        # Explicit pieces (computed from full U arrays)
        Dx_i = (U_new_half[i, jb:je + 1] - U_new_half[i - 1, jb:je + 1]) / dx
        Dy_i = (U_new_half[i, jb:je + 1] - U_new_half[i, jb - 1:je]) / dy
        Dxx_i = (U_new_half[i - 1, jb:je + 1] - 2.0 * U_new_half[i, jb:je + 1]
                 + U_new_half[i + 1, jb:je + 1]) / (dx ** 2)
        Dx_i_old = (Unm1[i, jb:je + 1] - Unm1[i - 1, jb:je + 1]) / dx

        # Base RHS (leapfrog)
        rhs = (Dy_i - Dx_i) + (0.5 * dt) * (Dxx_i - f_i) + Dx_i_old

        # ---- BC corrections for (Dy - 0.5*dt*Dyy) on the LHS moved to RHS ----
        # Dy backward diff needs bottom boundary at first interior point
        rhs[0] += phi_yB[i] / dy

        # Dyy needs both boundaries at first/last interior point, scaled by 0.5*dt
        rhs[0]  += 0.5 * dt * (phi_yB[i] / (dy ** 2))
        rhs[-1] += 0.5 * dt * (phi_yT[i] / (dy ** 2))

        # Solve line system
        U_new[i, jb:je+1] = solve_y(rhs)

    # SOR update
    U_new = (((1.0 - omega) * U_old) + (omega * U_new))
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def hyperbolic(params, phi_exact, get_all_f):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega, iter_max, norm = params.omega, params.iter_max, params.norm

    boundaries = bc.get_bcs_poisson_2d(params, phi_exact)

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, boundaries)

    # experiment on dt
    dt_factor = 1.0
    dt = dt_factor * cfl * np.min([dx, dy])

    # set up matrices for implicit-explicit systems

    # Sparse tridiagonals (1st derivatives)
    Dx = sp.diags([-1.0, 1.0], offsets=[-1, 0], shape=(Nx-2, Nx-2), format="csr") / dx
    Dy = sp.diags([-1.0, 1.0], offsets=[-1, 0], shape=(Ny-2, Ny-2), format="csr") / dy

    # Sparse tridiagonals (2nd derivatives)
    Dxx = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Nx-2, Nx-2), format="csr") / (dx ** 2)
    Dyy = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Ny-2, Ny-2), format="csr") / (dy ** 2)

    # Dx_Dy = Dx + Dy
    lhs_x = Dx - (0.5 * dt * Dxx)
    lhs_y = Dy - (0.5 * dt * Dyy)

    solve_x = spla.factorized(lhs_x.tocsc())
    solve_y = spla.factorized(lhs_y.tocsc())

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    #  error array (one value per iteration)
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

    metric = "exact"
    update = "leapfrog"     # "leapfrog" or "bdf1" (hyperbolic update rule)

    x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
    X, Y = np.meshgrid(x, y, indexing='ij')

    # print("using exact solution")
    U_exact = phi_exact(X, Y, rho0, R, G)

    U_old_half = U.copy()   # U^(n-1/2) initialization for use in leapfrog update
    Unm1 = U.copy()         # U^(n-1) initialization for use in bdf1 update
    # Unm2 = U.copy()

    while res_diff > threshold and iter_count < iter_max and res_exact < res_max:
        #  store the old solution to measure the steadiness
        U_old = U.copy()

        if update == "leapfrog":
            U_new_half, U_new = hyperbolic_update_leapfrog(U, U_old_half, params,
                                                           solve_x, solve_y, dt, f_all, boundaries)
            U_old_half = U_new_half.copy()
            U = U_new.copy()
        elif update == "bdf1":
            U_new = hyperbolic_update_bdf1(U, Unm1, params, solve_x, solve_y,
                                           dt, f_all, boundaries)
            Unm1 = U.copy()
            U = U_new.copy()

        # update BCs (not needed here)
        # bc.apply_bcs_poisson_2d(U, params, boundaries)

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

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res_exact, err_conv
