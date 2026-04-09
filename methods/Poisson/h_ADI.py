import BC_2D as bc
from methods.Poisson.poisson_helpers import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def get_params(config):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = config.cfl, config.dx, config.dy, config.Nx, config.Ny

    # experiment on dt
    dt_factor = 1.0
    dt = dt_factor * cfl * np.min([dx, dy])

    ### set up matrices for implicit-explicit systems ###

    # Sparse tridiagonals (1st derivatives)
    Dx = sp.diags([-1.0, 1.0], offsets=[-1, 0], shape=(Nx - 2, Nx - 2), format="csr") / dx
    Dy = sp.diags([-1.0, 1.0], offsets=[-1, 0], shape=(Ny - 2, Ny - 2), format="csr") / dy

    # Sparse tridiagonals (2nd derivatives)
    Dxx = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Nx - 2, Nx - 2), format="csr") / (dx ** 2)
    Dyy = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Ny - 2, Ny - 2), format="csr") / (dy ** 2)

    # Dx_Dy = Dx + Dy
    lhs_x = Dx - (0.5 * dt * Dxx)
    lhs_y = Dy - (0.5 * dt * Dyy)

    solve_x = spla.factorized(lhs_x.tocsc())
    solve_y = spla.factorized(lhs_y.tocsc())

    update = "leapfrog"  # "leapfrog" or "bdf1" (hyperbolic update rule)

    # get the accurate initial state (just to initialize U_old_half, Unm1)
    boundaries = bc.get_bcs_poisson_2d(config, config.phi_exact)
    U_init_copy = config.U0.copy()
    bc.apply_bcs_poisson_2d(U_init_copy, config, boundaries)

    U_old_half, Unm1, Unm2 = None, None, None

    if update == "leapfrog":    # U^(n-1/2) initialization for use in leapfrog update
        U_old_half = U_init_copy
    elif update == "bdf1":      # U^(n-1) initialization for use in bdf1 update
        Unm1 = U_init_copy

    params = {
        "dt": dt,
        "update": update,
        "solve_x": solve_x,
        "solve_y": solve_y,
        "U_old_half": U_old_half,
        "Unm1": Unm1,
        "Unm2": Unm2,
    }

    return params


# maybe should pass extra parameters to main `solve` method in poisson2D
# because this full sweep needs an extra timestep? (U_old_half/Unm1)

def full_sweep(U, params, boundaries, config):
    dt, update = params["dt"], params["update"]
    solve_x, solve_y = params["solve_x"], params["solve_y"]
    U_old_half, Unm1, Unm2 = params["U_old_half"], params["Unm1"], params["Unm2"]

    if update == "leapfrog":
        U_new_half, U_new = leapfrog_update(U, U_old_half, config, solve_x, solve_y, dt, boundaries)
        params["U_old_half"] = U_new_half.copy()
        U = U_new.copy()
    elif update == "bdf1":
        U_new = bdf1_update(U, Unm1, config, solve_x, solve_y, dt, boundaries)
        params["Unm1"] = U.copy()
        U = U_new.copy()

    return U


def leapfrog_update(U, U_old_half, params, solve_x, solve_y, dt, boundaries):
    Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega, f_all = params.omega, params.f_all

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


def bdf1_update(U, Unm1, params, solve_x, solve_y, dt, boundaries):
    Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega, f_all = params.omega, params.f_all

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


