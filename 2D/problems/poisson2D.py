import BC_2D as bc
from methods.Poisson import p_unsplit, p_ADI, h_ADI, h_unsplit, multigrid as mg
from methods.Poisson.poisson_helpers import *
from problems.poisson_tests.two_blob import A, B, f, phi_exact, rho


def get_all_f_from_xy(params) -> np.ndarray:
    x, y = params.x, params.y
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx+2*ngc, Ny+2*ngc)
    return params._eval_rhs(X, Y)


# config: Config object
# get_params, full_sweep: method-specific functions
def solve(config, get_params, full_sweep):
    # ------------------------------------------------------------
    # 1. setup
    # ------------------------------------------------------------
    params = get_params(config)
    dt = params["dt"]

    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    Nx, Ny = config.Nx, config.Ny
    norm, iter_max = config.norm, config.iter_max
    phi_exact, f_all = config.phi_exact, config.f_all
    U_exact = config.U_exact

    # tolerances (move these into config?)
    tol_update = 1e-5
    tol_residual = 1e-5
    res_max = 1e8

    # initialization
    boundaries = bc.get_bcs_poisson_2d(config, config.phi_exact)

    U = config.U0.copy()
    bc.apply_bcs_poisson_2d(U, config, boundaries)

    # error/residual storages
    update_hist = []        # U - U_old
    residual_hist = []      # ||L(U) - f||
    exact_hist = []         # U - U_exact

    # best iterate tracking (for diagnostics only)
    best_exact = np.inf
    U_best = U.copy()
    best_iter = 0

    t = 0.0
    iter_count = 0

    # ------------------------------------------------------------
    # 2. iteration loop
    # ------------------------------------------------------------
    while iter_count < iter_max:
        U_old = U.copy()

        # full sweep (method-specific)
        U = full_sweep(U, params, boundaries, config)

        # BC updates handled within method updates

        # -------------------------
        # update norm (primary)
        # -------------------------
        dU = U - U_old
        res_update = np.linalg.norm(dU[ib:ie+1, jb:je+1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_update /= np.sqrt(Nx * Ny)

        update_hist.append(res_update)

        # -------------------------
        # residual (||L(U) - f||)
        # -------------------------
        if f_all is not None:
            L = get_all_laplace(U, config)  # gets f(X, Y) over whole grid
            R = L - f_all
            res_residual = np.linalg.norm(R[ib:ie+1, jb:je+1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_residual /= np.sqrt(Nx * Ny)
        else:
            res_residual = np.nan

        residual_hist.append(res_residual)

        # -------------------------
        # exact error (diagnostic)
        # -------------------------
        if phi_exact is not None:
            err = U - U_exact
            res_exact = np.linalg.norm(err[ib:ie+1, jb:je+1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_exact /= np.sqrt(Nx * Ny)

            exact_hist.append(res_exact)

            # track best iterate (diagnostic only)
            if res_exact < best_exact:
                best_exact = res_exact
                U_best = U.copy()
                best_iter = iter_count
        else:
            exact_hist.append(np.nan)

        # -------------------------
        # stopping conditions
        # -------------------------
        converged_update = res_update < tol_update
        converged_residual = (f_all is None) or (res_residual < tol_residual)

        if converged_update and converged_residual:
            break

        if not np.isfinite(res_update) or res_update > res_max:
            break

        # advance time / iteration
        t += dt
        iter_count += 1

    # ------------------------------------------------------------
    # 3. prepare return
    # ------------------------------------------------------------
    histories = {
        "update": np.array(update_hist),
        "residual": np.array(residual_hist),
        "exact": np.array(exact_hist),
        "best_exact": best_exact,
        "best_iter": best_iter,
    }

    # ------------------------------------------------------------
    # 4. return
    # ------------------------------------------------------------
    return U, t, iter_count, histories


def solve_all(config):
    if config.multigrid:
        match config.method:
            case "p_unsplit":
                U, t, cycle_count, histories = mg.solve(config, p_unsplit.get_params, p_unsplit.full_sweep)
            case "h_unsplit":
                U, t, cycle_count, histories = mg.solve(config, h_unsplit.get_params, h_unsplit.full_sweep)
            case "p_ADI":
                U, t, cycle_count, histories = mg.solve(config, p_ADI.get_params, p_ADI.full_sweep)
            case "h_ADI":
                U, t, cycle_count, histories = mg.solve(config, h_ADI.get_params, h_ADI.full_sweep)
            case _:
                raise ValueError("Unknown multigrid solver method")
        return U, t, cycle_count, histories

    else:   # no multigrid
        match config.method:
            case "p_unsplit":
                U, t, iter_count, histories = solve(config, p_unsplit.get_params, p_unsplit.full_sweep)
            case "h_unsplit":
                U, t, iter_count, histories = solve(config, h_unsplit.get_params, h_unsplit.full_sweep)
            case "p_ADI":
                U, t, iter_count, histories = solve(config, p_ADI.get_params, p_ADI.full_sweep)
            case "h_ADI":
                U, t, iter_count, histories = solve(config, h_ADI.get_params, h_ADI.full_sweep)
            case _:
                raise ValueError("Unknown solver method")

    return U, t, iter_count, histories
