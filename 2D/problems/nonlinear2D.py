import numpy as np
import BC_2D as bc
from methods.ToyProblem.nonlinear import p_unsplit, h_unsplit, DS1, DS2

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
    # U_exact = config.U_exact

    # tolerances (move these into config?)
    tol_update = 1e-5
    res_max = 1e8

    # initialization
    U = config.U0.copy()
    bc.update_BCs(U, config)

    # error/residual storages
    update_hist = []        # U - U_old

    # best iterate tracking (for diagnostics only)
    best_update = np.inf
    U_res_min = U.copy()
    best_iter = 0

    t = 0.0
    iter_count = 0

    # ------------------------------------------------------------
    # 2. iteration loop
    # ------------------------------------------------------------
    while iter_count < iter_max:
        U_old = U.copy()

        # full sweep (method-specific)
        U = full_sweep(U, params, config)

        # BC updates handled within method updates

        # -------------------------
        # update norm (primary)
        # -------------------------
        dU = U - U_old
        res_update = np.linalg.norm(dU[ib:ie+1, jb:je+1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_update /= np.sqrt(Nx * Ny)

        update_hist.append(res_update)

        # track best iterate (diagnostic only)
        if res_update < best_update:
            best_update = res_update
            U_res_min = U.copy()
            best_iter = iter_count

        # -------------------------
        # stopping conditions
        # -------------------------
        converged_update = res_update < tol_update

        if converged_update:
            break

        if not np.isfinite(res_update) or res_update > res_max:
            break

        # advance time / iteration
        t += dt
        iter_count += 1

    if config.method == "DS2":  # DS2 needs final 1/2-sweep
        params["last_sweep"] = True
        U = full_sweep(U, params, config)

    # ------------------------------------------------------------
    # 3. prepare return
    # ------------------------------------------------------------
    histories = {
        "update": np.array(update_hist),
        "U_res_min": np.array(U_res_min),
        "best_update": best_update,
        "best_iter": best_iter,
    }

    # ------------------------------------------------------------
    # 4. return
    # ------------------------------------------------------------
    return U, t, iter_count, histories



def solve_all(config):
    match config.method:
        case "p_unsplit":       # parabolic unsplit
            U, t, iter_count, histories = solve(config, p_unsplit.get_params, p_unsplit.full_sweep)
        case "h_unsplit":       # hyperbolic unsplit
            U, t, iter_count, histories = solve(config, h_unsplit.get_params, h_unsplit.full_sweep)
        case "DS1":     # 1st-order dimensional splitting (DS1)
            U, t, iter_count, histories = solve(config, DS1.get_params, DS1.full_sweep)
        case "DS2":     # 2nd-order dimensional splitting (DS2)
            U, t, iter_count, histories = solve(config, DS2.get_params, DS2.full_sweep)
        case _:
            return -1

    return U, t, iter_count, histories
