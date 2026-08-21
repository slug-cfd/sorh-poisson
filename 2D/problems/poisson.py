"""Solver orchestration for registered 2D Poisson problems."""

import numpy as np

import boundary_conditions as bc
from methods.poisson import p_unsplit, p_adi, h_adi, h_unsplit, multigrid as mg
from methods.poisson.poisson_helpers import get_all_laplace
from problems.poisson_tests.two_blob import A, B, f, phi_exact, rho
from solver_utils import normalized_grid_norm, stopping_reason


# config: initialized SimulationContext
# get_params, full_sweep: method-specific functions
def solve(config, get_params, full_sweep):
    # ------------------------------------------------------------
    # 1. setup
    # ------------------------------------------------------------
    params = get_params(config)
    print("omega =", config.omega)

    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    norm, iter_max = config.norm, config.iter_max
    phi_exact, f_all = config.phi_exact, config.f_all
    U_exact = config.U_exact
    if f_all is None:
        raise ValueError("Poisson solve requires a discrete RHS array.")

    # initialization
    boundaries = bc.get_bcs_poisson_2d(config, config.phi_exact)

    U = config.U0.copy()
    bc.apply_bcs_poisson_2d(U, config, boundaries)

    # error/residual storages
    update_hist = []        # U - U_old
    residual_hist = []      # ||L(U) - f||
    exact_hist = []         # U - U_exact
    dt_hist = []

    # best iterate tracking (for diagnostics only)
    best_exact = np.inf
    U_best = U.copy()
    best_iter = 0

    t = 0.0
    iter_count = 0
    reason = "iter_max"
    converged = False
    stable = True

    # ------------------------------------------------------------
    # 2. iteration loop
    # ------------------------------------------------------------
    for iter_count in range(1, iter_max + 1):
        U_old = U.copy()

        # full sweep (method-specific)
        U = full_sweep(U, params, boundaries, config)

        # BC updates handled within method updates

        # -------------------------
        # update norm (stagnation safeguard)
        # -------------------------
        dU = U - U_old
        res_update = normalized_grid_norm(
            dU[ib:ie + 1, jb:je + 1],
            norm,
        )

        update_hist.append(res_update)

        dt = float(params.get("dt", 0.0))
        dt_hist.append(dt)
        t += dt

        # -------------------------
        # residual (primary: ||L(U) - f||)
        # -------------------------
        L = get_all_laplace(U, config, config.laplacian_stencil)
        R = L - f_all
        res_residual = normalized_grid_norm(
            R[ib:ie + 1, jb:je + 1],
            norm,
        )

        residual_hist.append(res_residual)

        # -------------------------
        # exact error (diagnostic)
        # -------------------------
        if phi_exact is not None:
            err = U - U_exact
            res_exact = normalized_grid_norm(
                err[ib:ie + 1, jb:je + 1],
                norm,
            )

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
        decision = stopping_reason(
            res_update,
            res_residual,
            config.stopping_config,
        )
        if decision is not None:
            reason = decision
            converged = decision == "tol_residual"
            stable = decision not in {"nonfinite", "res_max"}
            break

    # ------------------------------------------------------------
    # 3. prepare return
    # ------------------------------------------------------------
    histories = {
        "update": np.array(update_hist),
        "residual": np.array(residual_hist),
        "exact": np.array(exact_hist),
        "dt": np.array(dt_hist),
        "best_exact": best_exact,
        "best_iter": best_iter,
        "U_best": U_best,
        "reason": reason,
        "converged": converged,
        "stable": stable,
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
                U, t, cycle_count, histories = mg.solve(config, p_adi.get_params, p_adi.full_sweep)
            case "h_ADI":
                U, t, cycle_count, histories = mg.solve(config, h_adi.get_params, h_adi.full_sweep)
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
                U, t, iter_count, histories = solve(config, p_adi.get_params, p_adi.full_sweep)
            case "h_ADI":
                U, t, iter_count, histories = solve(config, h_adi.get_params, h_adi.full_sweep)
            case _:
                raise ValueError("Unknown solver method")

    return U, t, iter_count, histories
