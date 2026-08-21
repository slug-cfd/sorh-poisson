"""Solver orchestration for the nonlinear diffusion problem."""

import numpy as np
import boundary_conditions as bc
from methods.nonlinear_diffusion import ds1, ds2, h_unsplit, nl_helpers, p_unsplit
from solver_utils import normalized_grid_norm, stopping_reason

# config: initialized SimulationContext
# get_params, full_sweep: method-specific functions
def solve(config, get_params, full_sweep):
    # ------------------------------------------------------------
    # 1. setup
    # ------------------------------------------------------------
    params = get_params(config)

    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    norm, iter_max = config.norm, config.iter_max

    # initialization
    U = config.U0.copy()
    bc.update_BCs(U, config)

    # error/residual storages
    update_hist = []
    residual_hist = []
    dt_hist = []

    # best iterate tracking (diagnostic only)
    best_update = np.inf
    best_residual = np.inf
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

        try:
            U = full_sweep(U, params, config)
            bc.update_BCs(U, config)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            U = U_old
            update_hist.append(np.inf)
            residual_hist.append(np.inf)
            dt_hist.append(0.0)
            reason = "numerical_guard"
            stable = False
            break

        dt = float(params.get("dt", 0.0))
        dt_hist.append(dt)
        t += dt

        # -------------------------
        # update norm (stagnation safeguard)
        # -------------------------
        dU = U - U_old
        res_update = normalized_grid_norm(
            dU[ib:ie + 1, jb:je + 1],
            norm,
        )

        update_hist.append(res_update)

        # -------------------------
        # nonlinear equation residual (primary)
        # -------------------------
        try:
            residual = nl_helpers.nonlinear_flux_residual(U, config)
            res_residual = normalized_grid_norm(residual, norm)
        except FloatingPointError:
            res_residual = np.inf
        residual_hist.append(res_residual)

        if res_residual < best_residual:
            best_residual = res_residual
            best_update = res_update
            U_best = U.copy()
            best_iter = iter_count

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

    if config.method == "DS2" and converged:  # DS2 needs final 1/2-sweep
        params["last_sweep"] = True
        U_before_final = U.copy()
        try:
            U = full_sweep(U, params, config)
            bc.update_BCs(U, config)
            final_update = normalized_grid_norm(
                (U - U_before_final)[ib:ie + 1, jb:je + 1],
                norm,
            )
            final_residual = normalized_grid_norm(
                nl_helpers.nonlinear_flux_residual(U, config),
                norm,
            )
            if update_hist:
                update_hist[-1] = final_update
                residual_hist[-1] = final_residual
            final_decision = stopping_reason(
                final_update,
                final_residual,
                config.stopping_config,
            )
            if final_decision != "tol_residual":
                converged = False
                reason = "ds2_final_sweep_not_converged"
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            U = U_before_final
            converged = False
            stable = False
            reason = "numerical_guard"

    # ------------------------------------------------------------
    # 3. prepare return
    # ------------------------------------------------------------
    histories = {
        "update": np.array(update_hist),
        "residual": np.array(residual_hist),
        "dt": np.array(dt_hist),
        "U_best": U_best,
        "U_res_min": U_best,
        "best_update": best_update,
        "best_residual": best_residual,
        "best_iter": best_iter,
        "reason": reason,
        "converged": converged,
        "stable": stable,
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
            U, t, iter_count, histories = solve(config, ds1.get_params, ds1.full_sweep)
        case "DS2":     # 2nd-order dimensional splitting (DS2)
            U, t, iter_count, histories = solve(config, ds2.get_params, ds2.full_sweep)
        case _:
            raise ValueError(f"Unknown nonlinear_diffusion solver method '{config.method}'.")

    return U, t, iter_count, histories
