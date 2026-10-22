from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from config1D import Config1D
from methods.linear import (
    gs_parabolic as gs_linear_parabolic,
    jacobi_parabolic as jacobi_linear_parabolic,
    sor_hyperbolic as sor_linear_hyperbolic,
    sor_hyperbolic_fourth as sor_linear_hyperbolic_fourth,
    sor_parabolic as sor_linear_parabolic,
    sor_parabolic_fourth as sor_linear_parabolic_fourth,
)
from methods.nonlinear import (
    gs_parabolic as gs_nonlinear_parabolic,
    jacobi_parabolic as jacobi_nonlinear_parabolic,
    sor_hyperbolic as sor_nonlinear_hyperbolic,
    sor_parabolic as sor_nonlinear_parabolic,
)
from problem_helpers import equation_residual, exact_diagnostic


METHOD_REGISTRY = {
    "jacobi_linear_parabolic": jacobi_linear_parabolic,
    "jacobi_nonlinear_parabolic": jacobi_nonlinear_parabolic,
    "gs_linear_parabolic": gs_linear_parabolic,
    "gs_nonlinear_parabolic": gs_nonlinear_parabolic,
    "sor_linear_parabolic": sor_linear_parabolic,
    "sor_linear_parabolic_fourth": sor_linear_parabolic_fourth,
    "sor_nonlinear_parabolic": sor_nonlinear_parabolic,
    "sor_linear_hyperbolic": sor_linear_hyperbolic,
    "sor_linear_hyperbolic_fourth_comp": sor_linear_hyperbolic_fourth,
    "sor_linear_hyperbolic_fourth_bdf2": sor_linear_hyperbolic_fourth,
    "sor_nonlinear_hyperbolic": sor_nonlinear_hyperbolic,
}


@dataclass
class RunResult1D:
    converged: bool
    stable: bool
    iter_count: int
    residual_final: float
    residual_min: float
    residual_max: float
    elapsed_sec: float
    reason: str
    equation_residual_final: float | None = None


@dataclass
class RunRecord1D:
    config: Config1D
    results: RunResult1D
    meta: dict

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "results": asdict(self.results),
            "meta": self.meta,
        }


def _norm(values, ord_value, normalize_count: int) -> float:
    result = float(np.linalg.norm(values, ord=ord_value))
    if ord_value in (1, 2):
        result /= np.sqrt(normalize_count)
    return result


def _interior(U, config):
    return U[config.ib:config.ie + 1]


def solve(config: Config1D):
    method_module = METHOD_REGISTRY[config.method]
    params = method_module.get_params(config)

    U = config.U0.copy()
    config.apply_bcs(U)

    update_hist = []
    exact_hist = []
    equation_hist = []
    dt_hist = []

    best_exact = np.inf
    best_iter = 0
    U_best = U.copy()

    t = 0.0
    reason = "iter_max"
    stable = True
    converged = False

    for iter_count in range(1, config.iter_max + 1):
        U_old = U.copy()
        U = method_module.full_sweep(U, params, config)
        config.apply_bcs(U)

        update_res = _norm(
            _interior(U - U_old, config),
            config.norm,
            config.interior_size,
        )
        exact_res = _norm(
            _interior(exact_diagnostic(U, config), config),
            config.norm,
            config.interior_size,
        )
        equation_values = equation_residual(U, config)
        equation_res = (
            _norm(
                _interior(equation_values, config),
                config.norm,
                config.interior_size,
            )
            if equation_values is not None
            else np.nan
        )

        update_hist.append(update_res)
        exact_hist.append(exact_res)
        equation_hist.append(equation_res)

        dt = params.get("dt", 0.0)
        dt_hist.append(dt)
        t += dt

        if exact_res < best_exact:
            best_exact = exact_res
            best_iter = iter_count
            U_best = U.copy()

        if not np.isfinite(update_res) or update_res > config.solver_config.res_max:
            stable = False
            reason = "unstable"
            break

        if update_res < config.solver_config.tol_update:
            converged = True
            reason = "tol_update"
            break

        tol_exact = config.solver_config.tol_exact
        if config.solver_config.stop_on_exact and tol_exact is not None and exact_res < tol_exact:
            converged = True
            reason = "tol_exact"
            break
    else:
        iter_count = config.iter_max

    histories = {
        "update": np.array(update_hist),
        "exact": np.array(exact_hist),
        "equation": np.array(equation_hist),
        "dt": np.array(dt_hist),
        "best_exact": best_exact,
        "best_iter": best_iter,
        "U_best": U_best,
        "reason": reason,
        "converged": converged,
        "stable": stable,
        "method_params": {
            key: value
            for key, value in params.items()
            if key in {"c0", "lhs_stencil", "p1", "p2", "r", "v"}
        },
    }

    return U, t, iter_count, histories


def run_simulation(config: Config1D):
    start = time.perf_counter()
    U, t, iter_count, histories = solve(config)
    elapsed = time.perf_counter() - start

    residuals = histories["update"]
    if residuals.size == 0:
        residual_final = np.inf
        residual_min = np.inf
        residual_max = np.inf
    else:
        residual_final = float(residuals[-1])
        residual_min = float(np.min(residuals))
        residual_max = float(np.max(residuals))

    equation_residuals = histories["equation"]
    equation_residual_final = (
        float(equation_residuals[-1])
        if equation_residuals.size and np.isfinite(equation_residuals[-1])
        else None
    )

    record = RunRecord1D(
        config=config,
        results=RunResult1D(
            converged=histories["converged"],
            stable=histories["stable"],
            iter_count=iter_count,
            residual_final=residual_final,
            residual_min=residual_min,
            residual_max=residual_max,
            equation_residual_final=equation_residual_final,
            elapsed_sec=elapsed,
            reason=histories["reason"],
        ),
        meta={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_display_name": config.run_display_name,
            "t": t,
            "exact_final": float(histories["exact"][-1]) if histories["exact"].size else np.inf,
            "best_exact": float(histories["best_exact"]),
            "best_iter": histories["best_iter"],
            "method_params": histories["method_params"],
        },
    )

    return U, t, iter_count, histories, record
