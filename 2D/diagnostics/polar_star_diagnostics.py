"""Diagnostics for the Poisson polar-star problem.

This script intentionally does not modify or patch solver implementations.
It checks:

1. continuous consistency of rhs with Delta(exact),
2. discrete consistency of L_h(exact) with rhs,
3. exact solution of the discrete Poisson problem via a sparse direct solve,
4. iterative solver convergence against that direct discrete solution,
5. RHS component scaling and local max-error locations.

Run with:
    uv run python diagnostics/polar_star_diagnostics.py
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config_storage
from config import Config, HyperbolicUnsplitConfig, MultiGridConfig
from methods.Poisson.poisson_helpers import get_all_laplace
from problems import poisson2D
from problems.poisson_tests import polar_star


def _stored_omega(method: str, n: int) -> float:
    if n >= 128 and hasattr(config_storage, "omegas128"):
        omega_table = config_storage.omegas128
    elif n >= 64:
        omega_table = config_storage.omegas64
    else:
        omega_table = config_storage.omegas32
    method_names = config_storage.methods["poisson_base"]
    return omega_table["poisson_base"][method_names.index(method)]


def make_config(
    n: int,
    epsilon: float,
    method: str = "h_unsplit",
    iter_max: int = 4000,
) -> Config:
    polar_star.PROBLEM.constants["epsilon"] = epsilon
    with contextlib.redirect_stdout(io.StringIO()):
        return Config(
            problem="poisson",
            poisson_problem="polar_star",
            method=method,
            bc_type="poisson",
            Nx=n,
            Ny=n,
            cfl=0.8,
            omega=_stored_omega(method, n),
            iter_max=iter_max,
            norm="fro",
            mg_config=MultiGridConfig(),
            h_unsplit_config=HyperbolicUnsplitConfig(
                mode="weighted_hybrid",
                theta=0.25 * np.pi,
                alpha=0.8,
                beta=0.8,
            ),
            multigrid=False,
        )


def interior_norms(values: np.ndarray, config: Config) -> tuple[float, float, float, float]:
    interior = values[config.ib : config.ie + 1, config.jb : config.je + 1]
    scaled_fro = np.linalg.norm(interior, ord="fro") / np.sqrt(config.Nx * config.Ny)
    return (
        float(scaled_fro),
        float(np.max(np.abs(interior))),
        float(np.min(interior)),
        float(np.max(interior)),
    )


def direct_discrete_solve(config: Config) -> np.ndarray:
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    ny = je - jb + 1
    num_unknowns = (ie - ib + 1) * ny
    dx2 = config.dx**2
    dy2 = config.dy**2

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.empty(num_unknowns)

    def flat_index(i: int, j: int) -> int:
        return (i - ib) * ny + (j - jb)

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            row = flat_index(i, j)
            rhs[row] = config.f_all[i, j]
            rows.append(row)
            cols.append(row)
            vals.append(-2.0 / dx2 - 2.0 / dy2)

            for ni, nj, coefficient in (
                (i - 1, j, 1.0 / dx2),
                (i + 1, j, 1.0 / dx2),
                (i, j - 1, 1.0 / dy2),
                (i, j + 1, 1.0 / dy2),
            ):
                if ib <= ni <= ie and jb <= nj <= je:
                    rows.append(row)
                    cols.append(flat_index(ni, nj))
                    vals.append(coefficient)
                else:
                    rhs[row] -= coefficient * config.U_exact[ni, nj]

    operator = sparse.csr_matrix((vals, (rows, cols)), shape=(num_unknowns, num_unknowns))
    interior_solution = spsolve(operator, rhs)

    solution = config.U_exact.copy()
    for i in range(ib, ie + 1):
        start = (i - ib) * ny
        solution[i, jb : je + 1] = interior_solution[start : start + ny]
    return solution


def continuous_rhs_check(epsilon: float, n: int = 64, samples: int = 200) -> dict[str, float]:
    config = make_config(n, epsilon)
    rng = np.random.default_rng(1)
    x = rng.uniform(config.xa, config.xb, samples)
    y = rng.uniform(config.ya, config.yb, samples)
    h = max(epsilon * 1e-3, 1e-5)

    with contextlib.redirect_stdout(io.StringIO()):
        u0 = config._eval_exact(x, y)
        ux_plus = config._eval_exact(x + h, y)
        ux_minus = config._eval_exact(x - h, y)
        uy_plus = config._eval_exact(x, y + h)
        uy_minus = config._eval_exact(x, y - h)
        rhs = config._eval_rhs(x, y)

    laplace_fd = (ux_plus - 2.0 * u0 + ux_minus) / h**2
    laplace_fd += (uy_plus - 2.0 * u0 + uy_minus) / h**2

    return {
        "epsilon": epsilon,
        "h": h,
        "rmse_lap_minus_rhs": float(np.linalg.norm(laplace_fd - rhs) / np.sqrt(samples)),
        "rmse_lap_plus_rhs": float(np.linalg.norm(laplace_fd + rhs) / np.sqrt(samples)),
        "max_abs_rhs": float(np.max(np.abs(rhs))),
    }


def rhs_component_stats(config: Config) -> dict[str, tuple[float, float, float, float]]:
    x_grid, y_grid = np.meshgrid(config.x, config.y, indexing="ij")
    totals = {
        "s1_over_r2": np.zeros_like(x_grid),
        "s2_over_r2": np.zeros_like(x_grid),
        "s3": np.zeros_like(x_grid),
        "neg_s4": np.zeros_like(x_grid),
    }

    for star in polar_star.POLAR_STARS:
        rad = polar_star.radius(x_grid, y_grid, star["x0"], star["y0"])
        rad_safe = np.where(rad > 1e-12, rad, np.inf)
        ang = polar_star.theta_polar_star(x_grid, y_grid, star["x0"], star["y0"])
        phi_i = polar_star.phi(x_grid, y_grid, star, config.epsilon)
        p_i = polar_star.p(x_grid, y_grid, star)
        sech2 = polar_star.sech(phi_i) ** 2
        tanh = np.tanh(phi_i)

        n = star["n"]
        r0 = star["r0"]
        r1 = star["r1"]
        s1 = (p_i**2) * tanh * sech2 / config.epsilon**2
        s2 = -(n**2) * r0 * r1 * np.cos(n * ang) * sech2 / (2.0 * config.epsilon)
        s3 = tanh * sech2 / config.epsilon**2
        s4 = sech2 / (2.0 * rad_safe * config.epsilon)

        totals["s1_over_r2"] += s1 / rad_safe**2
        totals["s2_over_r2"] += s2 / rad_safe**2
        totals["s3"] += s3
        totals["neg_s4"] += -s4

    totals["rhs"] = sum(totals.values())
    return {name: interior_norms(values, config) for name, values in totals.items()}


def print_continuous_check() -> None:
    check = continuous_rhs_check(0.01)
    print("Continuous RHS check, epsilon=0.01")
    print(
        "h={h:.1e} rmse(Delta u - rhs)={rmse_lap_minus_rhs:.3e} "
        "rmse(Delta u + rhs)={rmse_lap_plus_rhs:.3e} max_abs_rhs={max_abs_rhs:.3e}".format(
            **check
        )
    )


def print_discrete_grid_study() -> None:
    print("\nDiscrete exact/direct grid study")
    print("epsilon,N,dx/epsilon,rhs_norm,exact_discrete_res,direct_exact_error,direct_max_error")
    for epsilon in (0.1, 0.05, 0.02, 0.01):
        for n in (32, 64, 96, 128):
            config = make_config(n, epsilon)
            laplace_exact = get_all_laplace(config.U_exact, config)
            exact_discrete_res = interior_norms(laplace_exact - config.f_all, config)[0]
            rhs_norm = interior_norms(config.f_all, config)[0]
            direct = direct_discrete_solve(config)
            direct_error, direct_max_error, _, _ = interior_norms(direct - config.U_exact, config)
            print(
                f"{epsilon:g},{n},{config.dx / epsilon:.3f},{rhs_norm:.3e},"
                f"{exact_discrete_res:.3e},{direct_error:.3e},{direct_max_error:.3e}"
            )


def print_high_resolution_discrete_check() -> None:
    print("\nHigh-resolution direct check, epsilon=0.01")
    print("N,dx/epsilon,exact_discrete_res,direct_exact_error,direct_max_error")
    for n in (160, 192, 256, 384):
        config = make_config(n, 0.01)
        laplace_exact = get_all_laplace(config.U_exact, config)
        exact_discrete_res = interior_norms(laplace_exact - config.f_all, config)[0]
        direct = direct_discrete_solve(config)
        direct_error, direct_max_error, _, _ = interior_norms(direct - config.U_exact, config)
        print(
            f"{n},{config.dx / config.epsilon:.3f},{exact_discrete_res:.3e},"
            f"{direct_error:.3e},{direct_max_error:.3e}"
        )


def print_solver_sweep() -> None:
    print("\nSolver vs direct discrete solution, N=64")
    print(
        "method,epsilon,omega,iters,update_final,equation_residual_final,"
        "exact_error_final,best_exact_error,error_vs_direct,direct_exact_error,elapsed_sec"
    )
    for method in ("h_unsplit", "p_unsplit"):
        for epsilon in (0.1, 0.05, 0.02, 0.01):
            config = make_config(64, epsilon, method=method)
            direct = direct_discrete_solve(config)
            direct_exact_error = interior_norms(direct - config.U_exact, config)[0]

            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                solution, _, iterations, histories = poisson2D.solve_all(config)
            elapsed = time.perf_counter() - start

            error_vs_direct = interior_norms(solution - direct, config)[0]
            print(
                f"{method},{epsilon:g},{config.omega:.3f},{iterations},"
                f"{histories['update'][-1]:.3e},{histories['residual'][-1]:.3e},"
                f"{histories['exact'][-1]:.3e},{histories['best_exact']:.3e},"
                f"{error_vs_direct:.3e},{direct_exact_error:.3e},{elapsed:.2f}"
            )


def print_rhs_scaling_and_local_error() -> None:
    print("\nRHS component scaling, N=64")
    print("epsilon,component,norm,max_abs,min,max")
    for epsilon in (0.1, 0.05, 0.02, 0.01):
        config = make_config(64, epsilon)
        for component, stats in rhs_component_stats(config).items():
            norm, max_abs, min_value, max_value = stats
            print(
                f"{epsilon:g},{component},{norm:.3e},{max_abs:.3e},"
                f"{min_value:.3e},{max_value:.3e}"
            )

    config = make_config(64, 0.01)
    direct = direct_discrete_solve(config)
    error = direct - config.U_exact
    interior_abs = np.abs(error[config.ib : config.ie + 1, config.jb : config.je + 1])
    largest = np.argpartition(interior_abs.ravel(), -5)[-5:]
    largest = largest[np.argsort(interior_abs.ravel()[largest])[::-1]]

    print("\nTop local direct-vs-exact errors, epsilon=0.01, N=64")
    print("x,y,abs_error,error,rhs,exact,direct")
    for flat_pos in largest:
        ii, jj = np.unravel_index(flat_pos, interior_abs.shape)
        i = ii + config.ib
        j = jj + config.jb
        print(
            f"{config.x[i]:.6f},{config.y[j]:.6f},{abs(error[i, j]):.3e},"
            f"{error[i, j]:.3e},{config.f_all[i, j]:.3e},"
            f"{config.U_exact[i, j]:.3e},{direct[i, j]:.3e}"
        )


if __name__ == "__main__":
    print_continuous_check()
    print_discrete_grid_study()
    print_high_resolution_discrete_check()
    print_solver_sweep()
    print_rhs_scaling_and_local_error()
