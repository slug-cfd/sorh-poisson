"""Diagnostics for the Poisson polar-star problem.

This script intentionally does not modify or patch solver implementations.
It checks:

1. continuous consistency of rhs with Delta(exact),
2. discrete consistency of L_h(exact) with rhs,
3. exact solution of the discrete Poisson problem via a sparse direct solve,
4. iterative solver convergence against that direct discrete solution,
5. RHS component scaling and local max-error locations.

Run from ``2D/`` with:
    uv run python -m diagnostics.diagnose_polar_star
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    ProblemConfig,
    RunConfig,
    SimulationContext,
    compute_omega_opt,
    initialize_simulation,
)
from methods.poisson.poisson_helpers import get_all_laplace
from problems import poisson
from problems.poisson_tests import polar_star


OMEGA_DATA = Path(__file__).resolve().parent / "data" / "reference_omega_tables.json"


def _stored_omega(method: str, n: int) -> float:
    with OMEGA_DATA.open() as stream:
        data = json.load(stream)
    grid = "128" if n >= 128 else "64" if n >= 64 else "32"
    methods = data["methods_by_solver"]["poisson_base"]
    return data["omega_by_grid"][grid]["poisson_base"][methods.index(method)]


def make_config(
    n: int,
    epsilon: float,
    method: str = "h_unsplit",
    iter_max: int = 4000,
    omega: float | None = None,
    use_reference_omega: bool = False,
) -> SimulationContext:
    if omega is not None and use_reference_omega:
        raise ValueError("Specify either omega or use_reference_omega, not both.")
    configured_omega = (
        _stored_omega(method, n)
        if use_reference_omega
        else 1.0 if omega is None else omega
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return initialize_simulation(
            Config(
                problem=ProblemConfig(
                    kind="poisson",
                    case="polar_star",
                    parameters={"epsilon": epsilon},
                ),
                grid=GridConfig(Nx=n, Ny=n),
                run=RunConfig(
                    method=method,
                    cfl=0.8,
                    omega=configured_omega,
                    use_omega_opt=omega is None and not use_reference_omega,
                    iter_max=iter_max,
                    norm="fro",
                ),
                methods=MethodConfig(
                    h_unsplit=HyperbolicUnsplitConfig(
                        mode="weighted_hybrid",
                        theta=0.25 * np.pi,
                        alpha=0.8,
                        beta=0.8,
                    ),
                ),
            )
        )


def omega_description(config: SimulationContext, use_reference_omega: bool) -> str:
    if use_reference_omega:
        return "reference table"
    if config.settings.run.use_omega_opt:
        expected = compute_omega_opt(config.settings, config.grid)
        if not np.isclose(config.omega, expected):
            raise AssertionError("initialized omega does not match compute_omega_opt")
        return "computed estimate"
    return "explicit"


def interior_norms(
    values: np.ndarray,
    config: SimulationContext,
) -> tuple[float, float, float, float]:
    interior = values[config.ib : config.ie + 1, config.jb : config.je + 1]
    scaled_fro = np.linalg.norm(interior, ord="fro") / np.sqrt(config.Nx * config.Ny)
    return (
        float(scaled_fro),
        float(np.max(np.abs(interior))),
        float(np.min(interior)),
        float(np.max(interior)),
    )


def direct_discrete_solve(config: SimulationContext) -> np.ndarray:
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

    operator = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(num_unknowns, num_unknowns),
    )
    interior_solution = spsolve(operator, rhs)

    solution = config.U_exact.copy()
    for i in range(ib, ie + 1):
        start = (i - ib) * ny
        solution[i, jb : je + 1] = interior_solution[start : start + ny]
    return solution


def continuous_rhs_check(
    epsilon: float,
    n: int = 64,
    samples: int = 200,
) -> dict[str, float]:
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
        "rmse_lap_minus_rhs": float(
            np.linalg.norm(laplace_fd - rhs) / np.sqrt(samples)
        ),
        "rmse_lap_plus_rhs": float(
            np.linalg.norm(laplace_fd + rhs) / np.sqrt(samples)
        ),
        "max_abs_rhs": float(np.max(np.abs(rhs))),
    }


def rhs_component_stats(
    config: SimulationContext,
) -> dict[str, tuple[float, float, float, float]]:
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


def print_continuous_check(
    epsilons: list[float],
    n: int,
    samples: int,
) -> None:
    print("Continuous RHS check")
    for epsilon in epsilons:
        check = continuous_rhs_check(epsilon, n=n, samples=samples)
        print(
            "epsilon={epsilon:g} h={h:.1e} "
            "rmse(Delta u - rhs)={rmse_lap_minus_rhs:.3e} "
            "rmse(Delta u + rhs)={rmse_lap_plus_rhs:.3e} "
            "max_abs_rhs={max_abs_rhs:.3e}".format(**check)
        )


def print_discrete_grid_study(
    epsilons: list[float],
    grid_sizes: list[int],
) -> None:
    print("\nDiscrete exact/direct grid study")
    print(
        "epsilon,N,dx/epsilon,rhs_norm,exact_discrete_res,"
        "direct_exact_error,direct_max_error"
    )
    for epsilon in epsilons:
        for n in grid_sizes:
            config = make_config(n, epsilon)
            laplace_exact = get_all_laplace(
                config.U_exact,
                config,
                config.laplacian_stencil,
            )
            exact_discrete_res = interior_norms(laplace_exact - config.f_all, config)[0]
            rhs_norm = interior_norms(config.f_all, config)[0]
            direct = direct_discrete_solve(config)
            direct_error, direct_max_error, _, _ = interior_norms(
                direct - config.U_exact,
                config,
            )
            print(
                f"{epsilon:g},{n},{config.dx / epsilon:.3f},{rhs_norm:.3e},"
                f"{exact_discrete_res:.3e},{direct_error:.3e},{direct_max_error:.3e}"
            )


def print_high_resolution_discrete_check(
    epsilon: float,
    grid_sizes: list[int],
) -> None:
    print(f"\nHigh-resolution direct check, epsilon={epsilon:g}")
    print("N,dx/epsilon,exact_discrete_res,direct_exact_error,direct_max_error")
    for n in grid_sizes:
        config = make_config(n, epsilon)
        laplace_exact = get_all_laplace(
            config.U_exact,
            config,
            config.laplacian_stencil,
        )
        exact_discrete_res = interior_norms(laplace_exact - config.f_all, config)[0]
        direct = direct_discrete_solve(config)
        direct_error, direct_max_error, _, _ = interior_norms(
            direct - config.U_exact,
            config,
        )
        print(
            f"{n},{config.dx / config.epsilon:.3f},{exact_discrete_res:.3e},"
            f"{direct_error:.3e},{direct_max_error:.3e}"
        )


def print_solver_sweep(
    epsilons: list[float],
    methods: list[str],
    n: int,
    iter_max: int,
    omega: float | None,
    use_reference_omega: bool,
) -> None:
    print(f"\nSolver vs direct discrete solution, N={n}")
    print(
        "method,epsilon,omega,omega_source,iters,update_final,equation_residual_final,"
        "exact_error_final,best_exact_error,error_vs_direct,direct_exact_error,elapsed_sec"
    )
    for method in methods:
        for epsilon in epsilons:
            config = make_config(
                n,
                epsilon,
                method=method,
                iter_max=iter_max,
                omega=omega,
                use_reference_omega=use_reference_omega,
            )
            direct = direct_discrete_solve(config)
            direct_exact_error = interior_norms(direct - config.U_exact, config)[0]

            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                solution, _, iterations, histories = poisson.solve_all(config)
            elapsed = time.perf_counter() - start

            error_vs_direct = interior_norms(solution - direct, config)[0]
            print(
                f"{method},{epsilon:g},{config.omega:.6g},"
                f"{omega_description(config, use_reference_omega)},{iterations},"
                f"{histories['update'][-1]:.3e},{histories['residual'][-1]:.3e},"
                f"{histories['exact'][-1]:.3e},{histories['best_exact']:.3e},"
                f"{error_vs_direct:.3e},{direct_exact_error:.3e},{elapsed:.2f}"
            )


def print_rhs_scaling_and_local_error(
    epsilons: list[float],
    n: int,
) -> None:
    print(f"\nRHS component scaling, N={n}")
    print("epsilon,component,norm,max_abs,min,max")
    for epsilon in epsilons:
        config = make_config(n, epsilon)
        for component, stats in rhs_component_stats(config).items():
            norm, max_abs, min_value, max_value = stats
            print(
                f"{epsilon:g},{component},{norm:.3e},{max_abs:.3e},"
                f"{min_value:.3e},{max_value:.3e}"
            )

    local_epsilon = epsilons[-1]
    config = make_config(n, local_epsilon)
    direct = direct_discrete_solve(config)
    error = direct - config.U_exact
    interior_abs = np.abs(error[config.ib : config.ie + 1, config.jb : config.je + 1])
    largest = np.argpartition(interior_abs.ravel(), -5)[-5:]
    largest = largest[np.argsort(interior_abs.ravel()[largest])[::-1]]

    print(
        f"\nTop local direct-vs-exact errors, "
        f"epsilon={local_epsilon:g}, N={n}"
    )
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


CHECKS = ("continuous", "discrete", "high_resolution", "solver", "rhs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=CHECKS,
        default=["continuous"],
        help="diagnostic sections to run; the bounded continuous check is the default",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.05, 0.02, 0.01],
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 96, 128],
    )
    parser.add_argument(
        "--high-resolution-grid-sizes",
        type=int,
        nargs="+",
        default=[160, 192, 256, 384],
    )
    parser.add_argument("--high-resolution-epsilon", type=float, default=0.01)
    parser.add_argument("--continuous-n", type=int, default=64)
    parser.add_argument("--continuous-samples", type=int, default=200)
    parser.add_argument("--solver-n", type=int, default=64)
    parser.add_argument("--rhs-n", type=int, default=64)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("h_unsplit", "p_unsplit"),
        default=["h_unsplit", "p_unsplit"],
    )
    parser.add_argument("--iter-max", type=int, default=4000)
    omega_group = parser.add_mutually_exclusive_group()
    omega_group.add_argument("--omega", type=float)
    omega_group.add_argument(
        "--use-reference-omega",
        action="store_true",
        help="use the legacy reference table instead of the computed estimate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "continuous" in args.checks:
        print_continuous_check(
            args.epsilons,
            n=args.continuous_n,
            samples=args.continuous_samples,
        )
    if "discrete" in args.checks:
        print_discrete_grid_study(args.epsilons, args.grid_sizes)
    if "high_resolution" in args.checks:
        print_high_resolution_discrete_check(
            args.high_resolution_epsilon,
            args.high_resolution_grid_sizes,
        )
    if "solver" in args.checks:
        print_solver_sweep(
            args.epsilons,
            args.methods,
            n=args.solver_n,
            iter_max=args.iter_max,
            omega=args.omega,
            use_reference_omega=args.use_reference_omega,
        )
    if "rhs" in args.checks:
        print_rhs_scaling_and_local_error(args.epsilons, n=args.rhs_n)


if __name__ == "__main__":
    main()
