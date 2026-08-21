"""Benchmarks for the 2D directional-ridge Poisson test.

This file intentionally exercises the public config/solver interfaces only.
It does not modify or reach into the method implementations.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    ProblemConfig,
    RunConfig,
    SimulationContext,
    initialize_simulation,
)
from methods.poisson import h_unsplit
from problems import poisson


PROBLEM_NAME = "directional_ridge"
DEFAULT_THETA = 0.75 * math.pi
DEFAULT_DIRECTION_NORM = math.sqrt(2.0)
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
TOL_UPDATE = 1e-5
TOL_RESIDUAL = 1e-5
RES_MAX = 1e8


def make_config(
    omega: float,
    theta: float = DEFAULT_THETA,
    *,
    nx: int = 64,
    ny: int = 64,
    iter_max: int = 5000,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    direction_norm: float = DEFAULT_DIRECTION_NORM,
) -> SimulationContext:
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case=PROBLEM_NAME),
            grid=GridConfig(Nx=nx, Ny=ny),
            run=RunConfig(
                method="h_unsplit",
                cfl=0.8,
                omega=float(omega),
                iter_max=iter_max,
                norm="fro",
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(
                    mode="directional",
                    theta=float(theta),
                    normalize_direction=True,
                    direction_norm=float(direction_norm),
                    alpha=float(alpha),
                    beta=float(beta),
                ),
            ),
        )
    )


def _last(history: np.ndarray) -> float:
    if len(history) == 0:
        return math.nan
    return float(history[-1])


def run_case(config: SimulationContext) -> dict[str, float | int | bool | str]:
    start = time.perf_counter()
    _, _, iter_count, histories = poisson.solve_all(config)
    elapsed = time.perf_counter() - start

    update_hist = histories["update"]
    residual_hist = histories["residual"]
    exact_hist = histories["exact"]
    update_final = _last(update_hist)
    residual_final = _last(residual_hist)
    exact_final = _last(exact_hist)
    sweeps = len(update_hist)

    stable = (
        np.all(np.isfinite(update_hist))
        and np.all(np.isfinite(residual_hist))
        and float(np.nanmax(update_hist)) < RES_MAX
        and float(np.nanmax(residual_hist)) < RES_MAX
    )
    converged = (
        stable
        and update_final < TOL_UPDATE
        and residual_final < TOL_RESIDUAL
    )

    hu_config = config.h_unsplit_config
    quadrant_mode = h_unsplit.get_params(config)["quadrant_mode"]
    return {
        "problem": config.problem_case,
        "method": config.method,
        "mode": hu_config.mode,
        "quadrant_mode": quadrant_mode,
        "Nx": config.Nx,
        "Ny": config.Ny,
        "omega": config.omega,
        "theta": hu_config.theta,
        "theta_degrees": math.degrees(hu_config.theta),
        "direction_norm": hu_config.direction_norm,
        "alpha": hu_config.alpha,
        "beta": hu_config.beta,
        "iter_count": int(iter_count),
        "sweeps": sweeps,
        "converged": converged,
        "stable": stable,
        "update_final": update_final,
        "residual_final": residual_final,
        "exact_final": exact_final,
        "elapsed_sec": elapsed,
    }


def run_logged_case(config: SimulationContext) -> dict[str, float | int | bool | str]:
    hu_config = config.h_unsplit_config
    print(
        f"running omega={config.omega:.2f}, "
        f"theta={math.degrees(hu_config.theta):.1f} deg",
        flush=True,
    )
    row = run_case(config)
    print(
        f"  sweeps={int(row['sweeps'])}, "
        f"converged={row['converged']}, "
        f"stable={row['stable']}, "
        f"residual={float(row['residual_final']):.3e}",
        flush=True,
    )
    return row


def _float_range(start: float, stop: float, step: float) -> list[float]:
    values = np.arange(start, stop + 0.5 * step, step)
    return [round(float(value), 10) for value in values]


def _sort_key(row: dict[str, float | int | bool | str]) -> tuple[int, int, float]:
    if not row["stable"]:
        return (2, int(row["sweeps"]), float(row["elapsed_sec"]))
    if not row["converged"]:
        return (1, int(row["sweeps"]), float(row["elapsed_sec"]))
    return (0, int(row["sweeps"]), float(row["elapsed_sec"]))


def optimize_omega(
    *,
    theta: float = DEFAULT_THETA,
    nx: int = 64,
    ny: int = 64,
    iter_max: int = 5000,
    coarse_min: float = 0.90,
    coarse_max: float = 0.99,
    coarse_step: float = 0.03,
    fine_radius: float = 0.04,
    fine_step: float = 0.01,
) -> tuple[float, list[dict[str, float | int | bool | str]]]:
    rows = []
    tested = set()

    for omega in _float_range(coarse_min, coarse_max, coarse_step):
        config = make_config(omega, theta, nx=nx, ny=ny, iter_max=iter_max)
        rows.append(run_logged_case(config))
        tested.add(round(omega, 10))

    coarse_best = min(rows, key=_sort_key)
    center = float(coarse_best["omega"])
    fine_min = max(coarse_min, center - fine_radius)
    fine_max = min(coarse_max, center + fine_radius)

    for omega in _float_range(fine_min, fine_max, fine_step):
        key = round(omega, 10)
        if key in tested:
            continue
        config = make_config(omega, theta, nx=nx, ny=ny, iter_max=iter_max)
        rows.append(run_logged_case(config))
        tested.add(key)

    best = min(rows, key=_sort_key)
    return float(best["omega"]), sorted(rows, key=lambda row: float(row["omega"]))


def sweep_theta(
    omega: float,
    *,
    nx: int = 64,
    ny: int = 64,
    iter_max: int = 5000,
    theta_values: tuple[float, ...] = (
        math.pi / 12.0,
        math.pi / 6.0,
        math.pi / 4.0,
        math.pi / 3.0,
        5.0 * math.pi / 12.0,
        7.0 * math.pi / 12.0,
        2.0 * math.pi / 3.0,
        3.0 * math.pi / 4.0,
        5.0 * math.pi / 6.0,
        11.0 * math.pi / 12.0,
    ),
) -> list[dict[str, float | int | bool | str]]:
    rows = []
    for theta in theta_values:
        config = make_config(omega, theta, nx=nx, ny=ny, iter_max=iter_max)
        rows.append(run_logged_case(config))
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | bool | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(title: str, rows: list[dict[str, float | int | bool | str]]) -> None:
    print(f"\n{title}")
    print("omega, theta_deg, quadrant, sweeps, converged, stable, residual_final, exact_final, elapsed_sec")
    for row in rows:
        print(
            f"{float(row['omega']):.2f}, "
            f"{float(row['theta_degrees']):.1f}, "
            f"{row['quadrant_mode']}, "
            f"{int(row['sweeps'])}, "
            f"{row['converged']}, "
            f"{row['stable']}, "
            f"{float(row['residual_final']):.3e}, "
            f"{float(row['exact_final']):.3e}, "
            f"{float(row['elapsed_sec']):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--iter-max", type=int, default=5000)
    parser.add_argument("--outdir", type=Path, default=Path("results/directional_ridge"))
    parser.add_argument("--omega-min", type=float, default=0.90)
    parser.add_argument("--omega-max", type=float, default=0.99)
    parser.add_argument("--omega-step", type=float, default=0.03)
    parser.add_argument("--fine-radius", type=float, default=0.04)
    parser.add_argument("--fine-step", type=float, default=0.01)
    args = parser.parse_args()

    best_omega, omega_rows = optimize_omega(
        nx=args.nx,
        ny=args.ny,
        iter_max=args.iter_max,
        coarse_min=args.omega_min,
        coarse_max=args.omega_max,
        coarse_step=args.omega_step,
        fine_radius=args.fine_radius,
        fine_step=args.fine_step,
    )
    theta_rows = sweep_theta(
        best_omega,
        nx=args.nx,
        ny=args.ny,
        iter_max=args.iter_max,
    )

    write_rows(args.outdir / "omega_default_theta.csv", omega_rows)
    write_rows(args.outdir / "theta_sweep.csv", theta_rows)

    print_summary("Omega sweep at default theta", omega_rows)
    print(f"\nBest omega at theta={math.degrees(DEFAULT_THETA):.1f} deg: {best_omega:.2f}")
    print_summary("Theta sweep at best default-theta omega", theta_rows)


if __name__ == "__main__":
    main()
