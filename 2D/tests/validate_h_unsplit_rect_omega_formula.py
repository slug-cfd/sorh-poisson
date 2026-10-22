"""Validate the default SORh omega estimate on rectangular Poisson grids."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    ProblemConfig,
    RunConfig,
    initialize_simulation,
)
from problems import poisson


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = BASE_DIR / "results" / "generated_formula_validation" / "rectangular"
TOL_UPDATE = 1e-5
TOL_RESIDUAL = 1e-5
RES_MAX = 1e8

FIELDNAMES = (
    "poisson_problem",
    "mode",
    "Nx",
    "Ny",
    "cfl",
    "c_max",
    "omega_predicted",
    "omega",
    "omega_offset",
    "solver_iter_count",
    "sweeps",
    "converged",
    "stable",
    "update_final",
    "residual_final",
    "exact_final",
    "elapsed_sec",
)


def predicted_omega(Nx, Ny):
    nx_intervals = Nx + 1
    ny_intervals = Ny + 1
    c_max = (
        nx_intervals * math.cos(math.pi / nx_intervals)
        + ny_intervals * math.cos(math.pi / ny_intervals)
    ) / (nx_intervals + ny_intervals)
    omega = 1.0 / (1.0 + math.sqrt(1.0 - c_max**2))
    return omega, c_max


def centered_omega_values(center, half_width, step):
    offsets = np.arange(-half_width, half_width + 0.5 * step, step)
    values = [center + float(offset) for offset in offsets]
    if not any(math.isclose(value, center, rel_tol=0.0, abs_tol=1e-14) for value in values):
        values.append(center)
    return sorted({round(value, 14): value for value in values}.values())


def make_config(
    *,
    Nx,
    Ny,
    cfl,
    omega,
    iter_max,
    poisson_problem,
    laplacian_stencil,
    ghost_layers,
):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case=poisson_problem),
            grid=GridConfig(
                Nx=Nx,
                Ny=Ny,
                laplacian_stencil=laplacian_stencil,
                ghost_layers=ghost_layers,
            ),
            run=RunConfig(
                method="h_unsplit",
                cfl=cfl,
                omega=float(omega),
                iter_max=iter_max,
                norm="fro",
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(mode="default"),
            ),
        )
    )


def last(history):
    if len(history) == 0:
        return math.nan
    return float(history[-1])


def run_case(config, omega_pred, c_max):
    start = time.perf_counter()
    _, _, iter_count, histories = poisson.solve_all(config)
    elapsed = time.perf_counter() - start

    update = histories["update"]
    residual = histories["residual"]
    exact = histories["exact"]

    update_final = last(update)
    residual_final = last(residual)
    exact_final = last(exact)
    update_max = float(np.nanmax(update)) if len(update) else math.inf
    residual_max = float(np.nanmax(residual)) if len(residual) else math.inf

    stable = (
        np.all(np.isfinite(update))
        and np.all(np.isfinite(residual))
        and update_max < RES_MAX
        and residual_max < RES_MAX
    )
    converged = stable and update_final < TOL_UPDATE and residual_final < TOL_RESIDUAL

    return {
        "poisson_problem": config.problem_case,
        "mode": config.h_unsplit_config.mode,
        "Nx": config.Nx,
        "Ny": config.Ny,
        "cfl": config.cfl,
        "c_max": c_max,
        "omega_predicted": omega_pred,
        "omega": config.omega,
        "omega_offset": config.omega - omega_pred,
        "solver_iter_count": int(iter_count),
        "sweeps": len(update),
        "converged": converged,
        "stable": stable,
        "update_final": update_final,
        "residual_final": residual_final,
        "exact_final": exact_final,
        "elapsed_sec": elapsed,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    grid_keys = sorted({(row["Nx"], row["Ny"]) for row in rows})
    for Nx, Ny in grid_keys:
        group = [
            row for row in rows
            if row["Nx"] == Nx and row["Ny"] == Ny and row["converged"] and row["stable"]
        ]
        if not group:
            summary.append({"Nx": Nx, "Ny": Ny, "status": "no converged stable runs"})
            continue

        best_sweeps = min(row["sweeps"] for row in group)
        best_rows = [row for row in group if row["sweeps"] == best_sweeps]
        predicted_row = min(group, key=lambda row: abs(row["omega_offset"]))
        closest_best = min(best_rows, key=lambda row: abs(row["omega_offset"]))
        confirmed = predicted_row["sweeps"] == best_sweeps

        summary.append(
            {
                "Nx": Nx,
                "Ny": Ny,
                "status": "confirmed" if confirmed else "near_miss",
                "poisson_problem": predicted_row["poisson_problem"],
                "mode": predicted_row["mode"],
                "cfl": predicted_row["cfl"],
                "c_max": predicted_row["c_max"],
                "omega_predicted": predicted_row["omega_predicted"],
                "predicted_sweeps": predicted_row["sweeps"],
                "best_omega": closest_best["omega"],
                "best_omega_offset": closest_best["omega_offset"],
                "best_sweeps": best_sweeps,
                "sweep_gap": predicted_row["sweeps"] - best_sweeps,
                "best_elapsed_sec": closest_best["elapsed_sec"],
            }
        )
    return summary


def print_summary(summary):
    print("2D rectangular h_unsplit omega formula validation")
    print("problem: two_blob")
    print("method: h_unsplit, mode: default")
    print("formula: use Mx=Nx+1 and My=Ny+1 grid intervals")
    print("         c_max = (Mx*cos(pi/Mx) + My*cos(pi/My)) / (Mx + My)")
    print("         omega* = 1 / (1 + sqrt(1 - c_max**2))")
    print()
    for row in summary:
        if row["status"] == "no converged stable runs":
            print(f"Nx={row['Nx']}, Ny={row['Ny']}: no converged stable runs")
            continue
        print(
            f"Nx={row['Nx']:>3}, Ny={row['Ny']:>3}: "
            f"predicted={row['omega_predicted']:.12f} "
            f"pred_sweeps={row['predicted_sweeps']} "
            f"best={row['best_omega']:.12f} "
            f"best_sweeps={row['best_sweeps']} "
            f"offset={row['best_omega_offset']:+.6g} "
            f"status={row['status']}"
        )


def parse_grid(value):
    try:
        left, right = value.lower().split("x", maxsplit=1)
        Nx = int(left)
        Ny = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Grid '{value}' must be formatted like 32x64."
        ) from exc
    if Nx == Ny:
        raise argparse.ArgumentTypeError("Use non-square grids only for this test.")
    return Nx, Ny


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep omega around the rectangular-grid formula for the 2D two-blob "
            "Poisson h_unsplit default method."
        )
    )
    parser.add_argument(
        "--grids",
        type=parse_grid,
        nargs="+",
        default=[(16, 32), (24, 48), (32, 64), (48, 64)],
    )
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--half-width", type=float, default=0.02)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--iter-max", type=int, default=5000)
    parser.add_argument("--poisson-problem", default="two_blob")
    parser.add_argument("--laplacian-stencil", default="centered")
    parser.add_argument("--ghost-layers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def default_output_dir(cfl):
    value = f"{cfl:g}"
    if "." not in value:
        value += ".0"
    return (
        RESULTS_ROOT / f"cfl{value.replace('.', 'p')}"
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.cfl)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    rows = []
    for Nx, Ny in args.grids:
        omega_pred, c_max = predicted_omega(Nx, Ny)
        for omega in centered_omega_values(omega_pred, args.half_width, args.step):
            config = make_config(
                Nx=Nx,
                Ny=Ny,
                cfl=args.cfl,
                omega=omega,
                iter_max=args.iter_max,
                poisson_problem=args.poisson_problem,
                laplacian_stencil=args.laplacian_stencil,
                ghost_layers=args.ghost_layers,
            )
            rows.append(run_case(config, omega_pred, c_max))

    summary = summarize(rows)
    write_csv(output_dir / "omega_sweep.csv", rows)
    with (output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print_summary(summary)
    print()
    print(f"wrote {output_dir / 'omega_sweep.csv'}")
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
