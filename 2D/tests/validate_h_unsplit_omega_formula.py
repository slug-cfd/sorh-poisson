"""Validate the default SORh omega estimate on square Poisson grids."""

import argparse
import csv
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
RESULTS_DIR = BASE_DIR / "results" / "h_unsplit_omega_formula"
TOL_UPDATE = 1e-5
TOL_RESIDUAL = 1e-5
RES_MAX = 1e8

FIELDNAMES = (
    "poisson_problem",
    "mode",
    "Nx",
    "Ny",
    "cfl",
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


def predicted_omega(interior_count):
    return 1.0 / (1.0 + math.sin(math.pi / (interior_count + 1)))


def centered_omega_values(center, half_width, step):
    offsets = np.arange(-half_width, half_width + 0.5 * step, step)
    values = [center + float(offset) for offset in offsets]
    if not any(math.isclose(value, center, rel_tol=0.0, abs_tol=1e-14) for value in values):
        values.append(center)
    return sorted({round(value, 14): value for value in values}.values())


def make_config(
    *,
    N,
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
                Nx=N,
                Ny=N,
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


def run_case(config, omega_pred):
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
    for N in sorted({row["Nx"] for row in rows}):
        group = [
            row for row in rows
            if row["Nx"] == N and row["converged"] and row["stable"]
        ]
        if not group:
            summary.append({"N": N, "status": "no converged stable runs"})
            continue

        best_sweeps = min(row["sweeps"] for row in group)
        best_rows = [row for row in group if row["sweeps"] == best_sweeps]
        predicted_row = min(group, key=lambda row: abs(row["omega_offset"]))
        closest_best = min(best_rows, key=lambda row: abs(row["omega_offset"]))
        confirmed = predicted_row["sweeps"] == best_sweeps

        summary.append(
            {
                "N": N,
                "status": "confirmed" if confirmed else "near_miss",
                "poisson_problem": predicted_row["poisson_problem"],
                "mode": predicted_row["mode"],
                "cfl": predicted_row["cfl"],
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
    print("2D h_unsplit omega formula validation")
    print("problem: two_blob")
    print("method: h_unsplit, mode: default")
    print("formula: omega* = 1 / (1 + sin(pi / (M + 1))); M = interior count")
    print()
    for row in summary:
        if row["status"] == "no converged stable runs":
            print(f"N={row['N']}: no converged stable runs")
            continue
        print(
            f"N={row['N']:>4}: predicted={row['omega_predicted']:.12f} "
            f"pred_sweeps={row['predicted_sweeps']} "
            f"best={row['best_omega']:.12f} "
            f"best_sweeps={row['best_sweeps']} "
            f"offset={row['best_omega_offset']:+.6g} "
            f"status={row['status']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep omega around 1/(1+sin(pi/(M+1))) for the 2D two-blob "
            "Poisson h_unsplit default method."
        )
    )
    parser.add_argument("--Ns", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--cfl", type=float, default=1.0)
    parser.add_argument("--half-width", type=float, default=0.02)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--iter-max", type=int, default=5000)
    parser.add_argument("--poisson-problem", default="two_blob")
    parser.add_argument("--laplacian-stencil", default="centered")
    parser.add_argument("--ghost-layers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for N in args.Ns:
        omega_pred = predicted_omega(N)
        for omega in centered_omega_values(omega_pred, args.half_width, args.step):
            config = make_config(
                N=N,
                cfl=args.cfl,
                omega=omega,
                iter_max=args.iter_max,
                poisson_problem=args.poisson_problem,
                laplacian_stencil=args.laplacian_stencil,
                ghost_layers=args.ghost_layers,
            )
            rows.append(run_case(config, omega_pred))

    summary = summarize(rows)
    output_dir = args.output_dir
    write_csv(output_dir / "omega_sweep.csv", rows)
    with (output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print_summary(summary)
    print()
    print(f"wrote {output_dir / 'omega_sweep.csv'}")
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
