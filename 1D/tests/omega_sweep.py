"""Run one solver over a uniform range of relaxation parameters."""

import argparse
import contextlib
import csv
import io
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config1D import Config1D, Solver1DConfig
from solver1D import run_simulation


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = BASE_DIR / "results" / "studies" / "omega_sweeps" / "latest.csv"

CFL = 0.8
V = 0.5
TOL_UPDATE = 1e-7
ITER_MAX = 50000

METHODS = {
    ("linear", "Jacobi"): "jacobi_linear_parabolic",
    ("linear", "GS"): "gs_linear_parabolic",
    ("linear", "SORp"): "sor_linear_parabolic",
    ("linear", "SORh"): "sor_linear_hyperbolic",
    ("nonlinear", "Jacobi"): "jacobi_nonlinear_parabolic",
    ("nonlinear", "GS"): "gs_nonlinear_parabolic",
    ("nonlinear", "SORp"): "sor_nonlinear_parabolic",
    ("nonlinear", "SORh"): "sor_nonlinear_hyperbolic",
}

FIELDNAMES = (
    "problem",
    "method_label",
    "N",
    "interior_unknowns",
    "cfl",
    "omega",
    "converged",
    "stable",
    "reason",
    "iter_count",
    "elapsed_sec",
    "final_update",
)


def run_case(
    problem,
    method_label,
    N,
    omega,
    *,
    cfl=CFL,
    v=V,
    tol_update=TOL_UPDATE,
    iter_max=ITER_MAX,
):
    config = Config1D(
        problem=problem,
        method=METHODS[(problem, method_label)],
        N=N,
        cfl=cfl,
        omega=float(omega),
        use_omega_opt=False,
        iter_max=iter_max,
        norm=2,
        solver_config=Solver1DConfig(tol_update=tol_update),
        plot_enabled=False,
        linear_v=v,
        nonlinear_vmax=v,
        hyperbolic_linear_v=v,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, _, record = run_simulation(config)

    return {
        "problem": problem,
        "method_label": method_label,
        "N": N,
        "interior_unknowns": N - 1,
        "cfl": cfl,
        "omega": float(omega),
        "converged": record.results.converged,
        "stable": record.results.stable,
        "reason": record.results.reason,
        "iter_count": record.results.iter_count,
        "elapsed_sec": record.results.elapsed_sec,
        "final_update": record.results.residual_final,
    }


def run_sweep(problem, method_label, N, omega_min, omega_max, omega_step, **case_options):
    omegas = np.arange(omega_min, omega_max + 0.5 * omega_step, omega_step)
    return [
        run_case(
            problem,
            method_label,
            N,
            round(float(omega), 12),
            **case_options,
        )
        for omega in omegas
    ]


def successful(row):
    return bool(row["converged"] and row["stable"])


def first_stable_range(rows):
    """Return the first contiguous group of stable, converged samples."""
    band = []
    for row in sorted(rows, key=lambda item: item["omega"]):
        if successful(row):
            band.append(row)
        elif band:
            break
    return band


def best_stable_row(rows):
    band = first_stable_range(rows)
    if not band:
        raise RuntimeError("The omega sweep has no stable, converged range.")
    return min(band, key=lambda row: (row["iter_count"], row["omega"]))


def rows_for(rows, problem, method_label, N):
    return [
        row for row in rows
        if row["problem"] == problem
        and row["method_label"] == method_label
        and row["N"] == N
    ]


def write_rows(path, rows, fieldnames=FIELDNAMES):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open() as fh:
        raw_rows = list(csv.DictReader(fh))

    rows = []
    for raw in raw_rows:
        rows.append(
            {
                "problem": raw["problem"],
                "method_label": raw["method_label"],
                "N": int(raw["N"]),
                "interior_unknowns": int(raw.get("interior_unknowns") or int(raw["N"]) - 1),
                "cfl": float(raw["cfl"]),
                "omega": float(raw.get("omega") or raw["omega_experimental"]),
                "converged": raw["converged"] == "True",
                "stable": raw["stable"] == "True",
                "reason": raw["reason"],
                "iter_count": int(raw["iter_count"]),
                "elapsed_sec": float(raw["elapsed_sec"]),
                "final_update": float(raw["final_update"]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=("linear", "nonlinear"))
    parser.add_argument("method", choices=("SORp", "SORh"))
    parser.add_argument("N", type=int)
    parser.add_argument("omega_min", type=float)
    parser.add_argument("omega_max", type=float)
    parser.add_argument("omega_step", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = run_sweep(
        args.problem,
        args.method,
        args.N,
        args.omega_min,
        args.omega_max,
        args.omega_step,
    )
    write_rows(args.output, rows)
    best = best_stable_row(rows)
    print(f"best omega: {best['omega']:.6g} ({best['iter_count']} iterations)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
