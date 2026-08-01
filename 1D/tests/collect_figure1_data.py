"""Collect linear method performance data for Figure 1."""

import argparse
import csv
from pathlib import Path

from omega_sweep import best_stable_row, read_rows, rows_for, run_case, run_sweep, write_rows


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    BASE_DIR
    / "results"
    / "final_report_figure_data"
    / "consistency_rerun_20260801"
    / "figure1_linear_performance.csv"
)

N_VALUES = (17, 33, 65, 129, 257)
CFL = 0.8
V = 0.5
TOL_UPDATE = 1e-7
ITER_MAX = 200000

SWEEPS = {
    "SORh": (0.25, 1.00, 0.01),
    "SORp": (1.50, 3.00, 0.01),
}

FIELDNAMES = (
    "method_label",
    "interior_unknowns",
    "omega",
    "iter_count",
    "elapsed_sec",
)


def collect_sweeps():
    rows = []
    for N in N_VALUES:
        for method_label, omega_range in SWEEPS.items():
            rows.extend(
                run_sweep(
                    "linear",
                    method_label,
                    N,
                    *omega_range,
                    cfl=CFL,
                    v=V,
                    tol_update=TOL_UPDATE,
                    iter_max=ITER_MAX,
                )
            )
    return rows


def collect_performance(sweeps):
    rows = []
    for N in N_VALUES:
        for method_label in ("Jacobi", "GS", "SORp", "SORh"):
            if method_label in SWEEPS:
                result = best_stable_row(rows_for(sweeps, "linear", method_label, N))
                omega = result["omega"]
            else:
                omega = 1.0
                result = run_case(
                    "linear",
                    method_label,
                    N,
                    omega,
                    cfl=CFL,
                    v=V,
                    tol_update=TOL_UPDATE,
                    iter_max=ITER_MAX,
                )
            rows.append(
                {
                    "method_label": method_label,
                    "interior_unknowns": N - 1,
                    "omega": omega,
                    "iter_count": result["iter_count"],
                    "elapsed_sec": result["elapsed_sec"],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega-data", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sweeps = (
        [row for path in args.omega_data for row in read_rows(path)]
        if args.omega_data
        else collect_sweeps()
    )
    if not args.omega_data:
        write_rows(args.output.with_name("figure1_omega_sweeps.csv"), sweeps)

    rows = collect_performance(sweeps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
