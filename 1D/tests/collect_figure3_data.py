"""Collect new omega data in the completed thesis figure format."""

import argparse
import csv
import math
from pathlib import Path

from omega_sweep import best_stable_row, read_rows, rows_for, run_sweep, write_rows


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    BASE_DIR
    / "results"
    / "generated_report_data"
    / "figure3_omega_optima.csv"
)

N_VALUES = (17, 33, 65, 129, 257)
CFL = 0.8
V = 0.5
TOL_UPDATE = 1e-7
ITER_MAX = 50000

LINEAR_SWEEPS = {
    "SORh": (0.25, 1.00, 0.01),
    "SORp": (1.50, 3.00, 0.01),
}

NONLINEAR_SWEEPS = {
    "SORh": (0.70, 1.00, 0.005),
    "SORp": (2.20, 3.20, 0.005),
}

FIGURE4_SWEEPS = {
    "SORh": (0.721, 0.994, 0.001),
    "SORp": (2.264, 2.683, 0.001),
}
FIGURE4_N = 129
FIGURE4_ITER_MAX = 10000

FIELDNAMES = (
    "method_label",
    "N",
    "interior_unknowns",
    "omega_formula",
    "omega_linear",
    "omega_nonlinear",
)


def omega_formula(method_label, N):
    scale = 2.0 / CFL if method_label == "SORp" else 1.0
    return scale / (1.0 + math.sin(math.pi / N))


def collect_sweeps():
    rows = []
    for N in N_VALUES:
        for method_label in ("SORp", "SORh"):
            rows.extend(
                run_sweep(
                    "linear",
                    method_label,
                    N,
                    *LINEAR_SWEEPS[method_label],
                    cfl=CFL,
                    v=V,
                    tol_update=TOL_UPDATE,
                    iter_max=ITER_MAX,
                )
            )
            rows.extend(
                run_sweep(
                    "nonlinear",
                    method_label,
                    N,
                    *(
                        FIGURE4_SWEEPS[method_label]
                        if N == FIGURE4_N
                        else NONLINEAR_SWEEPS[method_label]
                    ),
                    cfl=CFL,
                    v=V,
                    tol_update=TOL_UPDATE,
                    iter_max=FIGURE4_ITER_MAX if N == FIGURE4_N else ITER_MAX,
                )
            )
    return rows


def collect_figure_rows(sweeps):
    rows = []
    for N in N_VALUES:
        for method_label in ("SORh", "SORp"):
            linear = best_stable_row(rows_for(sweeps, "linear", method_label, N))
            nonlinear = best_stable_row(rows_for(sweeps, "nonlinear", method_label, N))
            rows.append(
                {
                    "method_label": method_label,
                    "N": N,
                    "interior_unknowns": N - 1,
                    "omega_formula": omega_formula(method_label, N),
                    "omega_linear": linear["omega"],
                    "omega_nonlinear": nonlinear["omega"],
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
        write_rows(args.output.with_name("figure3_omega_sweeps.csv"), sweeps)

    rows = collect_figure_rows(sweeps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
