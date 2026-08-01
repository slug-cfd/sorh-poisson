"""Collect the two fixed omega-sensitivity curves used by Figure 4."""

import argparse
import csv
from pathlib import Path

from omega_sweep import (
    best_stable_row,
    first_stable_range,
    read_rows,
    rows_for,
    run_sweep,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    BASE_DIR
    / "results"
    / "final_report_figure_data"
    / "consistency_rerun_20260801"
    / "figure4_sensitivity_converged.csv"
)

N = 129
CFL = 0.8
V = 0.5
TOL_UPDATE = 1e-7
ITER_MAX = 10000

SWEEPS = {
    "SORh": (0.721, 0.994, 0.001),
    "SORp": (2.264, 2.683, 0.001),
}

FIELDNAMES = (
    "method_label",
    "omega",
    "iter_count",
    "omega_experimental",
    "omega_predicted",
    "is_instability_cutoff",
)


def collect(sweeps=None):
    figure_rows = []
    for method_label, (omega_min, omega_max, omega_step) in SWEEPS.items():
        if sweeps is None:
            sweep = run_sweep(
                "nonlinear",
                method_label,
                N,
                omega_min,
                omega_max,
                omega_step,
                cfl=CFL,
                v=V,
                tol_update=TOL_UPDATE,
                iter_max=ITER_MAX,
            )
        else:
            sweep = rows_for(sweeps, "nonlinear", method_label, N)
        stable_range = first_stable_range(sweep)
        best = best_stable_row(sweep)

        for row in stable_range:
            figure_rows.append(
                {
                    "method_label": method_label,
                    "omega": row["omega"],
                    "iter_count": row["iter_count"],
                    "omega_experimental": best["omega"],
                    "omega_predicted": "",
                    "is_instability_cutoff": row is stable_range[-1],
                }
            )

        print(
            f"{method_label}: omega*={best['omega']:.6g}, "
            f"limit={stable_range[-1]['omega']:.6g}"
        )
    return figure_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega-data", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = collect(read_rows(args.omega_data) if args.omega_data else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
