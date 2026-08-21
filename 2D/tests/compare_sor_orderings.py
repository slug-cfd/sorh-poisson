"""Manual omega/performance comparison for Poisson SORh and SORp orderings.

Run from ``2D`` with::

    uv run python tests/compare_sor_orderings.py

The default coarse sweep is refined independently around the best omega found
for each method, ordering, and grid.  Use ``--h-bounds`` or ``--p-bounds`` if
``boundary_hit`` is true in the summary.
"""

import argparse
import contextlib
import csv
import io
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
    ParabolicUnsplitConfig,
    ProblemConfig,
    RunConfig,
    initialize_simulation,
)
from problems import poisson


CASES = (
    ("SORh", "lexicographic", "h_unsplit", "default"),
    ("SORh", "red_black", "h_unsplit", "red_black"),
    ("SORp", "lexicographic", "p_unsplit", "default"),
    ("SORp", "red_black", "p_unsplit", "default"),
)

SWEEP_FIELDS = (
    "family", "ordering", "mode", "N", "cfl", "c_max",
    "omega_reference", "omega", "stage", "sweeps", "elapsed_sec",
    "seconds_per_sweep", "converged", "stable", "reason", "update_final",
    "residual_final", "exact_final",
)


def omega_reference(family, N, cfl):
    c_max = math.cos(math.pi / (N + 1))
    denominator = 1.0 + math.sqrt(max(0.0, 1.0 - c_max * c_max))
    omega = 1.0 / denominator
    if family == "SORp":
        omega = 2.0 / (cfl * denominator)
    return omega, c_max


def omega_grid(lower, upper, step, extra=()):
    values = list(np.arange(lower, upper + 0.5 * step, step))
    values.extend((lower, upper, *extra))
    return sorted({round(float(value), 12) for value in values
                   if lower <= value <= upper})


def make_config(method, mode, ordering, N, cfl, omega, iter_max):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case="two_blob"),
            grid=GridConfig(Nx=N, Ny=N),
            run=RunConfig(
                method=method,
                cfl=cfl,
                omega=omega,
                iter_max=iter_max,
                norm="fro",
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(mode=mode),
                p_unsplit=ParabolicUnsplitConfig(gs_ordering=ordering),
            ),
        )
    )


def final_value(values):
    return float(values[-1]) if len(values) else math.nan


def run_case(family, ordering, method, mode, N, cfl, omega, reference,
             c_max, stage, iter_max):
    config = make_config(method, mode, ordering, N, cfl, omega, iter_max)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, histories = poisson.solve_all(config)
    elapsed = time.perf_counter() - start

    return {
        "family": family,
        "ordering": ordering,
        "mode": mode,
        "N": N,
        "cfl": cfl,
        "c_max": c_max,
        "omega_reference": reference,
        "omega": omega,
        "stage": stage,
        "sweeps": len(histories["update"]),
        "elapsed_sec": elapsed,
        "seconds_per_sweep": elapsed / max(1, len(histories["update"])),
        "converged": histories["converged"],
        "stable": histories["stable"],
        "reason": histories["reason"],
        "update_final": final_value(histories["update"]),
        "residual_final": final_value(histories["residual"]),
        "exact_final": final_value(histories["exact"]),
    }


def best_result(rows):
    def rank(row):
        residual = row["residual_final"]
        residual = residual if math.isfinite(residual) else math.inf
        if row["converged"]:
            return 0, row["sweeps"], residual, row["elapsed_sec"]
        if row["stable"]:
            return 1, residual, row["sweeps"], row["elapsed_sec"]
        return 2, residual, row["sweeps"], row["elapsed_sec"]

    return min(rows, key=rank)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Ns", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--h-bounds", type=float, nargs=2, default=(0.7, 1.1))
    parser.add_argument("--p-bounds", type=float, nargs=2, default=(1.8, 2.7))
    parser.add_argument("--h-step", type=float, default=0.025)
    parser.add_argument("--p-step", type=float, default=0.05)
    parser.add_argument("--refine-factor", type=int, default=5)
    parser.add_argument("--iter-max", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(N < 16 or N > 64 for N in args.Ns):
        raise ValueError("All grid sizes must satisfy 16 <= N <= 64.")
    if args.h_bounds[0] >= args.h_bounds[1] or args.p_bounds[0] >= args.p_bounds[1]:
        raise ValueError("Each omega lower bound must be below its upper bound.")
    if args.h_step <= 0.0 or args.p_step <= 0.0:
        raise ValueError("Omega sweep steps must be positive.")
    if args.refine_factor < 2:
        raise ValueError("refine-factor must be at least 2.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (
        Path(__file__).resolve().parents[1]
        / "results" / "sor_ordering_comparison" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "omega_sweep.csv"
    summary_rows = []

    with sweep_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SWEEP_FIELDS)
        writer.writeheader()

        for N in args.Ns:
            for family, ordering, method, mode in CASES:
                bounds = args.h_bounds if family == "SORh" else args.p_bounds
                coarse_step = args.h_step if family == "SORh" else args.p_step
                lower, upper = bounds
                reference, c_max = omega_reference(family, N, args.cfl)
                coarse = omega_grid(lower, upper, coarse_step, (reference,))
                case_rows = []

                for stage, candidates in (("coarse", coarse),):
                    print(f"N={N:>2} {family} {ordering:<13} {stage}")
                    for omega in candidates:
                        row = run_case(
                            family, ordering, method, mode, N, args.cfl,
                            omega, reference, c_max, stage, args.iter_max,
                        )
                        writer.writerow(row)
                        stream.flush()
                        case_rows.append(row)

                coarse_best = best_result(case_rows)
                boundary_hit = math.isclose(coarse_best["omega"], lower) or math.isclose(
                    coarse_best["omega"], upper
                )
                if coarse_best["converged"] and not boundary_hit:
                    fine_step = coarse_step / args.refine_factor
                    fine_lower = max(lower, coarse_best["omega"] - coarse_step)
                    fine_upper = min(upper, coarse_best["omega"] + coarse_step)
                    tested = {row["omega"] for row in case_rows}
                    fine = [value for value in omega_grid(
                        fine_lower, fine_upper, fine_step, (reference,)
                    ) if value not in tested]
                    print(f"N={N:>2} {family} {ordering:<13} refine")
                    for omega in fine:
                        row = run_case(
                            family, ordering, method, mode, N, args.cfl,
                            omega, reference, c_max, "refine", args.iter_max,
                        )
                        writer.writerow(row)
                        stream.flush()
                        case_rows.append(row)

                best = best_result(case_rows)
                summary_rows.append({
                    **best,
                    "omega_delta": best["omega"] - reference,
                    "search_lower": lower,
                    "search_upper": upper,
                    "boundary_hit": (
                        math.isclose(best["omega"], lower)
                        or math.isclose(best["omega"], upper)
                    ),
                    "candidates_tested": len(case_rows),
                })

    summary_path = output_dir / "best_by_case.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "settings.json").open("w") as stream:
        json.dump(vars(args), stream, indent=2, default=str)

    print(f"\nFull sweep: {sweep_path}")
    print(f"Best cases: {summary_path}")


if __name__ == "__main__":
    main()
