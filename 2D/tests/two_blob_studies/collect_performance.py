"""Collect repeated timing and iteration-count data after omega selection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .common import (
    DEFAULT_SETTINGS,
    METHODS,
    RESULTS_ROOT,
    file_sha256,
    load_settings,
    make_config,
    prepare_output_dir,
    read_csv,
    run_poisson_case,
    sort_rows,
    successful,
    write_csv,
    write_global_metadata,
)


PERFORMANCE_FIELDS = (
    "N",
    "method",
    "cfl",
    "omega",
    "repetition",
    "execution_order",
    "iterations",
    "wall_time_sec",
    "cpu_time_sec",
    "seconds_per_iteration",
    "pseudo_time",
    "dt",
    "converged",
    "stable",
    "reason",
    "residual_final",
    "update_final",
)

PERFORMANCE_SUMMARY_FIELDS = (
    "N",
    "method",
    "cfl",
    "omega",
    "repetitions_completed",
    "converged_repetitions",
    "iterations_mean",
    "iterations_std",
    "iterations_min",
    "iterations_max",
    "wall_time_mean_sec",
    "wall_time_std_sec",
    "wall_time_min_sec",
    "wall_time_max_sec",
    "cpu_time_mean_sec",
    "cpu_time_std_sec",
    "seconds_per_iteration_mean",
    "seconds_per_iteration_std",
)

COMPARISON_FIELDS = (
    "N",
    "cfl",
    "sorh_omega",
    "sorp_omega",
    "sorh_iterations_mean",
    "sorp_iterations_mean",
    "iteration_ratio_sorp_over_sorh",
    "sorh_wall_time_mean_sec",
    "sorp_wall_time_mean_sec",
    "runtime_ratio_sorp_over_sorh",
)


def _case_key(row):
    return int(float(row["N"])), row["method"], round(float(row["cfl"]), 12)


def _run_key(row):
    return (*_case_key(row), int(float(row["repetition"])))


def latest_omega_results(path=None):
    if path is not None:
        candidate = Path(path)
        return (candidate / "best_omega.csv") if candidate.is_dir() else candidate
    matches = sorted((RESULTS_ROOT / "omega").glob("*/best_omega.csv"))
    if not matches:
        raise FileNotFoundError(
            "No two-blob omega results were found. Run collect_omega first or "
            "provide --omega-results."
        )
    return matches[-1]


def load_selected_omegas(path):
    rows = read_csv(path)
    selected = {
        _case_key(row): float(row["omega_selected"])
        for row in rows
    }
    return selected


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_performance(rows):
    summaries = []
    keys = sorted({_case_key(row) for row in rows})
    for N, method, cfl in keys:
        all_rows = [row for row in rows if _case_key(row) == (N, method, cfl)]
        group = [row for row in all_rows if successful(row)]
        if not group:
            summaries.append({
                "N": N,
                "method": method,
                "cfl": cfl,
                "omega": float(all_rows[0]["omega"]),
                "repetitions_completed": len(all_rows),
                "converged_repetitions": 0,
                "iterations_mean": math.nan,
                "iterations_std": math.nan,
                "iterations_min": math.nan,
                "iterations_max": math.nan,
                "wall_time_mean_sec": math.nan,
                "wall_time_std_sec": math.nan,
                "wall_time_min_sec": math.nan,
                "wall_time_max_sec": math.nan,
                "cpu_time_mean_sec": math.nan,
                "cpu_time_std_sec": math.nan,
                "seconds_per_iteration_mean": math.nan,
                "seconds_per_iteration_std": math.nan,
            })
            continue
        iterations = np.asarray(
            [float(row["iterations"]) for row in group], dtype=float
        )
        wall = np.asarray([float(row["wall_time_sec"]) for row in group])
        cpu = np.asarray([float(row["cpu_time_sec"]) for row in group])
        per_iteration = np.asarray(
            [float(row["seconds_per_iteration"]) for row in group]
        )
        summaries.append({
            "N": N,
            "method": method,
            "cfl": cfl,
            "omega": float(group[0]["omega"]),
            "repetitions_completed": len(all_rows),
            "converged_repetitions": len(group),
            "iterations_mean": float(np.mean(iterations)),
            "iterations_std": _sample_std(iterations),
            "iterations_min": int(np.min(iterations)),
            "iterations_max": int(np.max(iterations)),
            "wall_time_mean_sec": float(np.mean(wall)),
            "wall_time_std_sec": _sample_std(wall),
            "wall_time_min_sec": float(np.min(wall)),
            "wall_time_max_sec": float(np.max(wall)),
            "cpu_time_mean_sec": float(np.mean(cpu)),
            "cpu_time_std_sec": _sample_std(cpu),
            "seconds_per_iteration_mean": float(np.mean(per_iteration)),
            "seconds_per_iteration_std": _sample_std(per_iteration),
        })
    return summaries


def method_comparisons(summary_rows):
    summaries = {_case_key(row): row for row in summary_rows}
    result = []
    pairs = sorted({(key[0], key[2]) for key in summaries})
    for N, cfl in pairs:
        h_key = (N, "h_unsplit", cfl)
        p_key = (N, "p_unsplit", cfl)
        if h_key not in summaries or p_key not in summaries:
            continue
        h_row = summaries[h_key]
        p_row = summaries[p_key]
        h_iterations = float(h_row["iterations_mean"])
        p_iterations = float(p_row["iterations_mean"])
        h_time = float(h_row["wall_time_mean_sec"])
        p_time = float(p_row["wall_time_mean_sec"])
        result.append({
            "N": N,
            "cfl": cfl,
            "sorh_omega": float(h_row["omega"]),
            "sorp_omega": float(p_row["omega"]),
            "sorh_iterations_mean": h_iterations,
            "sorp_iterations_mean": p_iterations,
            "iteration_ratio_sorp_over_sorh": p_iterations / max(1.0, h_iterations),
            "sorh_wall_time_mean_sec": h_time,
            "sorp_wall_time_mean_sec": p_time,
            "runtime_ratio_sorp_over_sorh": p_time / max(1e-14, h_time),
        })
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--omega-results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--Ns", type=int, nargs="+")
    parser.add_argument("--cfls", type=float, nargs="+")
    parser.add_argument("--repetitions", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    settings, settings_path = load_settings(args.settings)
    Ns = args.Ns or [int(value) for value in settings["grid_sizes"]]
    cfls = args.cfls or [float(value) for value in settings["cfl_values"]]
    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else int(settings["performance"]["repetitions"])
    )
    if repetitions < 2:
        raise ValueError("Performance timing requires at least two repetitions.")

    omega_path = latest_omega_results(args.omega_results).resolve()
    selected_omegas = load_selected_omegas(omega_path)
    required = {
        (int(N), method, round(float(cfl), 12))
        for N in Ns for method in METHODS for cfl in cfls
    }
    missing = sorted(required - set(selected_omegas))
    if missing:
        raise KeyError(f"Selected omega data are missing cases: {missing}")

    output_dir = prepare_output_dir("performance", args.output_dir, args.resume)
    run_path = output_dir / "performance_runs.csv"
    runs = read_csv(run_path) if args.resume else []
    completed_runs = {_run_key(row) for row in runs}
    execution_order = 1 + max(
        (int(float(row["execution_order"])) for row in runs),
        default=0,
    )

    write_global_metadata(
        output_dir,
        "two_blob_performance",
        settings,
        settings_path,
        {
            "N": Ns,
            "method": list(METHODS),
            "cfl": cfls,
            "repetition": list(range(1, repetitions + 1)),
        },
        {
            "repetitions": repetitions,
            "timed_region": "poisson.solve_all only",
            "method_execution_order": "alternated by repetition",
            "omega_results": str(omega_path),
            "omega_results_sha256": file_sha256(omega_path),
        },
    )

    def checkpoint():
        write_csv(
            run_path,
            sort_rows(runs, ("N", "cfl", "repetition", "execution_order")),
            PERFORMANCE_FIELDS,
        )

    for N in Ns:
        for cfl in cfls:
            for repetition in range(1, repetitions + 1):
                method_order = METHODS if repetition % 2 else tuple(reversed(METHODS))
                for method in method_order:
                    key = (int(N), method, round(float(cfl), 12), repetition)
                    if key in completed_runs:
                        continue
                    omega = selected_omegas[key[:3]]
                    print(
                        f"performance: N={N} method={method} CFL={cfl:g} "
                        f"repetition={repetition}/{repetitions}"
                    )
                    config = make_config(settings, method, N, cfl, omega)
                    run, _, _ = run_poisson_case(config)
                    row = {
                        "N": run["N"],
                        "method": run["method"],
                        "cfl": run["cfl"],
                        "omega": run["omega"],
                        "repetition": repetition,
                        "execution_order": execution_order,
                        "iterations": run["iterations"],
                        "wall_time_sec": run["wall_time_sec"],
                        "cpu_time_sec": run["cpu_time_sec"],
                        "seconds_per_iteration": run["seconds_per_iteration"],
                        "pseudo_time": run["pseudo_time"],
                        "dt": run["dt"],
                        "converged": run["converged"],
                        "stable": run["stable"],
                        "reason": run["reason"],
                        "residual_final": run["residual_final"],
                        "update_final": run["update_final"],
                    }
                    execution_order += 1
                    runs.append(row)
                    completed_runs.add(key)

                    checkpoint()

    checkpoint()
    summary_rows = summarize_performance(runs)
    comparison_rows = method_comparisons(summary_rows)
    write_csv(
        output_dir / "performance_summary.csv",
        sort_rows(summary_rows, ("N", "method", "cfl")),
        PERFORMANCE_SUMMARY_FIELDS,
    )
    write_csv(
        output_dir / "method_comparison.csv",
        sort_rows(comparison_rows, ("N", "cfl")),
        COMPARISON_FIELDS,
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
