"""Find experimental two-blob optima with a local formula-centered search."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import (
    DEFAULT_SETTINGS,
    METHODS,
    formula_omega,
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


TRIAL_FIELDS = (
    "N",
    "method",
    "cfl",
    "omega_formula",
    "omega",
    "omega_delta",
    "direction",
    "step_index",
    "trial_order",
    "iterations",
    "converged",
    "stable",
    "reason",
    "residual_final",
    "update_final",
    "exact_final",
    "wall_time_sec",
    "cpu_time_sec",
    "seconds_per_iteration",
)

BEST_FIELDS = (
    "N",
    "method",
    "cfl",
    "omega_formula",
    "formula_iterations",
    "omega_selected",
    "omega_delta",
    "selected_iterations",
    "selected_residual_final",
    "selected_exact_final",
    "trials_tested",
    "search_step",
    "left_stop_reason",
    "right_stop_reason",
)


def _case_key(row):
    return int(float(row["N"])), row["method"], round(float(row["cfl"]), 12)


def _candidate_key(value):
    return round(float(value), 12)


def _select_best(rows):
    converged = [row for row in rows if successful(row)]
    if not converged:
        raise RuntimeError("No stable converged omega was found for this case.")
    return min(
        converged,
        key=lambda row: (
            int(float(row["iterations"])),
            abs(float(row["omega_delta"])),
            float(row["wall_time_sec"]),
        ),
    )


def search_case(settings, N, method, cfl, existing_rows, persist):
    search = settings["omega_search"]
    omega_formula = formula_omega(settings, method, N, cfl)
    step = max(
        float(search["minimum_step"]),
        abs(omega_formula) * float(search["relative_step"]),
    )
    patience = int(search["increase_patience"])
    max_steps = int(search["max_steps_per_side"])

    case_rows = [
        row for row in existing_rows
        if _case_key(row) == (int(N), method, round(float(cfl), 12))
    ]
    cached = {_candidate_key(row["omega"]): row for row in case_rows}
    next_order = 1 + max(
        (int(float(row["trial_order"])) for row in case_rows),
        default=-1,
    )

    def evaluate(omega, direction, step_index):
        nonlocal next_order
        omega = _candidate_key(omega)
        if omega in cached:
            return cached[omega]

        print(
            f"omega search: N={N} method={method} CFL={cfl:g} "
            f"omega={omega:.12g}"
        )
        config = make_config(settings, method, N, cfl, omega)
        run, _, _ = run_poisson_case(config)
        row = {
            "N": run["N"],
            "method": run["method"],
            "cfl": run["cfl"],
            "omega_formula": omega_formula,
            "omega": run["omega"],
            "omega_delta": run["omega"] - omega_formula,
            "direction": direction,
            "step_index": step_index,
            "trial_order": next_order,
            "iterations": run["iterations"],
            "converged": run["converged"],
            "stable": run["stable"],
            "reason": run["reason"],
            "residual_final": run["residual_final"],
            "update_final": run["update_final"],
            "exact_final": run["exact_final"],
            "wall_time_sec": run["wall_time_sec"],
            "cpu_time_sec": run["cpu_time_sec"],
            "seconds_per_iteration": run["seconds_per_iteration"],
        }
        next_order += 1
        cached[omega] = row
        case_rows.append(row)
        existing_rows.append(row)
        persist()
        return row

    formula_row = evaluate(omega_formula, "formula", 0)
    formula_iterations = (
        int(float(formula_row["iterations"])) if successful(formula_row) else math.inf
    )
    state = {
        -1: {"active": True, "non_improving": 0, "best": formula_iterations},
        1: {"active": True, "non_improving": 0, "best": formula_iterations},
    }
    stop_reasons = {-1: "max_steps", 1: "max_steps"}

    for step_index in range(1, max_steps + 1):
        for direction in (-1, 1):
            side = state[direction]
            if not side["active"]:
                continue
            omega = omega_formula + direction * step_index * step
            if omega <= 0.0:
                side["active"] = False
                stop_reasons[direction] = "nonpositive_omega"
                continue

            row = evaluate(
                omega,
                "left" if direction < 0 else "right",
                step_index,
            )
            if not successful(row):
                side["active"] = False
                stop_reasons[direction] = f"{row['reason']}"
                continue

            iterations = int(float(row["iterations"]))
            if iterations < side["best"]:
                side["best"] = iterations
                side["non_improving"] = 0
            else:
                side["non_improving"] += 1
                if side["non_improving"] >= patience:
                    side["active"] = False
                    stop_reasons[direction] = "iteration_counts_non_improving"

        if not any(side["active"] for side in state.values()):
            break

    best = _select_best(case_rows)
    return {
        "N": int(N),
        "method": method,
        "cfl": float(cfl),
        "omega_formula": omega_formula,
        "formula_iterations": (
            int(float(formula_row["iterations"])) if successful(formula_row) else ""
        ),
        "omega_selected": float(best["omega"]),
        "omega_delta": float(best["omega"]) - omega_formula,
        "selected_iterations": int(float(best["iterations"])),
        "selected_residual_final": float(best["residual_final"]),
        "selected_exact_final": float(best["exact_final"]),
        "trials_tested": len(case_rows),
        "search_step": step,
        "left_stop_reason": stop_reasons[-1],
        "right_stop_reason": stop_reasons[1],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--Ns", type=int, nargs="+")
    parser.add_argument("--cfls", type=float, nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    settings, settings_path = load_settings(args.settings)
    Ns = args.Ns or [int(value) for value in settings["grid_sizes"]]
    cfls = args.cfls or [float(value) for value in settings["cfl_values"]]
    output_dir = prepare_output_dir("omega", args.output_dir, args.resume)
    trials_path = output_dir / "omega_trials.csv"
    best_path = output_dir / "best_omega.csv"
    trials = read_csv(trials_path) if args.resume else []
    summaries = read_csv(best_path) if args.resume else []
    completed = {_case_key(row) for row in summaries}

    write_global_metadata(
        output_dir,
        "two_blob_omega_search",
        settings,
        settings_path,
        {"N": Ns, "method": list(METHODS), "cfl": cfls},
        {
            **settings["omega_search"],
            "formula_implementation": "configuration.compute_omega_opt",
            "selection_metric": "minimum converged iteration count",
            "selection_tiebreakers": ["distance from formula", "wall time"],
            "search_order": "formula, then alternating left/right increments",
        },
    )

    def persist_trials():
        ordered = sort_rows(trials, ("N", "method", "cfl", "trial_order"))
        write_csv(trials_path, ordered, TRIAL_FIELDS)

    for N in Ns:
        for method in METHODS:
            for cfl in cfls:
                key = (int(N), method, round(float(cfl), 12))
                if key in completed:
                    continue
                summary = search_case(
                    settings,
                    N,
                    method,
                    cfl,
                    trials,
                    persist_trials,
                )
                summaries.append(summary)
                completed.add(key)
                write_csv(
                    best_path,
                    sort_rows(summaries, ("N", "method", "cfl")),
                    BEST_FIELDS,
                )

    persist_trials()
    write_csv(
        best_path,
        sort_rows(summaries, ("N", "method", "cfl")),
        BEST_FIELDS,
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
