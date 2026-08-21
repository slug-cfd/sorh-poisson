"""Reoptimize N=32 SORp for the nonlinear ``u0=x`` initial guess.

This leaves the original initial-guess diagnostic unchanged.  It searches the
first stable omega branch for the linear-in-x guess, then compares the selected
result with both cases from the original fixed-omega comparison.
"""

import argparse
from pathlib import Path

from .common import (
    EXPLORATORY_MAX_SWEEPS,
    FINAL_MAX_SWEEPS,
    formula_reference,
    load_baseline_omegas,
    make_config,
    mark_search_summary,
    omega_values,
    run_case,
    run_omega_candidates,
    save_study,
    select_first_converged_branch,
    solution_difference,
)


N = 32
CFL = 0.8
METHOD = "p_unsplit"
SEARCH_GUESS = "linear_x"


def run_until_branch_end(candidates, *, stage, max_sweeps):
    """Evaluate ordered candidates through the first failed post-success run."""
    rows = []
    saw_converged = False
    for omega in sorted(candidates):
        candidate_rows = run_omega_candidates(
            METHOD,
            N,
            [omega],
            stage=stage,
            cfl=CFL,
            initial_guess=SEARCH_GUESS,
            max_sweeps=max_sweeps,
            extra_metadata={"comparison_role": "linear_x_omega_search"},
        )
        row = candidate_rows[0]
        rows.append(row)
        succeeded = bool(row["converged"] and row["stable"])
        if succeeded:
            saw_converged = True
        elif saw_converged:
            print(
                "stopping this search stage after the first failed "
                "post-convergence candidate"
            )
            break
    return rows


def comparison_metadata(role, initial_guess, omega):
    reference = formula_reference(METHOD, N, CFL)
    return {
        "stage": "initial_guess_selected_comparison",
        "comparison_role": role,
        "formula_reference": reference,
        "omega_delta": float(omega) - reference,
        "epsilon": 0.0,
        "coefficient_variant": "regularized",
        "ellipticity_ratio": 0.0,
        "initial_guess": initial_guess,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omega-results",
        type=Path,
        help="baseline omega result directory or best_by_case.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--coarse-lower", type=float, default=2.0)
    parser.add_argument("--coarse-upper", type=float, default=2.8)
    parser.add_argument("--coarse-step", type=float, default=0.1)
    parser.add_argument("--fine-step", type=float, default=0.02)
    parser.add_argument(
        "--search-max-sweeps",
        type=int,
        default=6_000,
        help="candidate-run cap; the stagnation safeguard may stop earlier",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.coarse_step <= 0.0 or args.fine_step <= 0.0:
        raise ValueError("coarse-step and fine-step must be positive.")
    if args.coarse_lower >= args.coarse_upper:
        raise ValueError("coarse-lower must be smaller than coarse-upper.")
    if args.search_max_sweeps <= 0:
        raise ValueError("search-max-sweeps must be positive.")

    baseline_omegas, omega_source = load_baseline_omegas(args.omega_results)
    baseline_omega = baseline_omegas[(N, METHOD)]
    reference = formula_reference(METHOD, N, CFL)

    coarse_candidates = omega_values(
        args.coarse_lower,
        args.coarse_upper,
        args.coarse_step,
        (reference, baseline_omega),
    )
    search_rows = run_until_branch_end(
        coarse_candidates,
        stage="initial_guess_omega_coarse",
        max_sweeps=args.search_max_sweeps,
    )
    coarse_best = select_first_converged_branch(search_rows)

    center = float(coarse_best["omega"])
    failed_above_center = sorted(
        float(row["omega"])
        for row in search_rows
        if float(row["omega"]) > center
        and not (row["converged"] and row["stable"])
    )
    fine_upper = min(
        center + args.coarse_step,
        failed_above_center[0] if failed_above_center else float("inf"),
    )
    existing = {round(float(row["omega"]), 12) for row in search_rows}
    fine_candidates = [
        value
        for value in omega_values(
            center - args.coarse_step,
            fine_upper,
            args.fine_step,
            (center,),
        )
        if value not in existing and value > 0.0
    ]
    if fine_candidates:
        search_rows.extend(
            run_until_branch_end(
                fine_candidates,
                stage="initial_guess_omega_fine",
                max_sweeps=args.search_max_sweeps,
            )
        )

    selected = mark_search_summary(
        select_first_converged_branch(search_rows),
        [float(row["omega"]) for row in search_rows],
    )
    selected_omega = float(selected["omega"])
    failed_above_selected = sorted(
        float(row["omega"])
        for row in search_rows
        if float(row["omega"]) > selected_omega
        and not (row["converged"] and row["stable"])
    )
    first_failed_omega = (
        failed_above_selected[0] if failed_above_selected else None
    )

    all_rows = list(search_rows)
    comparison_rows = []
    solutions = {}
    cases = (
        ("zero_guess_optimized", "zero", baseline_omega),
        ("linear_x_reused_zero_omega", "linear_x", baseline_omega),
        ("linear_x_reoptimized", "linear_x", selected_omega),
    )
    for role, initial_guess, omega in cases:
        config = make_config(
            METHOD,
            N,
            omega,
            cfl=CFL,
            initial_guess=initial_guess,
        )
        row, solution = run_case(
            config,
            comparison_metadata(role, initial_guess, omega),
            FINAL_MAX_SWEEPS,
        )
        all_rows.append(row)
        comparison_rows.append(row)
        solutions[role] = solution

    zero_row = comparison_rows[0]
    zero_solution = solutions["zero_guess_optimized"]
    for row in comparison_rows:
        role = row["comparison_role"]
        if row["converged"] and zero_row["converged"]:
            row["sweep_ratio_to_zero_guess"] = (
                float(row["sweeps"]) / float(zero_row["sweeps"])
            )
            config = make_config(
                METHOD,
                N,
                float(row["omega"]),
                cfl=CFL,
                initial_guess=row["initial_guess"],
            )
            row["solution_difference_from_zero_guess"] = solution_difference(
                solutions[role],
                zero_solution,
                config,
            )
        else:
            row["sweep_ratio_to_zero_guess"] = None
            row["solution_difference_from_zero_guess"] = None

    output = save_study(
        "initial_sorp_optimized",
        all_rows,
        comparison_rows,
        {
            "description": "N=32 SORp initial-guess comparison with local omega optimization",
            "omega_results": omega_source,
            "ordering": "lexicographic",
            "N": N,
            "cfl": CFL,
            "coefficient": "normalized epsilon=0 (original coefficient)",
            "search_initial_guess": SEARCH_GUESS,
            "coarse_range": [args.coarse_lower, args.coarse_upper],
            "coarse_step": args.coarse_step,
            "fine_step": args.fine_step,
            "search_max_sweeps": args.search_max_sweeps,
            "selection_policy": "best run on first contiguous converged branch",
            "selected_linear_x_omega": selected_omega,
            "first_failed_omega_above_selection": first_failed_omega,
            "search_boundary_hit": selected["boundary_hit"],
        },
        args.output_dir,
        summary_name="comparison.csv",
    )

    print("\nN=32 SORp optimized initial-guess comparison")
    for row in comparison_rows:
        ratio = row["sweep_ratio_to_zero_guess"]
        ratio_text = "n/a" if ratio is None else f"{float(ratio):.3f}"
        print(
            f"{row['comparison_role']}: omega={float(row['omega']):.6g} "
            f"sweeps={int(row['sweeps'])} converged={row['converged']} "
            f"ratio_to_zero={ratio_text} reason={row['reason']} "
            f"residual={float(row['residual_final']):.6e}"
        )
    print(
        f"selected linear_x omega={selected_omega:.6g}; "
        f"first_failed_omega={first_failed_omega}; "
        f"search_boundary_hit={selected['boundary_hit']}"
    )
    print(f"results: {output}")


if __name__ == "__main__":
    main()
