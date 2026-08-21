"""Calibrate and compare the N=32 frozen-coefficient SORp study variant."""

import argparse
from pathlib import Path

from problems import nonlinear_diffusion

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


def solve_frozen_sorp(config):
    """Use the canonical SORp dispatcher with frozen coefficients."""
    if config.p_unsplit_config.coefficient_mode != "frozen_per_sweep":
        raise ValueError("Frozen SORp study requires frozen_per_sweep mode.")
    return nonlinear_diffusion.solve_all(config)


def baseline_metadata(method, variant):
    reference = formula_reference(method, N, CFL)
    return {
        "stage": "controlled_baseline",
        "variant": variant,
        "formula_reference": reference,
        "omega_delta": None,
        "epsilon": 0.0,
        "coefficient_variant": "regularized",
        "ellipticity_ratio": 0.0,
        "initial_guess": "zero",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omega-results",
        type=Path,
        help="baseline omega result directory or best_by_case.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--coarse-lower", type=float, default=1.8)
    parser.add_argument("--coarse-upper", type=float, default=3.2)
    parser.add_argument("--coarse-step", type=float, default=0.1)
    parser.add_argument("--fine-step", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_omegas, omega_source = load_baseline_omegas(args.omega_results)
    rows = []

    coarse_candidates = omega_values(
        args.coarse_lower,
        args.coarse_upper,
        args.coarse_step,
        (formula_reference("p_unsplit", N, CFL),),
    )
    frozen_rows = run_omega_candidates(
        "p_unsplit",
        N,
        coarse_candidates,
        stage="frozen_sorp_coarse",
        cfl=CFL,
        max_sweeps=EXPLORATORY_MAX_SWEEPS,
        coefficient_mode="frozen_per_sweep",
        extra_metadata={"variant": "p_unsplit_frozen"},
        solve_callback=solve_frozen_sorp,
    )
    rows.extend(frozen_rows)

    coarse_best = select_first_converged_branch(frozen_rows)
    center = float(coarse_best["omega"])
    existing = {round(float(row["omega"]), 12) for row in frozen_rows}
    fine_candidates = [
        value
        for value in omega_values(
            center - args.coarse_step,
            center + args.coarse_step,
            args.fine_step,
            (center,),
        )
        if value not in existing and value > 0.0
    ]
    if fine_candidates:
        fine_rows = run_omega_candidates(
            "p_unsplit",
            N,
            fine_candidates,
            stage="frozen_sorp_fine",
            cfl=CFL,
            max_sweeps=EXPLORATORY_MAX_SWEEPS,
            coefficient_mode="frozen_per_sweep",
            extra_metadata={"variant": "p_unsplit_frozen"},
            solve_callback=solve_frozen_sorp,
        )
        frozen_rows.extend(fine_rows)
        rows.extend(fine_rows)

    selected = mark_search_summary(
        select_first_converged_branch(frozen_rows),
        [float(row["omega"]) for row in frozen_rows],
    )

    solutions = {}
    final_rows = []
    for method, variant in (
        ("h_unsplit", "h_unsplit"),
        ("p_unsplit", "p_unsplit_mixed_coefficients"),
    ):
        omega = baseline_omegas[(N, method)]
        config = make_config(method, N, omega, cfl=CFL)
        metadata = baseline_metadata(method, variant)
        metadata["omega_delta"] = omega - metadata["formula_reference"]
        row, solution = run_case(config, metadata, FINAL_MAX_SWEEPS)
        rows.append(row)
        final_rows.append(row)
        solutions[variant] = solution

    frozen_config = make_config(
        "p_unsplit",
        N,
        float(selected["omega"]),
        cfl=CFL,
        coefficient_mode="frozen_per_sweep",
    )
    frozen_metadata = baseline_metadata("p_unsplit", "p_unsplit_frozen")
    frozen_metadata.update(
        stage="frozen_sorp_selected",
        omega_delta=(
            float(selected["omega"])
            - frozen_metadata["formula_reference"]
        ),
        search_lower=selected["search_lower"],
        search_upper=selected["search_upper"],
        boundary_hit=selected["boundary_hit"],
        candidates_tested=selected["candidates_tested"],
    )
    frozen_final, frozen_solution = run_case(
        frozen_config,
        frozen_metadata,
        FINAL_MAX_SWEEPS,
        solve_callback=solve_frozen_sorp,
    )
    rows.append(frozen_final)
    final_rows.append(frozen_final)
    solutions["p_unsplit_frozen"] = frozen_solution

    comparisons = []
    for row in final_rows:
        variant = row["variant"]
        comparison = dict(row)
        comparison["solution_difference_from_sorh"] = solution_difference(
            solutions[variant],
            solutions["h_unsplit"],
            frozen_config,
        )
        comparison["solution_difference_from_current_sorp"] = solution_difference(
            solutions[variant],
            solutions["p_unsplit_mixed_coefficients"],
            frozen_config,
        )
        comparisons.append(comparison)

    output = save_study(
        "frozen_sorp",
        rows,
        comparisons,
        {
            "description": "N=32 frozen-coefficient SORp causal control",
            "omega_results": omega_source,
            "ordering": "lexicographic",
            "N": N,
            "cfl": CFL,
            "coefficient": "normalized epsilon=0 (original coefficient)",
            "initial_guess": "zero",
            "frozen_policy": "all nonlinear face coefficients fixed per sweep",
            "selection_policy": "best run on first contiguous converged branch",
        },
        args.output_dir,
        summary_name="comparison.csv",
    )

    print("\nN=32 controlled comparison")
    for row in comparisons:
        print(
            f"{row['variant']}: omega={float(row['omega']):.6g} "
            f"sweeps={int(row['sweeps'])} "
            f"elapsed={float(row['elapsed_sec']):.3f}s "
            f"residual={float(row['residual_final']):.6e}"
        )
    print(f"results: {output}")


if __name__ == "__main__":
    main()
