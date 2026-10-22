"""Collect modular 2D nonlinear SOR parameter-study data.

Run from ``2D`` as ``python -m tests.nonlinear_studies.collect STUDY`` where
STUDY is one of: omega, timestep, h-cfl, degeneracy, initial, performance.
"""

import argparse
from pathlib import Path

import numpy as np

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
from problems.nonlinear import ellipticity_ratio


METHODS = ("h_unsplit", "p_unsplit")
BASE_CFL = 0.8


def _refined_rows(method, N, coarse_rows, coarse_step, fine_step, **kwargs):
    coarse_best = select_first_converged_branch(coarse_rows)
    center = float(coarse_best["omega"])
    existing = {round(float(row["omega"]), 12) for row in coarse_rows}
    candidates = [
        value
        for value in omega_values(
            center - coarse_step,
            center + coarse_step,
            fine_step,
            (center,),
        )
        if value not in existing and value > 0.0
    ]
    return run_omega_candidates(
        method,
        N,
        candidates,
        stage="omega_fine",
        **kwargs,
    )


def collect_omega(output_dir=None):
    rows = []
    summary = []
    selected_32 = {}

    for method in METHODS:
        if method == "h_unsplit":
            lower, upper, coarse_step, fine_step = 0.70, 1.05, 0.025, 0.005
        else:
            lower, upper, coarse_step, fine_step = 1.80, 3.20, 0.10, 0.02
        reference = formula_reference(method, 32, BASE_CFL)
        candidates = omega_values(lower, upper, coarse_step, (reference,))
        method_rows = run_omega_candidates(
            method,
            32,
            candidates,
            stage="omega_coarse",
        )
        method_rows.extend(
            _refined_rows(
                method,
                32,
                method_rows,
                coarse_step,
                fine_step,
            )
        )
        rows.extend(method_rows)
        best = select_first_converged_branch(method_rows)
        selected_32[method] = float(best["omega"])
        summary.append(
            mark_search_summary(best, [row["omega"] for row in method_rows])
        )

    for N in (16, 64):
        for method in METHODS:
            reference = formula_reference(method, N, BASE_CFL)
            reference_32 = formula_reference(method, 32, BASE_CFL)
            center = selected_32[method] * reference / reference_32
            step = 0.01 if method == "h_unsplit" else 0.05
            candidates = sorted({
                round(center + offset * step, 12)
                for offset in range(-3, 4)
                if center + offset * step > 0.0
            } | {round(reference, 12)})
            method_rows = run_omega_candidates(
                method,
                N,
                candidates,
                stage="omega_grid_validation",
            )
            rows.extend(method_rows)
            best = select_first_converged_branch(method_rows)
            summary.append(mark_search_summary(best, candidates))

    return save_study(
        "omega",
        rows,
        sorted(summary, key=lambda row: (int(row["N"]), row["method"])),
        {
            "description": "Baseline nonlinear omega calibration",
            "ordering": "lexicographic",
            "cfl": BASE_CFL,
            "coefficient": "normalized epsilon=0 (original coefficient)",
            "initial_guess": "zero",
            "selection_policy": "best run on first contiguous converged branch",
        },
        output_dir,
    )


def collect_timestep(omega_results=None, output_dir=None):
    baseline, source = load_baseline_omegas(omega_results)
    rows = []
    summary = []
    for N in (32, 64):
        for method in METHODS:
            center = baseline[(N, method)]
            fixed_rows = run_omega_candidates(
                method,
                N,
                [center],
                stage="timestep_fixed",
                max_sweeps=FINAL_MAX_SWEEPS,
            )
            rows.extend(fixed_rows)
            fixed_best = dict(fixed_rows[0], candidates_tested=1, boundary_hit=False)
            summary.append(fixed_best)

            step = 0.01 if method == "h_unsplit" else 0.05
            candidates = [
                round(center + offset * step, 12)
                for offset in range(-4, 5)
                if center + offset * step > 0.0
            ]
            adaptive_rows = run_omega_candidates(
                method,
                N,
                candidates,
                stage="timestep_adaptive_local_omega",
                timestep_mode="adaptive",
                max_sweeps=FINAL_MAX_SWEEPS,
            )
            rows.extend(adaptive_rows)
            best = select_first_converged_branch(adaptive_rows)
            summary.append(mark_search_summary(best, candidates))

    return save_study(
        "timestep",
        rows,
        summary,
        {
            "description": "Fixed versus adaptive timestep comparison",
            "omega_results": source,
            "ordering": "lexicographic",
        },
        output_dir,
    )


def collect_h_cfl(omega_results=None, output_dir=None):
    baseline, source = load_baseline_omegas(omega_results)
    omega_0 = baseline[(32, "h_unsplit")]
    kappa_0 = BASE_CFL * omega_0
    pairs = {}
    for cfl in np.round(np.arange(0.5, 1.01, 0.1), 10):
        fixed_omega_key = (float(cfl), round(omega_0, 12))
        fixed_product_key = (float(cfl), round(float(kappa_0 / cfl), 12))
        pairs.setdefault(fixed_omega_key, []).append("fixed_omega")
        pairs.setdefault(fixed_product_key, []).append("fixed_omega_cfl")

    rows = []
    for (cfl, omega), paths in sorted(pairs.items()):
        config = make_config("h_unsplit", 32, omega, cfl=cfl)
        metadata = {
            "stage": "h_cfl_decomposition",
            "path": "|".join(paths),
            "formula_reference": formula_reference("h_unsplit", 32, cfl),
            "omega_delta": omega - formula_reference("h_unsplit", 32, cfl),
            "epsilon": 0.0,
            "coefficient_variant": "regularized",
            "ellipticity_ratio": 0.0,
            "initial_guess": "zero",
            "baseline_omega": omega_0,
            "baseline_omega_cfl": kappa_0,
        }
        row, _ = run_case(config, metadata, EXPLORATORY_MAX_SWEEPS)
        rows.append(row)

    return save_study(
        "h_cfl",
        rows,
        rows,
        {
            "description": "SORh propagation versus residual-injection study",
            "omega_results": source,
            "cfl_values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "ordering": "lexicographic",
        },
        output_dir,
        summary_name="h_cfl_comparison.csv",
        write_all_runs=False,
    )


def collect_degeneracy(omega_results=None, output_dir=None):
    baseline, source = load_baseline_omegas(omega_results)
    variants = [
        (f"epsilon_{epsilon:g}", "regularized", epsilon, ellipticity_ratio(epsilon))
        for epsilon in (0.0, 1e-3, 1e-2, 1e-1, 1.0)
    ]
    variants.append(("constant", "constant", 0.0, 1.0))
    rows = []
    summary = []

    for label, variant, epsilon, ratio in variants:
        for method in METHODS:
            nonlinear_center = baseline[(32, method)]
            linear_center = formula_reference(method, 32, BASE_CFL)
            center = linear_center + (1.0 - ratio) * (
                nonlinear_center - linear_center
            )
            step = 0.015 if method == "h_unsplit" else 0.05
            candidates = [
                round(center + offset * step, 12)
                for offset in range(-3, 4)
                if center + offset * step > 0.0
            ]
            method_rows = run_omega_candidates(
                method,
                32,
                candidates,
                stage="degeneracy_local_omega",
                epsilon=epsilon,
                coefficient_variant=variant,
                extra_metadata={"coefficient_label": label},
            )
            rows.extend(method_rows)
            best = select_first_converged_branch(method_rows)
            summary.append(mark_search_summary(best, candidates))

    return save_study(
        "degeneracy",
        rows,
        summary,
        {
            "description": "Coefficient degeneracy study",
            "omega_results": source,
            "epsilon_values": [0.0, 1e-3, 1e-2, 1e-1, 1.0],
            "constant_coefficient_control": 0.5,
            "coefficient_normalization": "v(1)=0.5",
            "ordering": "lexicographic",
        },
        output_dir,
    )


def collect_initial(omega_results=None, output_dir=None):
    baseline, source = load_baseline_omegas(omega_results)
    rows = []
    for N in (32, 64):
        for method in METHODS:
            for initial_guess in ("zero", "linear_x"):
                omega = baseline[(N, method)]
                config = make_config(
                    method,
                    N,
                    omega,
                    initial_guess=initial_guess,
                )
                metadata = {
                    "stage": "initial_guess",
                    "formula_reference": formula_reference(method, N, BASE_CFL),
                    "omega_delta": omega - formula_reference(method, N, BASE_CFL),
                    "epsilon": 0.0,
                    "coefficient_variant": "regularized",
                    "ellipticity_ratio": 0.0,
                    "initial_guess": initial_guess,
                }
                row, _ = run_case(config, metadata, FINAL_MAX_SWEEPS)
                rows.append(row)
    return save_study(
        "initial",
        rows,
        rows,
        {
            "description": "Zero versus u0=x initial-guess diagnostic",
            "omega_results": source,
            "ordering": "lexicographic",
        },
        output_dir,
        summary_name="initial_guess_comparison.csv",
        write_all_runs=False,
    )


def collect_performance(omega_results=None, output_dir=None):
    baseline, source = load_baseline_omegas(omega_results)
    rows = []
    solutions = {}
    configs = {}
    for N in (16, 32, 64):
        for method in METHODS:
            omega = baseline[(N, method)]
            config = make_config(method, N, omega)
            metadata = {
                "stage": "performance",
                "formula_reference": formula_reference(method, N, BASE_CFL),
                "omega_delta": omega - formula_reference(method, N, BASE_CFL),
                "epsilon": 0.0,
                "coefficient_variant": "regularized",
                "ellipticity_ratio": 0.0,
                "initial_guess": "zero",
            }
            row, U = run_case(config, metadata, FINAL_MAX_SWEEPS)
            rows.append(row)
            solutions[(N, method)] = U
            configs[N] = config

    by_key = {(int(row["N"]), row["method"]): row for row in rows}
    comparisons = []
    for N in (16, 32, 64):
        h_row = by_key[(N, "h_unsplit")]
        p_row = by_key[(N, "p_unsplit")]
        comparisons.append({
            "N": N,
            "sorh_omega": h_row["omega"],
            "sorp_omega": p_row["omega"],
            "sorh_sweeps": h_row["sweeps"],
            "sorp_sweeps": p_row["sweeps"],
            "sweep_speedup_sorh": p_row["sweeps"] / max(1, h_row["sweeps"]),
            "sorh_elapsed_sec": h_row["elapsed_sec"],
            "sorp_elapsed_sec": p_row["elapsed_sec"],
            "elapsed_speedup_sorh": p_row["elapsed_sec"] / max(
                1e-14, h_row["elapsed_sec"]
            ),
            "solution_difference": solution_difference(
                solutions[(N, "h_unsplit")],
                solutions[(N, "p_unsplit")],
                configs[N],
            ),
            "sorh_converged": h_row["converged"],
            "sorp_converged": p_row["converged"],
        })

    scaling = {}
    for method in METHODS:
        sizes = np.array((16, 32, 64), dtype=float)
        method_rows = [by_key[(N, method)] for N in (16, 32, 64)]
        if all(row["converged"] for row in method_rows):
            sweeps = np.array([row["sweeps"] for row in method_rows], dtype=float)
            scaling[method] = float(
                np.polyfit(np.log(sizes), np.log(sweeps), 1)[0]
            )
        else:
            scaling[method] = None

    return save_study(
        "performance",
        rows,
        comparisons,
        {
            "description": "Optimized baseline SORh versus SORp performance",
            "omega_results": source,
            "ordering": "lexicographic",
            "sweep_scaling_exponents": scaling,
        },
        output_dir,
        summary_name="performance_comparison.csv",
    )


STUDIES = {
    "omega": collect_omega,
    "timestep": collect_timestep,
    "h-cfl": collect_h_cfl,
    "degeneracy": collect_degeneracy,
    "initial": collect_initial,
    "performance": collect_performance,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--omega-results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.study == "omega":
        STUDIES[args.study](output_dir=args.output_dir)
    else:
        STUDIES[args.study](
            omega_results=args.omega_results,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
