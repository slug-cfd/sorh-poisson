"""Collect discrete-solution-validated accuracy data for the two-blob problem."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from .common import (
    DEFAULT_SETTINGS,
    METHODS,
    RESULTS_ROOT,
    accuracy_row,
    as_bool,
    file_sha256,
    load_settings,
    make_config,
    norm_metrics,
    prepare_output_dir,
    read_csv,
    run_poisson_case,
    sort_rows,
    successful,
    write_csv,
    write_global_metadata,
)


ACCURACY_FIELDS = (
    "N",
    "method",
    "cfl",
    "omega",
    "dx",
    "dy",
    "iterations",
    "converged",
    "stable",
    "reason",
    "accuracy_valid",
    "validation_reason",
    "error_l1",
    "error_rms",
    "error_linf",
    "residual_l1",
    "residual_rms",
    "residual_linf",
    "truncation_l1",
    "truncation_rms",
    "truncation_linf",
    "discrete_lambda_min",
    "algebraic_error_bound_rms",
    "algebraic_to_solution_error",
    "best_error_rms",
    "best_error_iter",
)

ORDER_FIELDS = (
    "method",
    "cfl",
    "N_coarse",
    "N_fine",
    "dx_coarse",
    "dx_fine",
    "solution_order_l1",
    "solution_order_rms",
    "solution_order_linf",
    "truncation_order_l1",
    "truncation_order_rms",
    "truncation_order_linf",
)

COMPARISON_FIELDS = (
    "N",
    "cfl",
    "sorh_omega",
    "sorp_omega",
    "sorh_iterations",
    "sorp_iterations",
    "sorh_accuracy_valid",
    "sorp_accuracy_valid",
    "both_accuracy_valid",
    "sorh_error_rms",
    "sorp_error_rms",
    "sorh_algebraic_bound_rms",
    "sorp_algebraic_bound_rms",
    "discrete_solution_difference_l1",
    "discrete_solution_difference_rms",
    "discrete_solution_difference_linf",
)


def _case_key(row):
    return int(float(row["N"])), row["method"], round(float(row["cfl"]), 12)


def _accuracy_valid(row):
    return as_bool(row["accuracy_valid"])


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
    return {
        _case_key(row): float(row["omega_selected"])
        for row in read_csv(path)
    }


def effective_accuracy_settings(settings):
    effective = deepcopy(settings)
    for field in ("tol_residual", "tol_update", "iter_max"):
        effective["solver"][field] = effective["accuracy"][field]
    return effective


def validate_accuracy(row, accuracy_settings):
    if not successful(row):
        return False, f"solver_{row['reason']}"
    if row["reason"] != "tol_residual":
        return False, f"unexpected_stop_{row['reason']}"
    residual_rms = float(row["residual_rms"])
    if not np.isfinite(residual_rms):
        return False, "nonfinite_residual"
    if residual_rms >= float(accuracy_settings["tol_residual"]):
        return False, "residual_above_accuracy_tolerance"
    algebraic_bound = float(row["algebraic_error_bound_rms"])
    algebraic_fraction = float(row["algebraic_to_solution_error"])
    if not np.isfinite(algebraic_bound) or not np.isfinite(algebraic_fraction):
        return False, "nonfinite_algebraic_error_bound"
    if algebraic_fraction > float(
        accuracy_settings["algebraic_error_fraction_max"]
    ):
        return False, "algebraic_error_not_negligible"
    return True, "validated_discrete_solution"


def observed_orders(rows):
    result = []
    valid = [row for row in rows if _accuracy_valid(row)]
    for method in METHODS:
        cfl_values = sorted({float(row["cfl"]) for row in valid})
        for cfl in cfl_values:
            group = sorted(
                [
                    row for row in valid
                    if row["method"] == method
                    and math.isclose(float(row["cfl"]), cfl)
                ],
                key=lambda row: int(float(row["N"])),
            )
            for coarse, fine in zip(group, group[1:]):
                dx_coarse = float(coarse["dx"])
                dx_fine = float(fine["dx"])
                denominator = math.log(dx_coarse / dx_fine)

                def order(field):
                    coarse_error = float(coarse[field])
                    fine_error = float(fine[field])
                    if coarse_error <= 0.0 or fine_error <= 0.0:
                        return math.nan
                    return math.log(coarse_error / fine_error) / denominator

                result.append({
                    "method": method,
                    "cfl": cfl,
                    "N_coarse": int(float(coarse["N"])),
                    "N_fine": int(float(fine["N"])),
                    "dx_coarse": dx_coarse,
                    "dx_fine": dx_fine,
                    "solution_order_l1": order("error_l1"),
                    "solution_order_rms": order("error_rms"),
                    "solution_order_linf": order("error_linf"),
                    "truncation_order_l1": order("truncation_l1"),
                    "truncation_order_rms": order("truncation_rms"),
                    "truncation_order_linf": order("truncation_linf"),
                })
    return result


def _cfl_token(cfl):
    return f"{float(cfl):.12g}".replace("-", "m").replace(".", "p")


def solution_path(output_dir, N, method, cfl):
    return (
        Path(output_dir)
        / "solutions"
        / f"N{int(N)}_{method}_cfl{_cfl_token(cfl)}.npz"
    )


def method_comparisons(rows, output_dir):
    by_key = {_case_key(row): row for row in rows}
    result = []
    pairs = sorted({(key[0], key[2]) for key in by_key})
    for N, cfl in pairs:
        h_key = (N, "h_unsplit", cfl)
        p_key = (N, "p_unsplit", cfl)
        if h_key not in by_key or p_key not in by_key:
            continue
        h_row = by_key[h_key]
        p_row = by_key[p_key]
        h_path = solution_path(output_dir, N, "h_unsplit", cfl)
        p_path = solution_path(output_dir, N, "p_unsplit", cfl)
        if not h_path.exists() or not p_path.exists():
            continue
        with np.load(h_path) as data:
            U_h = data["U_interior"]
        with np.load(p_path) as data:
            U_p = data["U_interior"]
        difference = norm_metrics(U_h - U_p)
        h_valid = _accuracy_valid(h_row)
        p_valid = _accuracy_valid(p_row)
        result.append({
            "N": N,
            "cfl": cfl,
            "sorh_omega": float(h_row["omega"]),
            "sorp_omega": float(p_row["omega"]),
            "sorh_iterations": int(float(h_row["iterations"])),
            "sorp_iterations": int(float(p_row["iterations"])),
            "sorh_accuracy_valid": h_valid,
            "sorp_accuracy_valid": p_valid,
            "both_accuracy_valid": h_valid and p_valid,
            "sorh_error_rms": float(h_row["error_rms"]),
            "sorp_error_rms": float(p_row["error_rms"]),
            "sorh_algebraic_bound_rms": float(
                h_row["algebraic_error_bound_rms"]
            ),
            "sorp_algebraic_bound_rms": float(
                p_row["algebraic_error_bound_rms"]
            ),
            "discrete_solution_difference_l1": difference["l1"],
            "discrete_solution_difference_rms": difference["rms"],
            "discrete_solution_difference_linf": difference["linf"],
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
    return parser.parse_args()


def main():
    args = parse_args()
    settings, settings_path = load_settings(args.settings)
    if settings["problem"]["laplacian_stencil"] != "centered":
        raise ValueError(
            "The discrete eigenvalue bound is implemented for the centered "
            "five-point Laplacian only."
        )
    effective_settings = effective_accuracy_settings(settings)
    Ns = args.Ns or [int(value) for value in settings["grid_sizes"]]
    cfls = args.cfls or [float(value) for value in settings["cfl_values"]]

    omega_path = latest_omega_results(args.omega_results).resolve()
    selected_omegas = load_selected_omegas(omega_path)
    required = {
        (int(N), method, round(float(cfl), 12))
        for N in Ns for method in METHODS for cfl in cfls
    }
    missing = sorted(required - set(selected_omegas))
    if missing:
        raise KeyError(f"Selected omega data are missing cases: {missing}")

    output_dir = prepare_output_dir("accuracy", args.output_dir, args.resume)
    accuracy_path = output_dir / "accuracy.csv"
    rows = read_csv(accuracy_path) if args.resume else []
    row_index = {_case_key(row): index for index, row in enumerate(rows)}

    write_global_metadata(
        output_dir,
        "two_blob_discrete_solution_accuracy",
        effective_settings,
        settings_path,
        {"N": Ns, "method": list(METHODS), "cfl": cfls},
        {
            "omega_results": str(omega_path),
            "omega_results_sha256": file_sha256(omega_path),
            "residual_requirement": settings["accuracy"]["tol_residual"],
            "update_stagnation_safeguard": settings["accuracy"]["tol_update"],
            "accuracy_iter_max": settings["accuracy"]["iter_max"],
            "algebraic_error_bound": "residual_rms / discrete_lambda_min",
            "algebraic_error_fraction_max": settings["accuracy"][
                "algebraic_error_fraction_max"
            ],
            "accuracy_validity": (
                "tol_residual stop and algebraic bound no more than the configured "
                "fraction of measured RMS solution error"
            ),
            "saved_solution": "interior discrete solution for each case",
        },
    )

    def checkpoint():
        write_csv(
            accuracy_path,
            sort_rows(rows, ("N", "method", "cfl")),
            ACCURACY_FIELDS,
        )

    for N in Ns:
        for cfl in cfls:
            for method in METHODS:
                key = (int(N), method, round(float(cfl), 12))
                saved_solution = solution_path(output_dir, N, method, cfl)
                if key in row_index and saved_solution.exists():
                    continue
                omega = selected_omegas[key]
                print(f"accuracy: N={N} method={method} CFL={cfl:g}")
                config = make_config(effective_settings, method, N, cfl, omega)
                _, U, histories = run_poisson_case(config)
                row = accuracy_row(config, U, histories)
                valid, validation_reason = validate_accuracy(
                    row,
                    settings["accuracy"],
                )
                row["accuracy_valid"] = valid
                row["validation_reason"] = validation_reason

                saved_solution.parent.mkdir(parents=True, exist_ok=True)
                U_interior = U[
                    config.ib:config.ie + 1,
                    config.jb:config.je + 1,
                ]
                np.savez_compressed(saved_solution, U_interior=U_interior)

                if key in row_index:
                    rows[row_index[key]] = row
                else:
                    row_index[key] = len(rows)
                    rows.append(row)
                checkpoint()

    checkpoint()
    order_rows = observed_orders(rows)
    comparison_rows = method_comparisons(rows, output_dir)
    write_csv(
        output_dir / "observed_orders.csv",
        sort_rows(order_rows, ("method", "cfl", "N_coarse")),
        ORDER_FIELDS,
    )
    write_csv(
        output_dir / "method_accuracy_comparison.csv",
        sort_rows(comparison_rows, ("N", "cfl")),
        COMPARISON_FIELDS,
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
