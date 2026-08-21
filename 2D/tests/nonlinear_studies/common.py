"""Shared configuration, execution, selection, and output helpers."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import boundary_conditions as bc
from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    NonlinearCoefficientConfig,
    ParabolicUnsplitConfig,
    ProblemConfig,
    RunConfig,
    StoppingConfig,
    compute_omega_opt,
    initialize_simulation,
)
from methods.nonlinear_diffusion import nl_helpers
from problems import nonlinear_diffusion
from problems.nonlinear import ellipticity_ratio
from solver_utils import normalized_grid_norm


RESULTS_ROOT = BASE_DIR / "results" / "nonlinear_studies"
TOLERANCE = 1e-5
RES_MAX = 1e8
EXPLORATORY_MAX_SWEEPS = 20_000
FINAL_MAX_SWEEPS = 60_000
CHUNK_SWEEPS = 500
STAGNATION_MIN_SWEEPS = 2_000
STAGNATION_WINDOW = 1_000
STAGNATION_RELATIVE_IMPROVEMENT = 1e-3


def make_config(
    method,
    N,
    omega,
    *,
    cfl=0.8,
    timestep_mode="fixed",
    epsilon=0.0,
    coefficient_variant="regularized",
    coefficient_mode="pointwise",
    initial_guess="zero",
):
    if coefficient_variant not in {"regularized", "constant"}:
        raise ValueError("coefficient_variant must be 'regularized' or 'constant'.")
    settings = Config(
        problem=ProblemConfig(
            kind="nonlinear_diffusion",
            case="unit_square_linear_x",
            coefficient=NonlinearCoefficientConfig(
                model=coefficient_variant,
                regularization_epsilon=epsilon,
                constant_value=0.5,
            ),
            initial_condition=initial_guess,
        ),
        grid=GridConfig(Nx=N, Ny=N),
        run=RunConfig(
            method=method,
            cfl=float(cfl),
            omega=float(omega),
            iter_max=CHUNK_SWEEPS,
            norm="fro",
            timestep_mode=timestep_mode,
            stopping=StoppingConfig(
                tol_update=TOLERANCE,
                tol_residual=TOLERANCE,
                res_max=RES_MAX,
            ),
        ),
        methods=MethodConfig(
            h_unsplit=HyperbolicUnsplitConfig(mode="default"),
            p_unsplit=ParabolicUnsplitConfig(
                gs_ordering="lexicographic",
                coefficient_mode=coefficient_mode,
            ),
        ),
    )
    return initialize_simulation(settings)


def formula_reference(method, N, cfl=0.8):
    """Use the optimal-omega formula already implemented on Config."""
    config = make_config(method, N, 1.0, cfl=cfl)
    return compute_omega_opt(config.settings, config.grid)


def omega_values(lower, upper, step, extras=()):
    values = list(np.arange(lower, upper + 0.5 * step, step))
    values.extend(extras)
    return sorted({
        round(float(value), 12)
        for value in values
        if lower <= value <= upper
    })


def is_stagnated(residuals):
    if len(residuals) < max(STAGNATION_MIN_SWEEPS, STAGNATION_WINDOW):
        return False
    window = np.asarray(residuals[-STAGNATION_WINDOW:], dtype=float)
    if not np.all(np.isfinite(window)):
        return False
    scale = max(abs(float(window[0])), 1e-14)
    relative_improvement = (float(window[0]) - float(np.min(window))) / scale
    return relative_improvement < STAGNATION_RELATIVE_IMPROVEMENT


def _face_stats(U, config):
    state = U.copy()
    bc.update_BCs(state, config)
    vx = nl_helpers.get_vx(state, config)
    vy = nl_helpers.get_vy(state, config)
    values = np.concatenate((vx.ravel(), vy.ravel()))
    return float(np.min(values)), float(np.max(values))


def run_case(
    config,
    metadata,
    max_sweeps=EXPLORATORY_MAX_SWEEPS,
    *,
    solve_callback=None,
):
    initial_face_min, initial_face_max = _face_stats(config.U0, config)
    U = config.U0.copy()
    updates = []
    residuals = []
    timesteps = []
    pseudo_time = 0.0
    converged = False
    stable = True
    reason = "max_sweeps"

    solve_callback = solve_callback or nonlinear_diffusion.solve_all
    start = time.perf_counter()
    while len(updates) < max_sweeps:
        config.U0 = U.copy()
        config.iter_max = min(CHUNK_SWEEPS, max_sweeps - len(updates))
        with contextlib.redirect_stdout(io.StringIO()):
            U, chunk_time, _, histories = solve_callback(config)
        updates.extend(histories["update"].tolist())
        residuals.extend(histories["residual"].tolist())
        timesteps.extend(histories["dt"].tolist())
        pseudo_time += float(chunk_time)

        if histories["converged"]:
            converged = True
            reason = histories["reason"]
            break
        if not histories["stable"]:
            stable = False
            reason = histories["reason"]
            break
        if is_stagnated(residuals):
            reason = "stagnated_residual"
            break
    elapsed = time.perf_counter() - start

    try:
        final_face_min, final_face_max = _face_stats(U, config)
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        # A divergent candidate may legitimately reach a state where the
        # study coefficient cannot be evaluated.  Preserve the solver's
        # stopping reason and record unavailable diagnostics instead of
        # aborting the remaining parameter sweep.
        final_face_min = math.nan
        final_face_max = math.nan
    dt = np.asarray(timesteps, dtype=float)
    row = {
        **metadata,
        "method": config.method,
        "N": config.Nx,
        "cfl": config.cfl,
        "omega": config.omega,
        "omega_cfl": config.omega * config.cfl,
        "timestep_mode": config.timestep_mode,
        "sweeps": len(updates),
        "elapsed_sec": elapsed,
        "seconds_per_sweep": elapsed / max(1, len(updates)),
        "pseudo_time": pseudo_time,
        "dt_initial": float(dt[0]) if dt.size else math.nan,
        "dt_min": float(np.min(dt)) if dt.size else math.nan,
        "dt_mean": float(np.mean(dt)) if dt.size else math.nan,
        "dt_final": float(dt[-1]) if dt.size else math.nan,
        "dt_max": float(np.max(dt)) if dt.size else math.nan,
        "converged": converged,
        "stable": stable,
        "reason": reason,
        "update_final": float(updates[-1]) if updates else math.nan,
        "residual_final": float(residuals[-1]) if residuals else math.nan,
        "residual_best": float(np.min(residuals)) if residuals else math.nan,
        "initial_face_v_min": initial_face_min,
        "initial_face_v_max": initial_face_max,
        "final_face_v_min": final_face_min,
        "final_face_v_max": final_face_max,
        "state_min": float(np.min(U)),
        "state_max": float(np.max(U)),
        "max_sweeps": max_sweeps,
    }
    return row, U


def run_omega_candidates(
    method,
    N,
    candidates,
    *,
    stage,
    cfl=0.8,
    timestep_mode="fixed",
    epsilon=0.0,
    coefficient_variant="regularized",
    coefficient_mode="pointwise",
    initial_guess="zero",
    max_sweeps=EXPLORATORY_MAX_SWEEPS,
    extra_metadata=None,
    solve_callback=None,
):
    reference = formula_reference(method, N, cfl)
    rows = []
    for omega in candidates:
        print(
            f"{stage}: N={N} method={method} CFL={cfl:g} "
            f"omega={omega:.6g}"
        )
        config = make_config(
            method,
            N,
            omega,
            cfl=cfl,
            timestep_mode=timestep_mode,
            epsilon=epsilon,
            coefficient_variant=coefficient_variant,
            coefficient_mode=coefficient_mode,
            initial_guess=initial_guess,
        )
        metadata = {
            "stage": stage,
            "formula_reference": reference,
            "omega_delta": float(omega) - reference,
            "epsilon": epsilon if coefficient_variant == "regularized" else None,
            "coefficient_variant": coefficient_variant,
            "ellipticity_ratio": (
                ellipticity_ratio(epsilon)
                if coefficient_variant == "regularized"
                else 1.0
            ),
            "initial_guess": initial_guess,
            **(extra_metadata or {}),
        }
        row, _ = run_case(
            config,
            metadata,
            max_sweeps=max_sweeps,
            solve_callback=solve_callback,
        )
        rows.append(row)
    return rows


def select_first_converged_branch(rows):
    branch = []
    saw_converged = False
    for row in sorted(rows, key=lambda item: float(item["omega"])):
        success = row["converged"] is True and row["stable"] is True
        if success:
            saw_converged = True
            branch.append(row)
        elif saw_converged:
            break
    if not branch:
        raise RuntimeError("No converged omega was found on the first stable branch.")
    return min(
        branch,
        key=lambda row: (
            int(row["sweeps"]),
            float(row["elapsed_sec"]),
            abs(float(row["omega_delta"])),
        ),
    )


def mark_search_summary(best, candidates):
    result = dict(best)
    result["search_lower"] = min(candidates)
    result["search_upper"] = max(candidates)
    result["boundary_hit"] = math.isclose(best["omega"], min(candidates)) or math.isclose(
        best["omega"], max(candidates)
    )
    result["candidates_tested"] = len(candidates)
    return result


def _fieldnames(rows):
    return list(dict.fromkeys(key for row in rows for key in row))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def save_study(study, rows, summary_rows, settings, output_dir=None, summary_name="best_by_case.csv"):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(output_dir) if output_dir else RESULTS_ROOT / study / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_runs.csv", rows)
    write_csv(output_dir / summary_name, summary_rows)
    payload = {
        **settings,
        "tol_update": TOLERANCE,
        "tol_residual": TOLERANCE,
        "res_max": RES_MAX,
        "exploratory_max_sweeps": EXPLORATORY_MAX_SWEEPS,
        "final_max_sweeps": FINAL_MAX_SWEEPS,
        "stagnation_min_sweeps": STAGNATION_MIN_SWEEPS,
        "stagnation_window": STAGNATION_WINDOW,
        "stagnation_relative_improvement": STAGNATION_RELATIVE_IMPROVEMENT,
    }
    with (output_dir / "settings.json").open("w") as stream:
        json.dump(payload, stream, indent=2, default=str)
    print(f"wrote {output_dir}")
    return output_dir


def latest_omega_results(path=None):
    if path:
        candidate = Path(path)
        return candidate / "best_by_case.csv" if candidate.is_dir() else candidate
    root = RESULTS_ROOT / "omega"
    matches = sorted(root.glob("*/best_by_case.csv"))
    if not matches:
        raise FileNotFoundError(
            "No nonlinear omega results found. Run the omega study first or "
            "provide --omega-results."
        )
    return matches[-1]


def load_baseline_omegas(path=None):
    result_path = latest_omega_results(path)
    with result_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (int(row["N"]), row["method"]): float(row["omega"])
        for row in rows
    }, result_path


def solution_difference(U_a, U_b, config):
    interior = (
        slice(config.ib, config.ie + 1),
        slice(config.jb, config.je + 1),
    )
    return normalized_grid_norm((U_a - U_b)[interior], "fro")
