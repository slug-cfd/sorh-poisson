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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))

from config1D import Config1D, Solver1DConfig
from problem_helpers import exact_diagnostic
from solver1D import METHOD_REGISTRY, RunRecord1D, RunResult1D, run_simulation
from sp_analysis_hyperbolic import spectral_radius as hyperbolic_spectral_radius
from sp_analysis_parabolic import spectral_radius as parabolic_spectral_radius


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "generated_report_data" / "comprehensive"

GRID_N_VALUES = (17, 33, 65, 129, 257)
METHOD_LABELS = {
    "jacobi": "Jacobi",
    "gs": "GS",
    "sorp": "SORp",
    "sorh": "SORh",
}
SOR_SCHEMES = ("sorp", "sorh")
FIGURE5_DEFAULT_SORP_OMEGAS = {
    17: 2.870,
    33: 2.851,
    65: 2.880,
    129: 2.797,
    257: 2.848,
}
FIGURE5_DEFAULT_SORH_OMEGAS = {
    17: 0.824,
    33: 0.899,
    65: 0.946,
    129: 0.965,
    257: 0.982,
}

STANDARD_FIELDNAMES = (
    "figure",
    "problem",
    "method_label",
    "method",
    "N",
    "interior_unknowns",
    "cfl",
    "omega",
    "omega_source",
    "tol_update",
    "iter_max",
    "converged",
    "stable",
    "reason",
    "iter_count",
    "elapsed_sec",
    "final_update",
    "final_exact",
    "best_exact",
    "best_iter",
    "pseudo_time",
)

OPTIMUM_FIELDNAMES = (
    "problem",
    "method_label",
    "method",
    "N",
    "interior_unknowns",
    "cfl",
    "omega_predicted",
    "omega_formula",
    "spectral_radius_predicted",
    "spectral_radius_formula",
    "spectral_sweep_step",
    "spectral_sweep_min",
    "spectral_sweep_max",
    "omega_experimental",
    "omega_experimental_3dp",
    "omega_offset",
    "iter_count",
    "elapsed_sec",
    "final_update",
    "final_exact",
    "converged",
    "stable",
    "reason",
    "search_stage",
)

SPECTRAL_FIELDNAMES = (
    "problem",
    "method_label",
    "method",
    "N",
    "interior_unknowns",
    "cfl",
    "omega",
    "spectral_radius",
    "is_spectral_minimum",
    "omega_formula",
    "spectral_radius_formula",
)

SENSITIVITY_FIELDNAMES = (
    "figure",
    "problem",
    "method_label",
    "method",
    "N",
    "interior_unknowns",
    "cfl",
    "omega",
    "omega_predicted",
    "omega_experimental",
    "is_predicted_omega",
    "is_experimental_omega",
    "is_instability_cutoff",
    "tol_update",
    "iter_max",
    "converged",
    "stable",
    "reason",
    "iter_count",
    "elapsed_sec",
    "final_update",
    "final_exact",
)

ERROR_FIELDNAMES = (
    "figure",
    "problem",
    "method_label",
    "method",
    "N",
    "interior_unknowns",
    "cfl",
    "omega",
    "omega_source",
    "tol_update",
    "iter_max",
    "converged",
    "stable",
    "reason",
    "iter_count",
    "elapsed_sec",
    "final_update",
    "final_exact",
    "plateau_error_mean",
    "plateau_error_min",
    "plateau_error_max",
    "plateau_error_std",
    "plateau_relative_range",
    "plateau_window",
)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def format_elapsed(seconds):
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


class Progress:
    def __init__(self, enabled=True, every=25):
        self.enabled = enabled
        self.every = max(1, every)
        self.start = time.perf_counter()

    def log(self, message):
        if self.enabled:
            elapsed = format_elapsed(time.perf_counter() - self.start)
            print(f"[{elapsed}] {message}", flush=True)

    def checkpoint(self, prefix, index, total, detail=""):
        if not self.enabled:
            return
        if index == 1 or index == total or index % self.every == 0:
            suffix = f" | {detail}" if detail else ""
            self.log(f"{prefix}: {index}/{total}{suffix}")


def method_name(problem, label):
    prefix = {
        "jacobi": "jacobi",
        "gs": "gs",
        "sorp": "sor",
        "sorh": "sor",
    }[label]
    scheme = "hyperbolic" if label == "sorh" else "parabolic"
    return f"{prefix}_{problem}_{scheme}"


def formula_predicted_omega(method_label, N, cfl):
    # Config1D uses N intervals and therefore M = N - 1 interior unknowns.
    # The analytical formulas require M + 1, which is exactly Config1D.N.
    config = Config1D(
        problem="linear",
        method=method_name("linear", method_label),
        N=N,
        cfl=cfl,
        omega=1.0,
        use_omega_opt=True,
        iter_max=1,
        norm=2,
        plot_enabled=False,
    )
    return config.omega


def spectral_radius_for_method(method_label, N, cfl, omega):
    if method_label == "sorp":
        rho, _ = parabolic_spectral_radius(N, cfl / 2.0, omega)
        return float(rho)
    if method_label == "sorh":
        rho, _ = hyperbolic_spectral_radius(N, cfl, omega)
        return float(rho)
    return math.nan


def spectral_prediction(problem, method_label, N, args):
    if problem != "linear":
        return None, None, []

    omega_formula = formula_predicted_omega(method_label, N, args.cfl)
    half_width = args.sorp_half_width if method_label == "sorp" else args.sorh_half_width
    omega_values = omega_candidates(
        omega_formula,
        half_width,
        args.spectral_omega_step,
    )
    rho_values = [
        spectral_radius_for_method(method_label, N, args.cfl, omega)
        for omega in omega_values
    ]
    minimum_index = int(np.argmin(rho_values))
    omega_predicted = omega_values[minimum_index]
    rho_formula = spectral_radius_for_method(
        method_label,
        N,
        args.cfl,
        omega_formula,
    )

    spectral_rows = []
    for index, (omega, rho) in enumerate(zip(omega_values, rho_values, strict=True)):
        spectral_rows.append(
            {
                "problem": problem,
                "method_label": METHOD_LABELS[method_label],
                "method": method_name(problem, method_label),
                "N": N,
                "interior_unknowns": N - 1,
                "cfl": args.cfl,
                "omega": omega,
                "spectral_radius": rho,
                "is_spectral_minimum": index == minimum_index,
                "omega_formula": omega_formula,
                "spectral_radius_formula": rho_formula,
            }
        )

    return omega_predicted, omega_formula, spectral_rows


def nonlinear_omega_range(method_label, args):
    if method_label == "sorh":
        return args.nonlinear_sorh_omega_min, args.nonlinear_sorh_omega_max
    return args.nonlinear_sorp_omega_min, args.nonlinear_sorp_omega_max


def make_config(problem, method_label, N, omega, args, tol_update=None, iter_max=None):
    tol = args.tol_update if tol_update is None else tol_update
    max_iter = args.iter_max if iter_max is None else iter_max
    return Config1D(
        problem=problem,
        method=method_name(problem, method_label),
        N=N,
        cfl=args.cfl,
        omega=float(omega),
        use_omega_opt=False,
        iter_max=max_iter,
        norm=2,
        solver_config=Solver1DConfig(
            tol_update=tol,
            res_max=args.res_max,
        ),
        linear_v=args.v,
        nonlinear_vmax=args.v,
        hyperbolic_linear_v=args.v,
        plot_enabled=False,
    )


def run_case(problem, method_label, N, omega, args, tol_update=None, iter_max=None):
    config = make_config(
        problem=problem,
        method_label=method_label,
        N=N,
        omega=omega,
        args=args,
        tol_update=tol_update,
        iter_max=iter_max,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, _, record = run_simulation(config)
    return record


def run_exploratory_case(problem, method_label, N, omega, args, tol_update=None, iter_max=None):
    config = make_config(
        problem=problem,
        method_label=method_label,
        N=N,
        omega=omega,
        args=args,
        tol_update=tol_update,
        iter_max=min(iter_max or args.iter_max, args.exploratory_iter_max),
    )
    if not args.exploratory_failsafe:
        return run_case(
            problem=problem,
            method_label=method_label,
            N=N,
            omega=omega,
            args=args,
            tol_update=tol_update,
            iter_max=config.iter_max,
        )

    method_module = METHOD_REGISTRY[config.method]
    params = method_module.get_params(config)
    U = config.U0.copy()
    config.apply_bcs(U)

    update_hist = []
    exact_hist = []
    dt_hist = []
    best_exact = math.inf
    best_iter = 0
    t = 0.0
    reason = "iter_max"
    stable = True
    converged = False
    start = time.perf_counter()

    for iter_count in range(1, config.iter_max + 1):
        U_old = U.copy()
        U = method_module.full_sweep(U, params, config)
        config.apply_bcs(U)

        update_res = norm(interior(U - U_old, config), config.norm, config.interior_size)
        exact_res = norm(
            interior(exact_diagnostic(U, config), config),
            config.norm,
            config.interior_size,
        )
        dt = params.get("dt", 0.0)
        t += dt

        update_hist.append(update_res)
        exact_hist.append(exact_res)
        dt_hist.append(dt)

        if exact_res < best_exact:
            best_exact = exact_res
            best_iter = iter_count

        if not np.isfinite(update_res) or update_res > config.solver_config.res_max:
            stable = False
            reason = "unstable"
            break

        if update_res < config.solver_config.tol_update:
            converged = True
            reason = "tol_update"
            break

        if _stagnated(update_hist, config.solver_config.tol_update, args):
            stable = False
            reason = "stagnated_update"
            break
    else:
        iter_count = config.iter_max

    elapsed = time.perf_counter() - start
    residuals = np.array(update_hist, dtype=float)
    exact_values = np.array(exact_hist, dtype=float)
    if residuals.size == 0:
        residual_final = math.inf
        residual_min = math.inf
        residual_max = math.inf
    else:
        residual_final = float(residuals[-1])
        residual_min = float(np.min(residuals))
        residual_max = float(np.max(residuals))

    return RunRecord1D(
        config=config,
        results=RunResult1D(
            converged=converged,
            stable=stable,
            iter_count=iter_count,
            residual_final=residual_final,
            residual_min=residual_min,
            residual_max=residual_max,
            elapsed_sec=elapsed,
            reason=reason,
        ),
        meta={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "t": t,
            "exact_final": float(exact_values[-1]) if exact_values.size else math.inf,
            "best_exact": float(best_exact),
            "best_iter": best_iter,
            "failsafe": {
                "enabled": True,
                "window": args.failsafe_window,
                "min_iter": args.failsafe_min_iter,
                "min_relative_improvement": args.failsafe_min_relative_improvement,
                "reason": reason if reason == "stagnated_update" else None,
            },
        },
    )


def _stagnated(update_hist, tol_update, args):
    window = args.failsafe_window
    iter_count = len(update_hist)
    if iter_count < max(args.failsafe_min_iter, 2 * window):
        return False
    if iter_count % args.failsafe_check_every:
        return False

    previous = min(update_hist[-2 * window:-window])
    recent = min(update_hist[-window:])
    if recent <= tol_update:
        return False

    scale = max(abs(previous), args.failsafe_residual_floor)
    relative_improvement = (previous - recent) / scale
    return relative_improvement < args.failsafe_min_relative_improvement


def run_performance_case(problem, method_label, N, omega, args):
    iter_max = args.iter_max
    for attempt in range(args.performance_max_retries + 1):
        record = run_case(
            problem=problem,
            method_label=method_label,
            N=N,
            omega=omega,
            args=args,
            tol_update=args.tol_update,
            iter_max=iter_max,
        )
        if record.results.converged and record.results.reason == "tol_update":
            return record

        if attempt == args.performance_max_retries:
            break

        next_iter_max = min(iter_max * args.performance_iter_growth, args.performance_iter_max_limit)
        if next_iter_max <= iter_max:
            break

        args.progress.log(
            "performance retry: "
            f"problem={problem} method={METHOD_LABELS[method_label]} N={N} "
            f"omega={omega:.3f} previous_iter_max={iter_max} "
            f"reason={record.results.reason} final_update={record.results.residual_final:.3e} "
            f"next_iter_max={next_iter_max}"
        )
        iter_max = next_iter_max

    raise RuntimeError(
        "Performance run did not reach tol_update; refusing to write it as valid data. "
        f"problem={problem} method={METHOD_LABELS[method_label]} N={N} omega={omega:.6g} "
        f"tol_update={args.tol_update:g} final_iter_max={iter_max}"
    )


def standard_row(record, figure, method_label, omega_source):
    config = record.config
    return {
        "figure": figure,
        "problem": config.problem,
        "method_label": METHOD_LABELS[method_label],
        "method": config.method,
        "N": config.N,
        "interior_unknowns": config.interior_size,
        "cfl": config.cfl,
        "omega": config.omega,
        "omega_source": omega_source,
        "tol_update": config.solver_config.tol_update,
        "iter_max": config.iter_max,
        "converged": record.results.converged,
        "stable": record.results.stable,
        "reason": record.results.reason,
        "iter_count": record.results.iter_count,
        "elapsed_sec": record.results.elapsed_sec,
        "final_update": record.results.residual_final,
        "final_exact": record.meta["exact_final"],
        "best_exact": record.meta["best_exact"],
        "best_iter": record.meta["best_iter"],
        "pseudo_time": record.meta["t"],
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unique_run_dir(parent, label):
    candidate = parent / label
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        suffixed = parent / f"{label}_{suffix}"
        if not suffixed.exists():
            return suffixed
        suffix += 1


def resolve_output_dir(args):
    if not args.separate_run_dir:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output:
            raise RuntimeError(
                f"Output directory {args.output_dir} already contains files. "
                "Use --separate-run-dir, choose a new --output-dir, or pass --overwrite-output."
            )
        return args.output_dir

    label = args.run_label
    if label is None:
        figure_text = "-".join(sorted(args.figures))
        label = f"{time.strftime('%Y%m%d_%H%M%S')}_{figure_text}"
    return unique_run_dir(args.output_dir, label)


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, default=json_default) + "\n")


def rounded_unique(values):
    result = []
    seen = set()
    for value in values:
        key = round(float(value), 12)
        if key not in seen:
            seen.add(key)
            result.append(float(value))
    return sorted(result)


def omega_candidates(center, half_width, step):
    lo = max(1.0e-8, center - half_width)
    hi = center + half_width
    values = np.arange(lo, hi + 0.5 * step, step)
    return rounded_unique([*values, center])


def format_omega_map(values):
    return ",".join(f"{N}:{omega:g}" for N, omega in sorted(values.items()))


def parse_omega_map(value, label):
    if isinstance(value, dict):
        return {int(N): float(omega) for N, omega in value.items()}
    result = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"{label} entries must be formatted as N:omega, got '{part}'."
            )
        raw_N, raw_omega = part.split(":", 1)
        try:
            N = int(raw_N)
            omega = float(raw_omega)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{label} entries must be numeric N:omega pairs, got '{part}'."
            ) from exc
        if omega <= 0:
            raise argparse.ArgumentTypeError(f"{label} omega values must be positive.")
        result[N] = omega
    if not result:
        raise argparse.ArgumentTypeError(f"{label} must contain at least one N:omega pair.")
    return result


def best_converged(rows, stop_at_first_failure=False):
    if stop_at_first_failure:
        branch_rows = []
        saw_converged = False
        for row in sorted(rows, key=lambda item: float(item["omega_experimental"])):
            success = row["converged"] is True and row["stable"] is True
            if success:
                saw_converged = True
                branch_rows.append(row)
            elif saw_converged:
                break
        rows = branch_rows

    candidates = [
        row for row in rows
        if row["converged"] is True and row["stable"] is True
    ]
    if not candidates:
        return None

    def omega_prediction_distance(row):
        omega_pred = row["omega_predicted"]
        if omega_pred in (None, ""):
            return 0.0
        return abs(float(row["omega_experimental"]) - float(omega_pred))

    return min(
        candidates,
        key=lambda row: (
            int(row["iter_count"]),
            float(row["elapsed_sec"]),
            omega_prediction_distance(row),
        ),
    )


def refine_omega(problem, method_label, N, args, raw_records_path):
    if problem == "linear":
        omega_pred, omega_formula, spectral_rows = spectral_prediction(
            problem, method_label, N, args
        )
        half_width = args.sorp_half_width if method_label == "sorp" else args.sorh_half_width
        coarse_candidates = omega_candidates(omega_pred, half_width, args.omega_coarse_step)
        coarse_stage = "spectral_centered_coarse"
        rho_pred = spectral_radius_for_method(method_label, N, args.cfl, omega_pred)
        rho_formula = spectral_radius_for_method(method_label, N, args.cfl, omega_formula)
    else:
        omega_pred = None
        omega_formula = None
        spectral_rows = []
        rho_pred = None
        rho_formula = None
        omega_min, omega_max = nonlinear_omega_range(method_label, args)
        coarse_candidates = rounded_unique(
            np.arange(
                omega_min,
                omega_max + 0.5 * args.nonlinear_omega_coarse_step,
                args.nonlinear_omega_coarse_step,
            )
        )
        coarse_stage = "experimental_broad_coarse"

    all_rows = []
    coarse_rows = []
    total_coarse = len(coarse_candidates)
    args.progress.log(
        f"omega search start: problem={problem} method={METHOD_LABELS[method_label]} "
        f"N={N} coarse_runs={total_coarse}"
    )
    for index, omega in enumerate(coarse_candidates, start=1):
        record = run_exploratory_case(problem, method_label, N, omega, args)
        append_jsonl(
            raw_records_path,
            {
                "stage": coarse_stage,
                "record": record.to_dict(),
            },
        )
        coarse_rows.append(
            omega_optimum_row(
                record,
                method_label,
                omega_pred,
                omega_formula,
                rho_pred,
                rho_formula,
                coarse_stage,
                args,
            )
        )
        args.progress.checkpoint(
            "omega coarse",
            index,
            total_coarse,
            f"{problem} {METHOD_LABELS[method_label]} N={N} omega={omega:.3f} "
            f"reason={record.results.reason} iters={record.results.iter_count}",
        )
    all_rows.extend(coarse_rows)

    stop_at_first_failure = problem == "nonlinear"
    coarse_best = best_converged(coarse_rows, stop_at_first_failure=stop_at_first_failure)
    if coarse_best is None:
        raise RuntimeError(
            f"No stable converged omega found for {problem} {method_label} N={N} "
            f"in the configured omega search range."
        )

    fine_center = float(coarse_best["omega_experimental"])
    fine_half_width = (
        max(args.omega_coarse_step, 2.0 * args.omega_fine_step)
        if problem == "linear"
        else max(args.nonlinear_omega_coarse_step, 2.0 * args.omega_fine_step)
    )
    fine_rows = []
    existing = {round(float(row["omega_experimental"]), 12) for row in all_rows}
    fine_candidates = [
        omega for omega in omega_candidates(fine_center, fine_half_width, args.omega_fine_step)
        if round(float(omega), 12) not in existing
    ]
    total_fine = len(fine_candidates)
    args.progress.log(
        f"omega fine start: problem={problem} method={METHOD_LABELS[method_label]} "
        f"N={N} center={fine_center:.3f} fine_runs={total_fine}"
    )
    for index, omega in enumerate(fine_candidates, start=1):
        if round(float(omega), 12) in existing:
            continue
        record = run_exploratory_case(problem, method_label, N, omega, args)
        append_jsonl(
            raw_records_path,
            {
                "stage": "omega_fine",
                "record": record.to_dict(),
            },
        )
        fine_rows.append(
            omega_optimum_row(
                record,
                method_label,
                omega_pred,
                omega_formula,
                rho_pred,
                rho_formula,
                "fine",
                args,
            )
        )
        args.progress.checkpoint(
            "omega fine",
            index,
            total_fine,
            f"{problem} {METHOD_LABELS[method_label]} N={N} omega={omega:.3f} "
            f"reason={record.results.reason} iters={record.results.iter_count}",
        )
    all_rows.extend(fine_rows)

    final_best = best_converged(all_rows, stop_at_first_failure=stop_at_first_failure)
    if final_best is None:
        raise RuntimeError(
            f"Fine search lost stable converged omega for {problem} {method_label} N={N}."
        )
    args.progress.log(
        f"omega search done: problem={problem} method={METHOD_LABELS[method_label]} "
        f"N={N} omega*={float(final_best['omega_experimental']):.3f} "
        f"iters={final_best['iter_count']}"
    )
    return final_best, all_rows, spectral_rows


def omega_optimum_row(
    record,
    method_label,
    omega_pred,
    omega_formula,
    rho_pred,
    rho_formula,
    search_stage,
    args,
):
    config = record.config
    return {
        "problem": config.problem,
        "method_label": METHOD_LABELS[method_label],
        "method": config.method,
        "N": config.N,
        "interior_unknowns": config.interior_size,
        "cfl": config.cfl,
        "omega_predicted": omega_pred,
        "omega_formula": omega_formula,
        "spectral_radius_predicted": rho_pred,
        "spectral_radius_formula": rho_formula,
        "spectral_sweep_step": args.spectral_omega_step if config.problem == "linear" else None,
        "spectral_sweep_min": (
            omega_formula - (args.sorp_half_width if method_label == "sorp" else args.sorh_half_width)
            if config.problem == "linear" and omega_formula is not None
            else None
        ),
        "spectral_sweep_max": (
            omega_formula + (args.sorp_half_width if method_label == "sorp" else args.sorh_half_width)
            if config.problem == "linear" and omega_formula is not None
            else None
        ),
        "omega_experimental": config.omega,
        "omega_experimental_3dp": round(config.omega, 3),
        "omega_offset": None if omega_pred is None else config.omega - omega_pred,
        "iter_count": record.results.iter_count,
        "elapsed_sec": record.results.elapsed_sec,
        "final_update": record.results.residual_final,
        "final_exact": record.meta["exact_final"],
        "converged": record.results.converged,
        "stable": record.results.stable,
        "reason": record.results.reason,
        "search_stage": search_stage,
    }


def required_optimum_problems(figures):
    problems = set()
    if {"figure1", "figure3"} & figures:
        problems.add("linear")
    if {"figure2", "figure4"} & figures:
        problems.add("nonlinear")
    return problems


def collect_optima(args, output_dir, raw_records_path):
    all_rows = []
    spectral_rows = []
    best_by_key = {}
    problems = sorted(required_optimum_problems(args.figures))
    if not problems:
        args.progress.log("omega optimization skipped: requested figures do not need omega search")
        return best_by_key

    for problem in problems:
        for method_label in SOR_SCHEMES:
            for N in args.Ns:
                best, rows, method_spectral_rows = refine_omega(
                    problem, method_label, N, args, raw_records_path
                )
                all_rows.extend(rows)
                spectral_rows.extend(method_spectral_rows)
                best_by_key[(problem, method_label, N)] = best

    write_csv(output_dir / "omega_optima_all_runs.csv", OPTIMUM_FIELDNAMES, all_rows)
    best_rows = list(best_by_key.values())
    write_csv(output_dir / "omega_optima_selected.csv", OPTIMUM_FIELDNAMES, best_rows)
    if spectral_rows:
        write_csv(output_dir / "figure3_spectral_radius_sweep.csv", SPECTRAL_FIELDNAMES, spectral_rows)
    return best_by_key


def omega_for_performance(problem, method_label, N, best_by_key):
    if method_label not in SOR_SCHEMES:
        return 1.0, "fixed_1"
    row = best_by_key[(problem, method_label, N)]
    return float(row["omega_experimental_3dp"]), "experimental_opt_3dp"


def collect_performance(problem, figure, args, best_by_key, output_dir, raw_records_path):
    rows = []
    total = len(args.Ns) * 4
    index = 0
    args.progress.log(f"{figure} performance start: problem={problem} runs={total}")
    for N in args.Ns:
        for method_label in ("jacobi", "gs", "sorp", "sorh"):
            index += 1
            omega, omega_source = omega_for_performance(problem, method_label, N, best_by_key)
            record = run_performance_case(problem, method_label, N, omega, args)
            append_jsonl(
                raw_records_path,
                {
                    "stage": figure,
                    "record": record.to_dict(),
                },
            )
            rows.append(standard_row(record, figure, method_label, omega_source))
            args.progress.checkpoint(
                f"{figure} performance",
                index,
                total,
                f"{METHOD_LABELS[method_label]} N={N} omega={omega:.3f} "
                f"reason={record.results.reason} iters={record.results.iter_count}",
            )

    filename = f"{figure}_{problem}_performance.csv"
    write_csv(output_dir / filename, STANDARD_FIELDNAMES, rows)
    args.progress.log(f"{figure} performance done: wrote {filename}")
    return rows


def collect_figure3(args, best_by_key, output_dir):
    rows = []
    problem = args.figure3_problem
    for N in args.Ns:
        for method_label in SOR_SCHEMES:
            selected = best_by_key[(problem, method_label, N)].copy()
            selected["search_stage"] = "selected"
            rows.append(selected)
    write_csv(output_dir / "figure3_omega_optima.csv", OPTIMUM_FIELDNAMES, rows)
    return rows


def sensitivity_values(method_label, omega_pred, omega_exp, args):
    if method_label == "sorh":
        lo = args.sorh_sensitivity_min
        hi = args.sorh_sensitivity_max
    else:
        lo = args.sorp_sensitivity_min
        hi = args.sorp_sensitivity_max

    anchors = [lo, hi, omega_exp]
    if omega_pred is not None:
        anchors.append(omega_pred)
    fine_windows = []
    for center in anchors:
        fine_windows.append((center - args.sensitivity_fine_width, center + args.sensitivity_fine_width))

    values = []
    omega = lo
    while omega <= hi + 0.5 * args.sensitivity_coarse_step:
        in_fine_window = any(left <= omega <= right for left, right in fine_windows)
        step = args.sensitivity_fine_step if in_fine_window else args.sensitivity_coarse_step
        values.append(omega)
        omega += step

    values.extend(anchors)
    return [
        value for value in rounded_unique(round(value, 3) for value in values)
        if lo - 1.0e-12 <= value <= hi + 1.0e-12
    ]


def collect_figure4(args, best_by_key, output_dir, raw_records_path):
    all_rows = []
    converged_rows = []
    summaries = []
    problem = args.figure4_problem
    N = args.figure4_N

    for method_label in SOR_SCHEMES:
        selected = best_by_key[(problem, method_label, N)]
        omega_pred = (
            None
            if selected["omega_predicted"] is None
            else float(selected["omega_predicted"])
        )
        omega_exp = float(selected["omega_experimental_3dp"])
        method_rows = []
        omega_values = sensitivity_values(method_label, omega_pred, omega_exp, args)
        total = len(omega_values)
        args.progress.log(
            f"figure4 sensitivity start: problem={problem} method={METHOD_LABELS[method_label]} "
            f"N={N} runs={total}"
        )

        for index, omega in enumerate(omega_values, start=1):
            record = run_exploratory_case(
                problem,
                method_label,
                N,
                omega,
                args,
                tol_update=args.sensitivity_tol_update,
                iter_max=args.sensitivity_iter_max,
            )
            append_jsonl(
                raw_records_path,
                {
                    "stage": "figure4_sensitivity",
                    "record": record.to_dict(),
                },
            )
            row = sensitivity_row(record, method_label, omega_pred, omega_exp)
            method_rows.append(row)
            args.progress.checkpoint(
                "figure4 sensitivity",
                index,
                total,
                f"{METHOD_LABELS[method_label]} omega={omega:.3f} "
                f"reason={record.results.reason} iters={record.results.iter_count}",
            )

        stable_converged = [
            row for row in method_rows
            if row["converged"] is True and row["stable"] is True
        ]
        cutoff = max((row["omega"] for row in stable_converged), default=None)
        if cutoff is not None:
            for row in method_rows:
                row["is_instability_cutoff"] = math.isclose(
                    float(row["omega"]),
                    float(cutoff),
                    rel_tol=0.0,
                    abs_tol=5.0e-4,
                )

        all_rows.extend(method_rows)
        converged_rows.extend(stable_converged)
        summaries.append(
            {
                "problem": problem,
                "method_label": METHOD_LABELS[method_label],
                "method": method_name(problem, method_label),
                "N": N,
                "omega_predicted": omega_pred,
                "omega_experimental_3dp": omega_exp,
                "lowest_converged_omega": min((row["omega"] for row in stable_converged), default=None),
                "highest_converged_omega": cutoff,
                "n_runs": len(method_rows),
                "n_converged": len(stable_converged),
            }
        )
        args.progress.log(
            f"figure4 sensitivity done: method={METHOD_LABELS[method_label]} "
            f"converged={len(stable_converged)}/{len(method_rows)} cutoff={cutoff}"
        )

    write_csv(output_dir / "figure4_sensitivity_all_runs.csv", SENSITIVITY_FIELDNAMES, all_rows)
    write_csv(output_dir / "figure4_sensitivity_converged.csv", SENSITIVITY_FIELDNAMES, converged_rows)
    (output_dir / "figure4_sensitivity_summary.json").write_text(
        json.dumps(summaries, indent=2, default=json_default) + "\n"
    )
    return converged_rows


def sensitivity_row(record, method_label, omega_pred, omega_exp):
    config = record.config
    omega = config.omega
    return {
        "figure": "figure4",
        "problem": config.problem,
        "method_label": METHOD_LABELS[method_label],
        "method": config.method,
        "N": config.N,
        "interior_unknowns": config.interior_size,
        "cfl": config.cfl,
        "omega": omega,
        "omega_predicted": omega_pred,
        "omega_experimental": omega_exp,
        "is_predicted_omega": (
            False
            if omega_pred is None
            else math.isclose(omega, omega_pred, rel_tol=0.0, abs_tol=5.0e-4)
        ),
        "is_experimental_omega": math.isclose(omega, omega_exp, rel_tol=0.0, abs_tol=5.0e-4),
        "is_instability_cutoff": False,
        "tol_update": config.solver_config.tol_update,
        "iter_max": config.iter_max,
        "converged": record.results.converged,
        "stable": record.results.stable,
        "reason": record.results.reason,
        "iter_count": record.results.iter_count,
        "elapsed_sec": record.results.elapsed_sec,
        "final_update": record.results.residual_final,
        "final_exact": record.meta["exact_final"],
    }


def norm(values, ord_value, normalize_count):
    result = float(np.linalg.norm(values, ord=ord_value))
    if ord_value in (1, 2):
        result /= np.sqrt(normalize_count)
    return result


def interior(values, config):
    return values[config.ib:config.ie + 1]


def plateau_stats(exact_hist, window, atol):
    if len(exact_hist) < window:
        return None
    values = np.asarray(exact_hist[-window:], dtype=float)
    mean = float(np.mean(values))
    spread = float(np.max(values) - np.min(values))
    relative_range = spread / max(abs(mean), atol)
    return {
        "mean": mean,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "relative_range": relative_range,
    }


def run_until_error_plateau(problem, method_label, N, omega, args):
    config = make_config(
        problem,
        method_label,
        N,
        omega,
        args,
        tol_update=args.error_tol_update,
        iter_max=args.error_iter_max,
    )
    method_module = METHOD_REGISTRY[config.method]
    params = method_module.get_params(config)
    U = config.U0.copy()
    config.apply_bcs(U)

    update_hist = []
    exact_hist = []
    t = 0.0
    stable = True
    converged = False
    reason = "iter_max"
    start = time.perf_counter()

    for iter_count in range(1, config.iter_max + 1):
        U_old = U.copy()
        U = method_module.full_sweep(U, params, config)
        config.apply_bcs(U)

        update_res = norm(interior(U - U_old, config), config.norm, config.interior_size)
        exact_res = norm(
            interior(exact_diagnostic(U, config), config),
            config.norm,
            config.interior_size,
        )
        update_hist.append(update_res)
        exact_hist.append(exact_res)
        t += params.get("dt", 0.0)

        if not np.isfinite(update_res) or update_res > config.solver_config.res_max:
            stable = False
            reason = "unstable"
            break

        stats = plateau_stats(exact_hist, args.error_plateau_window, args.error_plateau_atol)
        plateau_reached = (
            stats is not None
            and stats["relative_range"] <= args.error_plateau_rtol
        )
        if update_res < config.solver_config.tol_update and plateau_reached:
            converged = True
            reason = "tol_update_and_error_plateau"
            break
    else:
        iter_count = config.iter_max

    elapsed_sec = time.perf_counter() - start
    stats = plateau_stats(exact_hist, min(args.error_plateau_window, len(exact_hist)), args.error_plateau_atol)
    if stats is None:
        stats = {
            "mean": math.inf,
            "min": math.inf,
            "max": math.inf,
            "std": math.inf,
            "relative_range": math.inf,
        }

    row = {
        "figure": "figure5",
        "problem": problem,
        "method_label": METHOD_LABELS[method_label],
        "method": config.method,
        "N": config.N,
        "interior_unknowns": config.interior_size,
        "cfl": config.cfl,
        "omega": config.omega,
        "omega_source": "experimental_opt_3dp" if method_label in SOR_SCHEMES else "fixed_1",
        "tol_update": config.solver_config.tol_update,
        "iter_max": config.iter_max,
        "converged": converged,
        "stable": stable,
        "reason": reason,
        "iter_count": iter_count,
        "elapsed_sec": elapsed_sec,
        "final_update": float(update_hist[-1]) if update_hist else math.inf,
        "final_exact": float(exact_hist[-1]) if exact_hist else math.inf,
        "plateau_error_mean": stats["mean"],
        "plateau_error_min": stats["min"],
        "plateau_error_max": stats["max"],
        "plateau_error_std": stats["std"],
        "plateau_relative_range": stats["relative_range"],
        "plateau_window": min(args.error_plateau_window, len(exact_hist)),
    }
    raw_payload = {
        "stage": "figure5_error_plateau",
        "config": config.to_dict(),
        "results": {
            "converged": converged,
            "stable": stable,
            "iter_count": iter_count,
            "elapsed_sec": elapsed_sec,
            "reason": reason,
            "final_update": row["final_update"],
            "final_exact": row["final_exact"],
            "plateau_error_mean": row["plateau_error_mean"],
            "plateau_relative_range": row["plateau_relative_range"],
        },
    }
    return row, raw_payload


def figure5_omega(method_label, N, args):
    if method_label == "sorp":
        return args.figure5_sorp_omegas[N], "fixed_figure5_sorp_by_N"
    if method_label == "sorh":
        return args.figure5_sorh_omegas[N], "fixed_figure5_sorh_by_N"
    return 1.0, "fixed_1"


def collect_figure5(args, output_dir, raw_records_path):
    rows = []
    total = len(args.Ns) * 4
    index = 0
    args.progress.log(f"figure5 error plateau start: runs={total}")
    for N in args.Ns:
        for method_label in ("jacobi", "gs", "sorp", "sorh"):
            index += 1
            omega, omega_source = figure5_omega(method_label, N, args)
            row, raw_payload = run_until_error_plateau("nonlinear", method_label, N, omega, args)
            row["omega_source"] = omega_source
            append_jsonl(raw_records_path, raw_payload)
            rows.append(row)
            args.progress.checkpoint(
                "figure5 error plateau",
                index,
                total,
                f"{METHOD_LABELS[method_label]} N={N} omega={omega:.3f} "
                f"reason={row['reason']} iters={row['iter_count']}",
            )
    write_csv(output_dir / "figure5_nonlinear_final_error.csv", ERROR_FIELDNAMES, rows)
    args.progress.log("figure5 error plateau done: wrote figure5_nonlinear_final_error.csv")
    return rows


def parse_figures(values):
    requested = set()
    for value in values:
        for part in value.split(","):
            part = part.strip().lower()
            if part:
                requested.add(part)
    aliases = {
        "1": "figure1",
        "2": "figure2",
        "3": "figure3",
        "4": "figure4",
        "5": "figure5",
        "all": "all",
    }
    result = {aliases.get(item, item) for item in requested}
    valid = {"all", "figure1", "figure2", "figure3", "figure4", "figure5"}
    invalid = result - valid
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown figure selector(s): {sorted(invalid)}")
    if "all" in result:
        return {"figure1", "figure2", "figure3", "figure4", "figure5"}
    return result


def write_metadata(args, output_dir):
    metadata = {
        "description": (
            "New data collected in the completed thesis figure format. "
            "The frozen thesis dataset is not modified."
        ),
        "created_by": Path(__file__).name,
        "N_values": list(args.Ns),
        "N_note": (
            "These values are passed directly to Config1D.N. In the current code this gives "
            "interior_size = N - 1, matching the requested 16, 32, 64, 128, 256 interior unknowns."
        ),
        "methods": METHOD_LABELS,
        "cfl": args.cfl,
        "v": args.v,
        "tol_update": args.tol_update,
        "iter_max": args.iter_max,
        "performance_convergence_policy": {
            "required": "Figure 1 and Figure 2 performance rows must converge by tol_update.",
            "initial_iter_max": args.iter_max,
            "iter_growth": args.performance_iter_growth,
            "max_retries": args.performance_max_retries,
            "iter_max_limit": args.performance_iter_max_limit,
            "failure_behavior": "raise RuntimeError before writing an invalid performance row",
        },
        "figure3_problem": "linear",
        "figure4_problem": "nonlinear",
        "figure4_N": args.figure4_N,
        "figure5_problem": "nonlinear",
        "output": {
            "output_dir": str(output_dir),
            "separate_run_dir": args.separate_run_dir,
            "run_label": args.run_label,
        },
        "omega_refinement": {
            "linear_note": (
                "Linear SOR searches and spectral-radius sweeps use Config1D.N directly. "
                "Because M = N - 1 is the number of interior unknowns, this evaluates "
                "the analytical formula at M + 1 as required."
            ),
            "linear_formula_note": (
                "omega_formula is the analytical prediction; omega_predicted is the "
                "minimum of a spectral-radius sweep centered on that formula, using the "
                "same Config1D.N value."
            ),
            "linear_sorp_half_width": args.sorp_half_width,
            "linear_sorh_half_width": args.sorh_half_width,
            "linear_coarse_step": args.omega_coarse_step,
            "spectral_omega_step": args.spectral_omega_step,
            "nonlinear_note": "Nonlinear SOR searches are broad experimental sweeps only; no spectral prediction is used.",
            "nonlinear_selection_policy": (
                "Select the best converged omega on the first contiguous converged branch; "
                "ignore later isolated converged pockets after the first failed candidate."
            ),
            "nonlinear_sorp_range": [args.nonlinear_sorp_omega_min, args.nonlinear_sorp_omega_max],
            "nonlinear_sorh_range": [args.nonlinear_sorh_omega_min, args.nonlinear_sorh_omega_max],
            "nonlinear_coarse_step": args.nonlinear_omega_coarse_step,
            "fine_step": args.omega_fine_step,
            "selected_value": "experimental optimum rounded to 3 decimals",
        },
        "exploratory_failsafe": {
            "enabled": args.exploratory_failsafe,
            "applies_to": "Omega optimization and Figure 4 sensitivity sweeps only.",
            "iter_max": args.exploratory_iter_max,
            "window": args.failsafe_window,
            "min_iter": args.failsafe_min_iter,
            "check_every": args.failsafe_check_every,
            "min_relative_improvement": args.failsafe_min_relative_improvement,
            "residual_floor": args.failsafe_residual_floor,
            "reason": "stagnated_update",
        },
        "figure5_plateau": {
            "tol_update": args.error_tol_update,
            "iter_max": args.error_iter_max,
            "window": args.error_plateau_window,
            "relative_range_tolerance": args.error_plateau_rtol,
            "absolute_scale": args.error_plateau_atol,
            "reported_error": "mean numerical error over the final plateau window",
            "omega_policy": (
                "Use fixed method- and grid-specific omega values; "
                "do not run nonlinear omega search for Figure 5."
            ),
            "sorp_omegas": args.figure5_sorp_omegas,
            "sorh_omegas": args.figure5_sorh_omegas,
        },
        "discrepancies_resolved": [
            "The instructions call N a total grid-point count, but Config1D.N produces N-1 interior unknowns and an x array with N+1 entries. The script uses the requested N values as Config1D.N because that matches the requested interior-unknown counts.",
            "Spectral-analysis predictions are used only for the linear problem. Nonlinear SOR optima are selected by experimental broad sweep plus local refinement.",
            "Figure 3 does not name a problem. The script fixes it to the linear problem because the requested predicted-vs-experimental comparison depends on spectral analysis.",
            "Figure 4 is fixed to the nonlinear problem and does not use spectral predictions. Its omega_predicted values are null and is_predicted_omega is false.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect new data in the completed thesis figure format without "
            "modifying the frozen thesis dataset."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--separate-run-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--figures", nargs="+", default=["all"], help="all, 1..5, or figure1..figure5")
    parser.add_argument("--Ns", type=int, nargs="+", default=list(GRID_N_VALUES))
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--tol-update", type=float, default=1e-7)
    parser.add_argument("--iter-max", type=int, default=200000)
    parser.add_argument("--res-max", type=float, default=1e8)
    parser.add_argument("--performance-max-retries", type=int, default=3)
    parser.add_argument("--performance-iter-growth", type=int, default=2)
    parser.add_argument("--performance-iter-max-limit", type=int, default=1600000)

    parser.add_argument("--sorp-half-width", type=float, default=0.30)
    parser.add_argument("--sorh-half-width", type=float, default=0.12)
    parser.add_argument(
        "--spectral-omega-step",
        type=float,
        default=0.001,
        help="Retained for compatibility; Figure 3 now uses the analytical spectral formula.",
    )
    parser.add_argument("--omega-coarse-step", type=float, default=0.01)
    parser.add_argument("--omega-fine-step", type=float, default=0.001)
    parser.add_argument("--nonlinear-sorp-omega-min", type=float, default=1.0)
    parser.add_argument("--nonlinear-sorp-omega-max", type=float, default=3.5)
    parser.add_argument("--nonlinear-sorh-omega-min", type=float, default=0.2)
    parser.add_argument("--nonlinear-sorh-omega-max", type=float, default=1.2)
    parser.add_argument("--nonlinear-omega-coarse-step", type=float, default=0.01)
    parser.add_argument("--exploratory-iter-max", type=int, default=50000)
    parser.add_argument("--exploratory-failsafe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--failsafe-window", type=int, default=1000)
    parser.add_argument("--failsafe-min-iter", type=int, default=2000)
    parser.add_argument("--failsafe-check-every", type=int, default=250)
    parser.add_argument("--failsafe-min-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--failsafe-residual-floor", type=float, default=1e-14)

    parser.add_argument("--figure4-N", type=int, default=129)
    parser.add_argument("--sensitivity-tol-update", type=float, default=1e-7)
    parser.add_argument("--sensitivity-iter-max", type=int, default=10000)
    parser.add_argument("--sensitivity-coarse-step", type=float, default=0.005)
    parser.add_argument("--sensitivity-fine-step", type=float, default=0.001)
    parser.add_argument("--sensitivity-fine-width", type=float, default=0.015)
    parser.add_argument("--sorh-sensitivity-min", type=float, default=0.722)
    parser.add_argument("--sorh-sensitivity-max", type=float, default=0.993)
    parser.add_argument("--sorp-sensitivity-min", type=float, default=2.265)
    parser.add_argument("--sorp-sensitivity-max", type=float, default=2.682)

    parser.add_argument("--error-tol-update", type=float, default=1e-10)
    parser.add_argument("--error-iter-max", type=int, default=300000)
    parser.add_argument("--error-plateau-window", type=int, default=25)
    parser.add_argument("--error-plateau-rtol", type=float, default=1e-3)
    parser.add_argument("--error-plateau-atol", type=float, default=1e-14)
    parser.add_argument(
        "--figure5-sorp-omegas",
        default=format_omega_map(FIGURE5_DEFAULT_SORP_OMEGAS),
        help="Comma-separated N:omega map for Figure 5 SORp runs.",
    )
    parser.add_argument(
        "--figure5-sorh-omegas",
        default=format_omega_map(FIGURE5_DEFAULT_SORH_OMEGAS),
        help="Comma-separated N:omega map for Figure 5 SORh runs.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    args.figures = parse_figures(args.figures)
    args.Ns = tuple(args.Ns)
    args.figure5_sorp_omegas = parse_omega_map(args.figure5_sorp_omegas, "--figure5-sorp-omegas")
    args.figure5_sorh_omegas = parse_omega_map(args.figure5_sorh_omegas, "--figure5-sorh-omegas")
    if "figure4" in args.figures and args.figure4_N not in args.Ns:
        raise ValueError("--figure4-N must be included in --Ns so its optimized omega is available.")
    if args.spectral_omega_step <= 0:
        raise ValueError("--spectral-omega-step must be positive.")
    if args.exploratory_iter_max <= 0:
        raise ValueError("--exploratory-iter-max must be positive.")
    if args.failsafe_window <= 0:
        raise ValueError("--failsafe-window must be positive.")
    if args.failsafe_min_iter <= 0:
        raise ValueError("--failsafe-min-iter must be positive.")
    if args.failsafe_check_every <= 0:
        raise ValueError("--failsafe-check-every must be positive.")
    if args.failsafe_min_relative_improvement < 0:
        raise ValueError("--failsafe-min-relative-improvement must be nonnegative.")
    if args.error_tol_update <= 0:
        raise ValueError("--error-tol-update must be positive.")
    if args.error_iter_max <= 0:
        raise ValueError("--error-iter-max must be positive.")
    if args.error_plateau_window <= 0:
        raise ValueError("--error-plateau-window must be positive.")
    if args.error_plateau_rtol < 0:
        raise ValueError("--error-plateau-rtol must be nonnegative.")
    if args.error_plateau_atol <= 0:
        raise ValueError("--error-plateau-atol must be positive.")
    missing_sorp = sorted(set(args.Ns) - set(args.figure5_sorp_omegas))
    if missing_sorp:
        raise ValueError(f"--figure5-sorp-omegas is missing N values: {missing_sorp}")
    missing_sorh = sorted(set(args.Ns) - set(args.figure5_sorh_omegas))
    if missing_sorh:
        raise ValueError(f"--figure5-sorh-omegas is missing N values: {missing_sorh}")
    args.figure3_problem = "linear"
    args.figure4_problem = "nonlinear"
    args.progress = Progress(enabled=not args.quiet_progress, every=args.progress_every)
    return args


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records_path = output_dir / "raw_records.jsonl"
    if raw_records_path.exists() and args.overwrite_output:
        raw_records_path.unlink()

    write_metadata(args, output_dir)
    args.progress.log(
        f"collection start: figures={','.join(sorted(args.figures))} "
        f"Ns={list(args.Ns)} output={output_dir}"
    )
    best_by_key = collect_optima(args, output_dir, raw_records_path)

    if "figure1" in args.figures:
        collect_performance("linear", "figure1", args, best_by_key, output_dir, raw_records_path)
    if "figure2" in args.figures:
        collect_performance("nonlinear", "figure2", args, best_by_key, output_dir, raw_records_path)
    if "figure3" in args.figures:
        collect_figure3(args, best_by_key, output_dir)
    if "figure4" in args.figures:
        collect_figure4(args, best_by_key, output_dir, raw_records_path)
    if "figure5" in args.figures:
        collect_figure5(args, output_dir, raw_records_path)

    args.progress.log(f"collection done: wrote data to {output_dir}")


if __name__ == "__main__":
    main()
