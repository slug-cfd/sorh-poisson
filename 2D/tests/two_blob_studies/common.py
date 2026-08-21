"""Shared configuration, execution, and output helpers for two-blob studies."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    ParabolicUnsplitConfig,
    ProblemConfig,
    RunConfig,
    StoppingConfig,
    compute_omega_opt,
    initialize_simulation,
)
from methods.poisson.poisson_helpers import get_all_laplace
from problems import poisson
from problems.poisson_tests import get_poisson_problem


METHODS = ("h_unsplit", "p_unsplit")
RESULTS_ROOT = BASE_DIR / "results" / "two_blob_studies"
DEFAULT_SETTINGS = Path(__file__).resolve().parent / "study_defaults.json"


def load_settings(path=DEFAULT_SETTINGS):
    path = Path(path)
    with path.open() as stream:
        settings = json.load(stream)

    missing_methods = set(METHODS) - set(settings["methods"])
    if missing_methods:
        raise ValueError(f"Missing method settings: {sorted(missing_methods)}")
    if any(int(N) < 1 for N in settings["grid_sizes"]):
        raise ValueError("All grid sizes must be positive interior counts.")
    if any(float(cfl) <= 0.0 for cfl in settings["cfl_values"]):
        raise ValueError("All CFL values must be positive.")
    if settings["problem"]["initial_guess"] != "zero":
        raise ValueError("The two-blob collectors currently require a zero initial guess.")
    search = settings["omega_search"]
    if float(search["relative_step"]) <= 0.0:
        raise ValueError("omega_search.relative_step must be positive.")
    if float(search["minimum_step"]) <= 0.0:
        raise ValueError("omega_search.minimum_step must be positive.")
    if int(search["max_steps_per_side"]) < 1:
        raise ValueError("omega_search.max_steps_per_side must be at least one.")
    if int(search["increase_patience"]) < 1:
        raise ValueError("omega_search.increase_patience must be at least one.")
    if int(settings["performance"]["repetitions"]) < 2:
        raise ValueError("Performance timing requires at least two repetitions.")
    accuracy = settings["accuracy"]
    if float(accuracy["tol_residual"]) <= 0.0:
        raise ValueError("accuracy.tol_residual must be positive.")
    if float(accuracy["tol_update"]) <= 0.0:
        raise ValueError("accuracy.tol_update must be positive.")
    if int(accuracy["iter_max"]) < 1:
        raise ValueError("accuracy.iter_max must be at least one.")
    if float(accuracy["algebraic_error_fraction_max"]) <= 0.0:
        raise ValueError("accuracy.algebraic_error_fraction_max must be positive.")
    return settings, path.resolve()


def make_config(settings, method, N, cfl, omega):
    solver = settings["solver"]
    problem = settings["problem"]
    h_settings = settings["methods"]["h_unsplit"]
    p_settings = settings["methods"]["p_unsplit"]

    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case=problem["name"]),
            grid=GridConfig(
                Nx=int(N),
                Ny=int(N),
                laplacian_stencil=problem["laplacian_stencil"],
                ghost_layers=problem["ghost_layers"],
            ),
            run=RunConfig(
                method=method,
                cfl=float(cfl),
                omega=float(omega),
                use_omega_opt=False,
                iter_max=int(solver["iter_max"]),
                norm=solver["norm"],
                stopping=StoppingConfig(
                    tol_update=float(solver["tol_update"]),
                    tol_residual=float(solver["tol_residual"]),
                    res_max=float(solver["res_max"]),
                ),
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(mode=h_settings["mode"]),
                p_unsplit=ParabolicUnsplitConfig(
                    gs_ordering=p_settings["gs_ordering"]
                ),
            ),
        )
    )


def formula_omega(settings, method, N, cfl):
    manual = settings["methods"][method]["manual_omega"]
    config = make_config(settings, method, N, cfl, manual)
    return float(compute_omega_opt(config.settings, config.grid))


def _last(values):
    return float(values[-1]) if len(values) else math.nan


def run_poisson_case(config):
    """Time one solve; configuration construction is intentionally excluded."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with contextlib.redirect_stdout(io.StringIO()):
        U, pseudo_time, iter_count, histories = poisson.solve_all(config)
    cpu_time = time.process_time() - cpu_start
    wall_time = time.perf_counter() - wall_start

    iterations = len(histories["residual"])
    if int(iter_count) != iterations:
        raise RuntimeError(
            f"Solver iteration mismatch: iter_count={iter_count}, history={iterations}."
        )
    return {
        "N": config.Nx,
        "method": config.method,
        "cfl": config.cfl,
        "omega": config.omega,
        "iterations": iterations,
        "wall_time_sec": wall_time,
        "cpu_time_sec": cpu_time,
        "seconds_per_iteration": wall_time / max(1, iterations),
        "pseudo_time": float(pseudo_time),
        "dt": _last(histories["dt"]),
        "converged": bool(histories["converged"]),
        "stable": bool(histories["stable"]),
        "reason": histories["reason"],
        "update_final": _last(histories["update"]),
        "residual_final": _last(histories["residual"]),
        "exact_final": _last(histories["exact"]),
        "best_exact": float(histories["best_exact"]),
        "best_exact_iter": int(histories["best_iter"]),
    }, U, histories


def norm_metrics(values):
    values = np.asarray(values, dtype=float).ravel()
    absolute = np.abs(values)
    return {
        "l1": float(np.mean(absolute)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "linf": float(np.max(absolute)),
    }


def accuracy_row(config, U, histories):
    interior = (
        slice(config.ib, config.ie + 1),
        slice(config.jb, config.je + 1),
    )
    error = (U - config.U_exact)[interior]
    equation_residual = (
        get_all_laplace(U, config, config.laplacian_stencil) - config.f_all
    )[interior]
    truncation_error = (
        get_all_laplace(
            config.U_exact,
            config,
            config.laplacian_stencil,
        ) - config.f_all
    )[interior]
    error_metrics = norm_metrics(error)
    residual_metrics = norm_metrics(equation_residual)
    truncation_metrics = norm_metrics(truncation_error)
    lambda_min = (
        4.0 / config.dx**2
        * np.sin(np.pi / (2.0 * (config.Nx + 1))) ** 2
        + 4.0 / config.dy**2
        * np.sin(np.pi / (2.0 * (config.Ny + 1))) ** 2
    )
    algebraic_bound = residual_metrics["rms"] / lambda_min
    algebraic_fraction = algebraic_bound / max(error_metrics["rms"], 1e-300)

    return {
        "N": config.Nx,
        "method": config.method,
        "cfl": config.cfl,
        "omega": config.omega,
        "dx": config.dx,
        "dy": config.dy,
        "iterations": len(histories["residual"]),
        "converged": bool(histories["converged"]),
        "stable": bool(histories["stable"]),
        "reason": histories["reason"],
        "error_l1": error_metrics["l1"],
        "error_rms": error_metrics["rms"],
        "error_linf": error_metrics["linf"],
        "residual_l1": residual_metrics["l1"],
        "residual_rms": residual_metrics["rms"],
        "residual_linf": residual_metrics["linf"],
        "truncation_l1": truncation_metrics["l1"],
        "truncation_rms": truncation_metrics["rms"],
        "truncation_linf": truncation_metrics["linf"],
        "discrete_lambda_min": float(lambda_min),
        "algebraic_error_bound_rms": float(algebraic_bound),
        "algebraic_to_solution_error": float(algebraic_fraction),
        "best_error_rms": float(histories["best_exact"]),
        "best_error_iter": int(histories["best_iter"]),
    }


def successful(row):
    return as_bool(row["converged"]) and as_bool(row["stable"])


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(study, output_dir=None, resume=False):
    if output_dir is None:
        if resume:
            raise ValueError("--resume requires an explicit --output-dir.")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = RESULTS_ROOT / study / stamp
    else:
        output_dir = Path(output_dir)

    if resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --resume or another path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_global_metadata(
    output_dir,
    study,
    settings,
    settings_path,
    study_axes,
    study_options,
):
    problem = get_poisson_problem(settings["problem"]["name"])
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "study": study,
        "first_written_at": timestamp,
        "last_written_at": timestamp,
        "settings_source": str(settings_path),
        "settings_sha256": file_sha256(settings_path),
        "study_axes": study_axes,
        "study_options": study_options,
        "fixed_parameters": {
            "problem": "poisson",
            "poisson_problem": problem.name,
            "domain": problem.domain,
            "problem_constants": problem.constants,
            "boundary_condition": "analytic Dirichlet data",
            "initial_guess": settings["problem"]["initial_guess"],
            "laplacian_stencil": settings["problem"]["laplacian_stencil"],
            "ghost_layers": settings["problem"]["ghost_layers"],
            "multigrid": False,
            "norm": settings["solver"]["norm"],
            "tol_residual": settings["solver"]["tol_residual"],
            "tol_update_stagnation": settings["solver"]["tol_update"],
            "res_max": settings["solver"]["res_max"],
            "iter_max": settings["solver"]["iter_max"],
            "methods": settings["methods"],
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "timer": "time.perf_counter",
            "cpu_timer": "time.process_time",
        },
    }
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "global_config.json"
    if metadata_path.exists():
        with metadata_path.open() as stream:
            existing = json.load(stream)
        for key in ("study", "settings_sha256", "study_axes", "study_options"):
            if existing.get(key) != payload.get(key):
                raise ValueError(
                    f"Cannot resume with changed global metadata field '{key}'."
                )
        payload["first_written_at"] = existing.get("first_written_at", timestamp)
    with metadata_path.open("w") as stream:
        json.dump(payload, stream, indent=2, default=float)


def sort_rows(rows, fields):
    def key(row):
        values = []
        for field in fields:
            value = row[field]
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(str(value))
        return tuple(values)

    return sorted(rows, key=key)
