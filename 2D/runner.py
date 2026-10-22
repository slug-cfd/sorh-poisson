"""Production run orchestration and JSON result recording."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from configuration import SimulationContext
from run_record import RunRecord, RunResult


def run_simulation(config: SimulationContext, solver_func, plot=False):
    start = time.perf_counter()
    _, pseudo_time, iteration_count, histories = solver_func(config, plot=plot)
    elapsed = time.perf_counter() - start

    residuals = histories.get("residual", histories.get("update"))
    if residuals is None or len(residuals) == 0:
        raise ValueError("Solver histories must contain a nonempty residual or update history.")

    converged = bool(
        histories.get(
            "converged",
            residuals[-1] < config.stopping_config.tol_residual,
        )
    )
    stable = bool(
        histories.get(
            "stable",
            np.all(np.isfinite(residuals))
            and np.max(np.abs(residuals)) <= config.stopping_config.res_max,
        )
    )
    reason = histories.get("reason", "tol_residual" if converged else "unknown")

    result = RunResult(
        converged=converged,
        stable=stable,
        iter_count=iteration_count,
        residual_final=float(residuals[-1]),
        residual_min=float(np.min(residuals)),
        residual_max=float(np.max(residuals)),
        elapsed_sec=elapsed,
        reason=reason,
    )
    return RunRecord(
        config=config,
        results=result,
        meta={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_update": float(histories["update"][-1]),
            "final_residual": float(residuals[-1]),
            "pseudo_time": float(pseudo_time),
        },
    )


def run_compare(
    config1: SimulationContext,
    config2: SimulationContext,
    solver1,
    solver2,
):
    run1 = run_simulation(config1, solver1)
    run2 = run_simulation(config2, solver2)
    speedup = max(run1.results.elapsed_sec, run2.results.elapsed_sec) / min(
        run1.results.elapsed_sec,
        run2.results.elapsed_sec,
    )
    return run1, run2, speedup


def _json_default(obj):
    if callable(obj):
        return obj.__name__
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_record(record: RunRecord, filename: str | Path, *, overwrite: bool = False):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "config": record.config.to_dict(),
        "results": asdict(record.results),
        "meta": record.meta,
    }
    with filename.open("w" if overwrite else "x") as stream:
        json.dump(row, stream, default=_json_default)
        stream.write("\n")
