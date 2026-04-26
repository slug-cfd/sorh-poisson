import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config1D import Config1D, Solver1DConfig
from solver1D import run_simulation


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

SOR_METHODS = (
    "sor_linear_parabolic",
    "sor_linear_hyperbolic",
    "sor_nonlinear_parabolic",
    "sor_nonlinear_hyperbolic",
)

BASELINE_METHODS = (
    "jacobi_linear_parabolic",
    "gs_linear_parabolic",
    "jacobi_nonlinear_parabolic",
    "gs_nonlinear_parabolic",
)

ALL_BENCHMARK_METHODS = BASELINE_METHODS + SOR_METHODS

FIELDNAMES = (
    "problem",
    "method",
    "scheme",
    "N",
    "cfl",
    "omega",
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


def _problem_for_method(method):
    return "nonlinear" if "nonlinear" in method else "linear"


def _scheme_for_method(method):
    if "hyperbolic" in method:
        return "hyperbolic"
    if "parabolic" in method:
        return "parabolic"
    return "baseline"


def make_config(
    method="sor_nonlinear_hyperbolic",
    N=32,
    iter_max=20000,
    omega=0.9,
    cfl=0.8,
    tol_update=1e-7,
    res_max=1e8,
):
    return Config1D(
        problem=_problem_for_method(method),
        method=method,
        N=N,
        cfl=cfl,
        omega=omega,
        iter_max=iter_max,
        norm=2,
        solver_config=Solver1DConfig(tol_update=tol_update, res_max=res_max),
    )


def smoke_config(method="sor_nonlinear_hyperbolic", N=32, iter_max=5000, omega=0.9):
    return make_config(method=method, N=N, iter_max=iter_max, omega=omega)


def run_smoke(method="sor_nonlinear_hyperbolic", N=32, iter_max=5000, omega=0.9):
    return run_simulation(smoke_config(method, N, iter_max, omega))


def run_compare(config1, config2):
    _, _, _, _, record1 = run_simulation(config1)
    _, _, _, _, record2 = run_simulation(config2)
    return record1, record2


def _row_from_record(record):
    config = record.config
    return {
        "problem": config.problem,
        "method": config.method,
        "scheme": _scheme_for_method(config.method),
        "N": config.N,
        "cfl": config.cfl,
        "omega": config.omega,
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


def run_case(method, N, omega=1.0, iter_max=20000, cfl=0.8, tol_update=1e-7):
    config = make_config(
        method=method,
        N=N,
        omega=omega,
        iter_max=iter_max,
        cfl=cfl,
        tol_update=tol_update,
    )
    _, _, _, _, record = run_simulation(config)
    return _row_from_record(record)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    with Path(path).open() as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _truthy(value):
    return value is True or value == "True"


def generate_omega_map(
    config: Config1D,
    omega_min=0.2,
    omega_max=4.0,
    d_omega=0.2,
    output_dir=None,
):
    output_dir = RESULTS_DIR if output_dir is None else Path(output_dir)
    omegas = np.arange(omega_min, omega_max + 0.5 * d_omega, d_omega)
    rows = [
        run_case(
            config.method,
            config.N,
            omega=float(omega),
            iter_max=config.iter_max,
            cfl=config.cfl,
            tol_update=config.solver_config.tol_update,
        )
        for omega in omegas
    ]

    filename = output_dir / f"map_{config.method}_omega_vs_iter_count_n{config.N}.csv"
    _write_rows(filename, rows)
    return filename


def _dedupe_omegas(omegas):
    seen = set()
    result = []
    for omega in omegas:
        key = round(float(omega), 10)
        if key not in seen:
            seen.add(key)
            result.append(float(omega))
    return result


def _refinement_omegas(coarse_rows, method, N, omega_step):
    candidates = [
        row for row in coarse_rows
        if row["method"] == method
        and int(row["N"]) == N
        and _truthy(row["converged"])
        and _truthy(row["stable"])
    ]
    if not candidates:
        return []

    best = min(candidates, key=lambda row: (int(row["iter_count"]), float(row["elapsed_sec"])))
    center = float(best["omega"])
    lower = max(0.05, center - omega_step)
    upper = min(4.0, center + omega_step)
    existing = {round(float(row["omega"]), 10) for row in coarse_rows if row["method"] == method and int(row["N"]) == N}

    fine = np.arange(lower, upper + 0.0001, omega_step / 8.0)
    return [
        float(omega) for omega in fine
        if round(float(omega), 10) not in existing
    ]


def generate_experiment(
    output_dir=None,
    Ns=(16, 32, 64),
    tol_update=1e-7,
    iter_max=20000,
    cfl=0.8,
    omega_min=0.2,
    omega_max=4.0,
    omega_step=0.2,
):
    output_dir = RESULTS_DIR / "sor_performance" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "experiment_config.json"
    metadata = {
        "Ns": list(Ns),
        "tol_update": tol_update,
        "iter_max": iter_max,
        "cfl": cfl,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "omega_step": omega_step,
        "sor_methods": SOR_METHODS,
        "baseline_methods": BASELINE_METHODS,
    }
    config_path.write_text(json.dumps(metadata, indent=2) + "\n")

    baseline_rows = []
    for N in Ns:
        for method in ALL_BENCHMARK_METHODS:
            baseline_rows.append(
                run_case(
                    method=method,
                    N=N,
                    omega=1.0,
                    iter_max=iter_max,
                    cfl=cfl,
                    tol_update=tol_update,
                )
            )
    _write_rows(output_dir / "baseline_omega1.csv", baseline_rows)

    coarse_rows = []
    coarse_omegas = _dedupe_omegas(np.arange(omega_min, omega_max + 0.5 * omega_step, omega_step))
    for N in Ns:
        for method in SOR_METHODS:
            for omega in coarse_omegas:
                coarse_rows.append(
                    run_case(
                        method=method,
                        N=N,
                        omega=omega,
                        iter_max=iter_max,
                        cfl=cfl,
                        tol_update=tol_update,
                    )
                )
    _write_rows(output_dir / "sor_omega_coarse.csv", coarse_rows)

    fine_rows = []
    for N in Ns:
        for method in SOR_METHODS:
            for omega in _refinement_omegas(coarse_rows, method, N, omega_step):
                fine_rows.append(
                    run_case(
                        method=method,
                        N=N,
                        omega=omega,
                        iter_max=iter_max,
                        cfl=cfl,
                        tol_update=tol_update,
                    )
                )
    _write_rows(output_dir / "sor_omega_refined.csv", fine_rows)
    _write_rows(output_dir / "sor_omega_all.csv", coarse_rows + fine_rows)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate 1D method performance data.")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "sor_performance"))
    parser.add_argument("--tol-update", type=float, default=1e-7)
    parser.add_argument("--iter-max", type=int, default=20000)
    parser.add_argument("--Ns", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--omega-min", type=float, default=0.2)
    parser.add_argument("--omega-max", type=float, default=4.0)
    parser.add_argument("--omega-step", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = generate_experiment(
        output_dir=args.output_dir,
        Ns=tuple(args.Ns),
        tol_update=args.tol_update,
        iter_max=args.iter_max,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        omega_step=args.omega_step,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
