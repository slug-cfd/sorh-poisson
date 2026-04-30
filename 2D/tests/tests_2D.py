from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_record import *
from config import HyperbolicUnsplitConfig, MultiGridConfig

def run_simulation(config: Config, solver_func, plot=False):
    start = time.time()

    U, t, iter_count, histories = solver_func(config, plot=plot)

    elapsed = time.time() - start

    residuals = histories["exact"]

    tol = 1e-5

    result = RunResult(
        converged=residuals[-1] < tol,
        stable=max(residuals) < 1e6,
        iter_count=iter_count,
        residual_final=residuals[-1],
        residual_min=min(residuals),
        residual_max=max(residuals),
        elapsed_sec=elapsed
    )

    record = RunRecord(
        config=config,
        results=result,
        meta={"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    )

    return record


def run_compare(config1: Config, config2: Config, solver1, solver2):
    run1 = run_simulation(config1, solver1)
    # time.sleep(1)
    run2 = run_simulation(config2, solver2)

    method1 = run1.config.method
    method2 = run2.config.method

    elapsed1 = run1.results.elapsed_sec
    elapsed2 = run2.results.elapsed_sec

    print(f"\nComparison: {method1} vs. {method2}")
    print(f"{method1}: {elapsed1:.8f} seconds")
    print(f"{method2}: {elapsed2:.8f} seconds")
    max_time = max(elapsed1, elapsed2)
    min_time = min(elapsed1, elapsed2)

    print(f"speedup: {max_time / min_time:.8f}")


def run_poisson_problem_smoke(solver, poisson_problem="two_blob", method="p_unsplit"):
    config = Config(
        problem="poisson",
        poisson_problem=poisson_problem,
        method=method,
        bc_type="poisson",
        Nx=32,
        Ny=32,
        cfl=0.8,
        omega=1.0,
        iter_max=100,
        norm="fro",
        mg_config=MultiGridConfig(),
        h_unsplit_config=HyperbolicUnsplitConfig(),
        multigrid=False,
    )
    return run_simulation(config, solver, plot=False)


# plot phi_exact for Poisson test eqn
# plot_exact_solution(config)
