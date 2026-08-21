"""Small reusable smoke helpers for the 2D solvers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configuration import (
    Config,
    GridConfig,
    ProblemConfig,
    RunConfig,
    initialize_simulation,
)
from runner import run_simulation


def run_poisson_problem_smoke(solver, poisson_problem="two_blob", method="p_unsplit"):
    config = initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case=poisson_problem),
            grid=GridConfig(Nx=32, Ny=32),
            run=RunConfig(
                method=method,
                cfl=0.8,
                omega=1.0,
                iter_max=100,
                norm="fro",
            ),
        )
    )
    return run_simulation(config, solver, plot=False)


if __name__ == "__main__":
    from main import solve_poisson

    record = run_poisson_problem_smoke(solve_poisson)
    print(
        f"smoke: stable={record.results.stable}, "
        f"iterations={record.results.iter_count}, "
        f"reason={record.results.reason}"
    )
