import argparse
from datetime import datetime
import time
from pathlib import Path

from configuration import initialize_simulation, load_config
from problems import linear_diffusion, nonlinear_diffusion, poisson
from runner import run_simulation, save_record


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "configs" / "config_sample.json"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"


def _plot_solution(solution, iteration_count, history, config, history_label):
    from diagnostics.plot_solutions import plot_solution

    plot_solution(
        solution,
        iteration_count,
        history,
        config,
        history_label=history_label,
    )


def solve_poisson(config, plot=False):
    if config.problem != "poisson" or config.multigrid:
        raise ValueError("The base Poisson runner requires problem='poisson' and multigrid=false.")

    start = time.perf_counter()
    solution, pseudo_time, iteration_count, histories = poisson.solve_all(config)
    elapsed = time.perf_counter() - start

    print("method:", config.method)
    print("iterations:", iteration_count)
    print("final equation residual:", histories["residual"][-1])
    print("final update residual:", histories["update"][-1])
    print("final exact error:", histories["exact"][-1])
    print(f"best exact error: {histories['best_exact']} | iteration {histories['best_iter']}")
    print("stop reason:", histories["reason"])
    print("pseudo-time:", pseudo_time)
    print("elapsed time:", elapsed)

    if plot:
        _plot_solution(
            solution,
            iteration_count,
            histories["residual"],
            config,
            "Equation residual",
        )
    return solution, pseudo_time, iteration_count, histories


def solve_poisson_multigrid(config, plot=False):
    if config.problem != "poisson" or not config.multigrid:
        raise ValueError("The multigrid runner requires problem='poisson' and multigrid=true.")

    solution, pseudo_time, cycle_count, histories = poisson.solve_all(config)
    print("method:", config.method)
    print("cycle count:", cycle_count)
    print("final equation residual:", histories["residual"][-1])
    print("pseudo-time:", pseudo_time)

    if plot:
        _plot_solution(
            solution,
            cycle_count,
            histories["residual"],
            config,
            "Equation residual",
        )
    return solution, pseudo_time, cycle_count, histories


def solve_nonlinear_diffusion(config, plot=False):
    if config.problem != "nonlinear_diffusion":
        raise ValueError("The nonlinear runner requires problem='nonlinear_diffusion'.")

    start = time.perf_counter()
    solution, pseudo_time, iteration_count, histories = nonlinear_diffusion.solve_all(config)
    elapsed = time.perf_counter() - start

    print("method:", config.method)
    print("iterations:", iteration_count)
    print("final equation residual:", histories["residual"][-1])
    print("final update residual:", histories["update"][-1])
    print(f"best equation residual: {histories['best_residual']} | iteration {histories['best_iter']}")
    print("stop reason:", histories["reason"])
    print("pseudo-time:", pseudo_time)
    print("elapsed time:", elapsed)

    if plot:
        _plot_solution(
            solution,
            iteration_count,
            histories["residual"],
            config,
            "Equation residual",
        )
    return solution, pseudo_time, iteration_count, histories


def solve_linear_diffusion(config, plot=False):
    if config.problem != "linear_diffusion":
        raise ValueError("The linear runner requires problem='linear_diffusion'.")

    solution, pseudo_time, iteration_count, histories = linear_diffusion.solve(config)
    if plot:
        _plot_solution(
            solution,
            iteration_count,
            histories["update"],
            config,
            "Update norm",
        )
    return solution, pseudo_time, iteration_count, histories


SOLVERS = {
    "poisson_base": solve_poisson,
    "poisson_mg": solve_poisson_multigrid,
    "nonlinear_diffusion": solve_nonlinear_diffusion,
    "linear_diffusion": solve_linear_diffusion,
}


def _solver_name(config):
    if config.problem == "poisson":
        return "poisson_mg" if config.multigrid else "poisson_base"
    return config.problem


def parse_args():
    parser = argparse.ArgumentParser(description="Run a config-driven 2D solver.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot-exact", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {args.output}")
    settings = load_config(args.config)
    config = initialize_simulation(settings)
    plot = config.plot if args.plot is None else args.plot
    plot_exact = config.plot_exact if args.plot_exact is None else args.plot_exact

    if plot_exact:
        from diagnostics.plot_solutions import plot_exact_solution

        plot_exact_solution(config)

    solver = SOLVERS[_solver_name(config)]
    record = run_simulation(config, solver, plot)
    output = args.output or (
        DEFAULT_RESULTS_DIR / "generated_runs"
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
    )
    save_record(record, output, overwrite=args.overwrite)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
