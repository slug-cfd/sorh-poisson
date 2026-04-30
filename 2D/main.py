import os
import time
import json
from pathlib import Path

import load_config as lcf
from run_record import *
from diagnostics.plot_solutions import *

from tests import tests_2D as tests
from problems import poisson2D, nonlinear2D, linear2D


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config_sample.json"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"


problem_map = {
    'poisson': poisson2D,
    'toy_nonlinear': nonlinear2D,
    'toy_linear': linear2D,
}

def solve_p2D(config, plot=False):
    if config.problem != "poisson" or config.multigrid:
        print("invalid problem for poisson solver!")
        return 0

    start = time.perf_counter()

    U, t, iter_count, histories = poisson2D.solve_all(config)

    end = time.perf_counter()
    elapsed = end - start

    ### histories structure ###
    # histories = {
    #     "update": np.array(update_hist),
    #     "residual": np.array(residual_hist),
    #     "exact": np.array(exact_hist),
    #     "best_exact": best_exact,
    #     "best_iter": best_iter,
    # }

    conv_history = histories["exact"]
    res_final = conv_history[-1]

    print("method:", config.method)
    print("iterations:", iter_count)
    print("final residual: ", res_final)
    print(f"best residual: {histories["best_exact"]} | iteration {histories["best_iter"]}")
    # print("time t: ", t)
    print("elapsed time: ", elapsed)

    if plot:
        plot_solution(U, iter_count, conv_history, config)

    return U, t, iter_count, histories


def solve_p2d_multigrid(config, plot=False):
    if config.problem != "poisson" or not config.multigrid:
        print("invalid problem for poisson solver!")
        return 0

    U, t, cycle_count, histories = poisson2D.solve_all(config)

    ### histories structure ###
    # histories = {
    #     "update": np.array(update_hist),
    #     "residual": np.array(residual_hist),
    #     "exact": np.array(exact_hist),
    #     "cycle_time": np.array(cycle_time_hist),
    #     "best_exact": best_exact,
    #     "best_iter": best_iter,
    #     "elapsed": elapsed,
    #     "cycles": iter_count,
    #     "U_best": U_best,
    # }

    conv_history = histories["exact"]
    res_final = conv_history[-1]

    print()
    print("method:", config.method)
    print("total time: ", t)
    print("cycle count:", cycle_count)
    print("final residual: ", res_final)

    if plot:
        plot_solution(U, cycle_count, conv_history, config)

    return U, t, cycle_count, histories


def solve_nl2D(config, plot=False):
    if config.problem != "nonlinear":
        print("invalid problem for poisson solver!")
        return 0

    U, t, iter_count, histories = nonlinear2D.solve_all(config)

    ### histories structure ###
    # histories = {
    #     "update": np.array(update_hist),
    #     "U_res_min": np.array(U_res_min),
    #     "best_update": best_update,
    #     "best_iter": best_iter,
    # }

    print("method:", config.method)
    print("iterations:", iter_count)
    print("final residual: ", histories["update_hist"][-1])
    print(f"best residual: {histories["best_update"]} | iteration {histories["best_iter"]}")
    print("time t: ", t)

    if plot:
        plot_solution(U, iter_count, histories["update_hist"], config)

    return U, t, iter_count, histories


def solve_l2D(config, plot=False):
    if config.problem != "linear":
        print("invalid problem for poisson solver!")
        return 0

    U, t, iter_count, res, err_conv = linear2D.solve(config)

    if plot:
        plot_solution(U, iter_count, err_conv, config)

    return U, t, iter_count, res, err_conv


solver_map = {
    "poisson_base": solve_p2D,
    "poisson_mg": solve_p2d_multigrid,
    "toy_nonlinear": solve_nl2D,
    "toy_linear": solve_l2D,
}


def _get_solver_name(config):
    p, mg = config.problem, config.multigrid
    if p == "poisson":
        return "poisson_mg" if mg else "poisson_base"
    return p


def _json_default(obj):
    if callable(obj):
        return obj.__name__ if obj is not None else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):      # catches ALL numpy scalars at once
        return obj.item()                # converts to native Python type
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_record(record: RunRecord, filename: str):
    row = {
        "config":  record.config.to_dict(),
        "results": asdict(record.results),
        "meta":    record.meta,
    }
    with open(filename, "w") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")  # <-- make sure this is here


if __name__ == "__main__":
    plot = True
    plot_exact = False

    config1 = lcf.config_from_json(DEFAULT_CONFIG)

    if plot_exact:
        plot_exact_solution(config1)

    solver = solver_map[_get_solver_name(config1)]

    # run the simulation
    record = tests.run_simulation(config1, solver, plot)

    save_dir = DEFAULT_RESULTS_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_record(record, save_dir / "test_record.json")

# compare execution time between two methods/configurations
# tests.run_compare(config1, config1, solver, solver)
