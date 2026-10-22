import argparse
import json
from pathlib import Path

import numpy as np

from load_config1D import config_from_json
from diagnostics.plot1D import plot_solution
from solver1D import run_simulation


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "configs/config_sample1D.json"
DEFAULT_OUTPUT = BASE_DIR / "results" / "latest_run.json"


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_record(record, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w") as fh:
        fh.write(json.dumps(record.to_dict(), default=_json_default) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run a config-driven 1D solver.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--plot", action="store_true", help="Force plotting on.")
    args = parser.parse_args()

    config = config_from_json(args.config)
    if args.plot:
        config.plot_enabled = True

    U, t, iter_count, histories, record = run_simulation(config)

    print("---", config.run_display_name, "---")
    # print("method:", config.method)
    # print("problem:", config.problem_display_name)
    # print()
    print("omega: ", config.omega)
    print("interior size: ", config.interior_size)
    print("iterations:", iter_count)
    print()
    print("final update residual:", record.results.residual_final)
    if record.results.equation_residual_final is not None:
        print("final equation residual:", record.results.equation_residual_final)
    print("final exact diagnostic:", record.meta["exact_final"])
    print("best exact diagnostic:", record.meta["best_exact"], "| iteration", record.meta["best_iter"])
    print()
    print("pseudo-time t:", t)
    print("elapsed time:", record.results.elapsed_sec)
    print("stop reason:", record.results.reason)

    save_record(record, args.output)

    if config.plot_enabled:
        plot_solution(U, iter_count, histories, config)


if __name__ == "__main__":
    main()
