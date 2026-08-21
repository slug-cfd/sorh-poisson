from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    ProblemConfig,
    RunConfig,
    SimulationContext,
    initialize_simulation,
)
from problems import poisson


@dataclass
class BenchmarkRow:
    mode: str
    elapsed_sec: float
    solver_iter_count: int
    sweeps: int
    update_final: float
    poisson_residual_final: float
    exact_error_final: float
    update_min: float
    poisson_residual_min: float
    exact_error_min: float


def make_config(
    mode: str,
    nx: int,
    ny: int,
    cfl: float,
    omega: float,
    iter_max: int,
    poisson_problem: str,
    ghost_layers: int | None,
    laplacian_stencil: str,
    blend_eta: float,
) -> SimulationContext:
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case=poisson_problem),
            grid=GridConfig(
                Nx=nx,
                Ny=ny,
                laplacian_stencil=laplacian_stencil,
                ghost_layers=ghost_layers,
            ),
            run=RunConfig(
                method="h_unsplit",
                cfl=cfl,
                omega=omega,
                iter_max=iter_max,
                norm="fro",
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(
                    mode=mode,
                    default2_blend_eta=blend_eta,
                ),
            ),
        )
    )


def run_case(config: SimulationContext, trials: int) -> BenchmarkRow:
    best_elapsed = np.inf
    best_result = None

    for _ in range(trials):
        start = time.perf_counter()
        _, _, iter_count, histories = poisson.solve_all(config)
        elapsed = time.perf_counter() - start

        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_result = (iter_count, histories)

    iter_count, histories = best_result
    update = histories["update"]
    residual = histories["residual"]
    exact = histories["exact"]

    return BenchmarkRow(
        mode=config.h_unsplit_config.mode,
        elapsed_sec=best_elapsed,
        solver_iter_count=iter_count,
        sweeps=len(update),
        update_final=update[-1],
        poisson_residual_final=residual[-1],
        exact_error_final=exact[-1],
        update_min=np.min(update),
        poisson_residual_min=np.min(residual),
        exact_error_min=np.min(exact),
    )


def print_rows(rows: list[BenchmarkRow]) -> None:
    headers = (
        "mode",
        "time(s)",
        "iter",
        "sweeps",
        "update_final",
        "poisson_res_final",
        "exact_final",
    )
    print(
        f"{headers[0]:<10} {headers[1]:>10} {headers[2]:>8} {headers[3]:>8} "
        f"{headers[4]:>14} {headers[5]:>18} {headers[6]:>14}"
    )
    for row in rows:
        print(
            f"{row.mode:<10} {row.elapsed_sec:10.6f} "
            f"{row.solver_iter_count:8d} {row.sweeps:8d} "
            f"{row.update_final:14.6e} {row.poisson_residual_final:18.6e} "
            f"{row.exact_error_final:14.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare h_unsplit default and default2 back-back sweeps."
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--omega", type=float, default=0.713)
    parser.add_argument("--iter-max", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--poisson-problem", default="two_blob")
    parser.add_argument("--ghost-layers", type=int, default=None)
    parser.add_argument("--laplacian-stencil", default="centered")
    parser.add_argument("--blend-eta", type=float, default=0.5)
    args = parser.parse_args()

    rows = []
    for mode in ("default", "default2_blend", "default2"):
        config = make_config(
            mode=mode,
            nx=args.nx,
            ny=args.ny,
            cfl=args.cfl,
            omega=args.omega,
            iter_max=args.iter_max,
            poisson_problem=args.poisson_problem,
            ghost_layers=args.ghost_layers,
            laplacian_stencil=args.laplacian_stencil,
            blend_eta=args.blend_eta,
        )
        rows.append(run_case(config, args.trials))

    print_rows(rows)


if __name__ == "__main__":
    main()
