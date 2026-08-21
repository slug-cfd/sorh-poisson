"""One-sweep finiteness checks for every production 2D solver variant."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    MultiGridConfig,
    ParabolicUnsplitConfig,
    ProblemConfig,
    RunConfig,
    initialize_simulation,
)
from problems import linear_diffusion, nonlinear_diffusion, poisson


POISSON_H_MODE_CASES = (
    ("default", "default", 0.25 * np.pi),
    ("red_black", "red_black", 0.25 * np.pi),
    ("default2", "default2", 0.25 * np.pi),
    ("default2_blend", "default2_blend", 0.25 * np.pi),
    ("directional:back_back", "directional", 0.25 * np.pi),
    ("directional:forward_back", "directional", 0.75 * np.pi),
    ("directional:forward_forward", "directional", 1.25 * np.pi),
    ("directional:back_forward", "directional", 1.75 * np.pi),
    ("weighted_hybrid", "weighted_hybrid", 0.25 * np.pi),
)


def poisson_cases():
    yield (
        "p_unsplit:lexicographic",
        "p_unsplit",
        "default",
        "lexicographic",
        0.25 * np.pi,
    )
    yield (
        "p_unsplit:red_black",
        "p_unsplit",
        "default",
        "red_black",
        0.25 * np.pi,
    )
    for label, mode, theta in POISSON_H_MODE_CASES:
        yield f"h_unsplit:{label}", "h_unsplit", mode, "lexicographic", theta
    yield "p_ADI", "p_ADI", "default", "lexicographic", 0.25 * np.pi
    yield "h_ADI", "h_ADI", "default", "lexicographic", 0.25 * np.pi


def make_poisson_config(method, h_mode, p_ordering, theta, *, multigrid=False):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case="two_blob"),
            grid=GridConfig(
                Nx=7 if multigrid else 5,
                Ny=5 if multigrid else 4,
            ),
            run=RunConfig(
                method=method,
                cfl=0.5,
                omega=0.5,
                iter_max=1,
                norm="fro",
                multigrid=multigrid,
            ),
            methods=MethodConfig(
                multigrid=MultiGridConfig(
                    solver="rep_v_cycles",
                    max_cycles=1,
                    nu1=1,
                    nu2=1,
                    min_coarse_n=2,
                    coarse_max_sweeps=1,
                ),
                h_unsplit=HyperbolicUnsplitConfig(mode=h_mode, theta=theta),
                p_unsplit=ParabolicUnsplitConfig(gs_ordering=p_ordering),
            ),
        )
    )


def make_nonlinear_config(
    method,
    p_ordering="lexicographic",
    coefficient_mode="pointwise",
):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(
                kind="nonlinear_diffusion",
                case="unit_square_linear_x",
            ),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method=method,
                cfl=0.5,
                omega=0.5,
                iter_max=1,
                norm="fro",
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(),
                p_unsplit=ParabolicUnsplitConfig(
                    gs_ordering=p_ordering,
                    coefficient_mode=coefficient_mode,
                ),
            ),
        )
    )


def make_linear_config(method):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="linear_diffusion", case="default"),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method=method,
                cfl=0.5,
                omega=0.5,
                iter_max=1,
                norm="fro",
            ),
        )
    )


class SolverFinitenessTests(unittest.TestCase):
    def assert_finite_run(self, config, solution, iteration_count, histories):
        self.assertEqual(iteration_count, 1)
        self.assertEqual(solution.shape, (config.Nx_full, config.Ny_full))
        self.assertTrue(np.all(np.isfinite(solution)))
        for name in ("update", "residual"):
            with self.subTest(history=name):
                self.assertEqual(len(histories[name]), 1)
                self.assertTrue(np.all(np.isfinite(histories[name])))

    def test_poisson_base_methods_modes_and_orderings_are_finite(self):
        for label, method, h_mode, p_ordering, theta in poisson_cases():
            with self.subTest(case=label):
                config = make_poisson_config(method, h_mode, p_ordering, theta)
                solution, _, iterations, histories = poisson.solve_all(config)
                self.assert_finite_run(config, solution, iterations, histories)

    def test_poisson_multigrid_methods_modes_and_orderings_are_finite(self):
        for label, method, h_mode, p_ordering, theta in poisson_cases():
            with self.subTest(case=label):
                config = make_poisson_config(
                    method,
                    h_mode,
                    p_ordering,
                    theta,
                    multigrid=True,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    solution, _, cycles, histories = poisson.solve_all(config)
                self.assert_finite_run(config, solution, cycles, histories)

    def test_nonlinear_methods_and_orderings_are_finite(self):
        cases = (
            (
                "p_unsplit:pointwise:lexicographic",
                "p_unsplit",
                "lexicographic",
                "pointwise",
            ),
            (
                "p_unsplit:pointwise:red_black",
                "p_unsplit",
                "red_black",
                "pointwise",
            ),
            (
                "p_unsplit:frozen_per_sweep:lexicographic",
                "p_unsplit",
                "lexicographic",
                "frozen_per_sweep",
            ),
            (
                "p_unsplit:frozen_per_sweep:red_black",
                "p_unsplit",
                "red_black",
                "frozen_per_sweep",
            ),
            ("h_unsplit", "h_unsplit", "lexicographic", "pointwise"),
            ("DS1", "DS1", "lexicographic", "pointwise"),
            ("DS2", "DS2", "lexicographic", "pointwise"),
        )
        for label, method, p_ordering, coefficient_mode in cases:
            with self.subTest(case=label):
                config = make_nonlinear_config(
                    method,
                    p_ordering,
                    coefficient_mode,
                )
                solution, _, iterations, histories = nonlinear_diffusion.solve_all(
                    config
                )
                self.assert_finite_run(config, solution, iterations, histories)

    def test_linear_methods_are_finite(self):
        for method in ("parabolic_jacobi", "parabolic_gs", "parabolic_sor"):
            with self.subTest(method=method):
                config = make_linear_config(method)
                solution, _, iterations, histories = linear_diffusion.solve(config)
                self.assert_finite_run(config, solution, iterations, histories)


if __name__ == "__main__":
    unittest.main()
