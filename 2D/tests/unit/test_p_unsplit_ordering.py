import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    ParabolicUnsplitConfig,
    ProblemConfig,
    RunConfig,
    initialize_simulation,
    load_config,
)
from methods.poisson import p_unsplit as poisson_p_unsplit
from methods.nonlinear_diffusion import p_unsplit as nonlinear_p_unsplit
from methods.nonlinear_diffusion import pu_methods as nonlinear_pu_methods
from problems import nonlinear_diffusion, poisson


LEXICOGRAPHIC_ORDER = [(1, 1), (1, 2), (2, 1), (2, 2)]
RED_BLACK_ORDER = [(1, 1), (2, 2), (1, 2), (2, 1)]


def make_config(problem: str, gs_ordering: str):
    case = "two_blob" if problem == "poisson" else "unit_square_linear_x"
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind=problem, case=case),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method="p_unsplit",
                cfl=0.5,
                omega=0.5,
                iter_max=1,
                norm="fro",
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(),
                p_unsplit=ParabolicUnsplitConfig(gs_ordering=gs_ordering),
            ),
        )
    )


class SorPOrderingConfigTests(unittest.TestCase):
    def test_default_ordering_is_lexicographic(self):
        self.assertEqual(
            ParabolicUnsplitConfig().gs_ordering,
            "lexicographic",
        )

    def test_invalid_ordering_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gs_ordering"):
            ParabolicUnsplitConfig(gs_ordering="diagonal")

    def test_json_loader_and_serialization_expose_ordering(self):
        raw = {
            "problem": {
                "kind": "nonlinear_diffusion",
                "case": "unit_square_linear_x",
            },
            "grid": {"Nx": 5, "Ny": 4},
            "run": {
                "method": "p_unsplit",
                "cfl": 0.5,
                "omega": 0.5,
                "iter_max": 1,
                "norm": "fro",
            },
            "methods": {
                "p_unsplit": {"gs_ordering": "lexicographic"},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            config = initialize_simulation(load_config(path))

        self.assertEqual(
            config.p_unsplit_config.gs_ordering,
            "lexicographic",
        )
        self.assertEqual(
            config.to_dict()["methods"]["p_unsplit"]["gs_ordering"],
            "lexicographic",
        )


class SorPOrderingSweepTests(unittest.TestCase):
    def test_poisson_sweep_uses_selected_point_order(self):
        config = make_config("poisson", "red_black")
        config.ib = config.jb = 1
        config.ie = config.je = 2
        visited = []

        def record_point(*args):
            visited.append((args[2], args[3]))

        with (
            mock.patch.object(
                poisson_p_unsplit,
                "_update_point",
                side_effect=record_point,
            ),
            mock.patch.object(
                poisson_p_unsplit.bc,
                "apply_bcs_poisson_2d",
            ),
        ):
            poisson_p_unsplit.full_sweep(
                np.zeros((4, 4)),
                {"dt": 0.1},
                [],
                config,
            )

        self.assertEqual(visited, RED_BLACK_ORDER)

        config = make_config("poisson", "lexicographic")
        config.ib = config.jb = 1
        config.ie = config.je = 2
        visited.clear()
        with (
            mock.patch.object(
                poisson_p_unsplit,
                "_update_point",
                side_effect=record_point,
            ),
            mock.patch.object(
                poisson_p_unsplit.bc,
                "apply_bcs_poisson_2d",
            ),
        ):
            poisson_p_unsplit.full_sweep(
                np.zeros((4, 4)),
                {"dt": 0.1},
                [],
                config,
            )

        self.assertEqual(visited, LEXICOGRAPHIC_ORDER)

    def test_nonlinear_sweep_uses_selected_point_order(self):
        config = make_config("nonlinear_diffusion", "red_black")
        config.ib = config.jb = 1
        config.ie = config.je = 2
        visited = []

        def record_point(*args):
            visited.append((args[2], args[3]))

        with (
            mock.patch.object(
                nonlinear_pu_methods,
                "_pointwise_update",
                side_effect=record_point,
            ),
            mock.patch.object(nonlinear_pu_methods.bc, "update_BCs"),
        ):
            nonlinear_p_unsplit.full_sweep(
                np.zeros((4, 4)),
                {"rx": 0.1, "ry": 0.1},
                config,
            )

        self.assertEqual(visited, RED_BLACK_ORDER)

        config = make_config("nonlinear_diffusion", "lexicographic")
        config.ib = config.jb = 1
        config.ie = config.je = 2
        visited.clear()
        with (
            mock.patch.object(
                nonlinear_pu_methods,
                "_pointwise_update",
                side_effect=record_point,
            ),
            mock.patch.object(nonlinear_pu_methods.bc, "update_BCs"),
        ):
            nonlinear_p_unsplit.full_sweep(
                np.zeros((4, 4)),
                {"rx": 0.1, "ry": 0.1},
                config,
            )

        self.assertEqual(visited, LEXICOGRAPHIC_ORDER)

    def test_orderings_produce_distinct_first_sweeps(self):
        solutions = {}

        for problem, solver in (
            ("poisson", poisson.solve_all),
            ("nonlinear_diffusion", nonlinear_diffusion.solve_all),
        ):
            for gs_ordering in ("lexicographic", "red_black"):
                with self.subTest(
                    problem=problem,
                    gs_ordering=gs_ordering,
                ):
                    config = make_config(problem, gs_ordering)
                    with contextlib.redirect_stdout(io.StringIO()):
                        solution, _, _, _ = solver(config)
                    solutions[(problem, gs_ordering)] = solution

            difference = np.max(
                np.abs(
                    solutions[(problem, "lexicographic")]
                    - solutions[(problem, "red_black")]
                )
            )
            self.assertGreater(float(difference), 0.0)

if __name__ == "__main__":
    unittest.main()
