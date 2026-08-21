import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from configuration import (
    Config,
    GridConfig,
    NonlinearCoefficientConfig,
    ProblemConfig,
    RunConfig,
    SimulationContext,
    StoppingConfig,
    initialize_simulation,
    load_config,
)
from problems.poisson_tests import polar_star


class ConfigurationInitializationTests(unittest.TestCase):
    def test_loading_and_initialization_are_distinct_stages(self):
        settings = load_config(BASE_DIR / "configs" / "config_nonlinear.json")

        self.assertIsInstance(settings, Config)
        self.assertFalse(hasattr(settings, "U0"))

        context = initialize_simulation(settings)
        self.assertIsInstance(context, SimulationContext)
        self.assertEqual(context.U0.shape, (34, 34))
        self.assertEqual(context.problem_instance.name, "unit_square_linear_x")

    def test_problem_variants_use_the_standard_initialization_pipeline(self):
        settings = Config(
            problem=ProblemConfig(
                kind="nonlinear_diffusion",
                case="unit_square_linear_x",
                coefficient=NonlinearCoefficientConfig(
                    model="constant",
                    constant_value=0.25,
                ),
                initial_condition="linear_x",
            ),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method="p_unsplit",
                cfl=0.5,
                omega=1.0,
                iter_max=1,
                norm="fro",
                stopping=StoppingConfig(),
            ),
        )

        context = initialize_simulation(settings)
        np.testing.assert_allclose(
            context.U0,
            np.broadcast_to(context.x[:, None], context.U0.shape),
        )
        np.testing.assert_allclose(
            context.problem_instance.coefficient(context.U0),
            0.25,
        )

    def test_poisson_parameters_override_only_the_initialized_problem(self):
        default_epsilon = polar_star.PROBLEM.constants["epsilon"]
        settings = Config(
            problem=ProblemConfig(
                kind="poisson",
                case="polar_star",
                parameters={"epsilon": 0.05},
            ),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method="h_unsplit",
                cfl=0.8,
                omega=0.8,
                iter_max=1,
                norm="fro",
            ),
        )

        context = initialize_simulation(settings)

        self.assertEqual(context.epsilon, 0.05)
        self.assertEqual(polar_star.PROBLEM.constants["epsilon"], default_epsilon)

    def test_unknown_poisson_parameter_is_rejected(self):
        settings = Config(
            problem=ProblemConfig(
                kind="poisson",
                case="polar_star",
                parameters={"unknown": 1.0},
            ),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method="h_unsplit",
                cfl=0.8,
                omega=0.8,
                iter_max=1,
                norm="fro",
            ),
        )

        with self.assertRaisesRegex(ValueError, "Unknown parameters"):
            initialize_simulation(settings)

    def test_json_loader_accepts_poisson_parameters(self):
        raw = {
            "problem": {
                "kind": "poisson",
                "case": "polar_star",
                "parameters": {"epsilon": 0.02},
            },
            "grid": {"Nx": 5, "Ny": 4},
            "run": {
                "method": "h_unsplit",
                "cfl": 0.8,
                "omega": 0.8,
                "iter_max": 1,
                "norm": "fro",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            context = initialize_simulation(load_config(path))

        self.assertEqual(context.epsilon, 0.02)

    def test_flat_legacy_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"problem": "poisson"}))
            with self.assertRaisesRegex(ValueError, "nested"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
