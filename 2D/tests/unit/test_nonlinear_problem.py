from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import boundary_conditions as bc
from configuration import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    NonlinearCoefficientConfig,
    ProblemConfig,
    RunConfig,
    StoppingConfig,
    initialize_simulation,
    load_config,
)
from problems import nonlinear_diffusion
from methods.nonlinear_diffusion.nl_helpers import nonlinear_flux_residual
from problems.nonlinear.coefficients import regularized_coefficient
from solver_utils import normalized_grid_norm, stopping_reason


def make_config(
    method="h_unsplit",
    *,
    iter_max=1000,
    timestep_mode="fixed",
    regularization_epsilon=0.0,
):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(
                kind="nonlinear_diffusion",
                case="unit_square_linear_x",
                coefficient=NonlinearCoefficientConfig(
                    regularization_epsilon=regularization_epsilon,
                ),
            ),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method=method,
                cfl=0.8,
                omega=0.82,
                iter_max=iter_max,
                norm="fro",
                timestep_mode=timestep_mode,
                stopping=StoppingConfig(
                    tol_update=1e-10,
                    tol_residual=1e-4,
                ),
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(),
            ),
        )
    )


class NonlinearProblemTests(unittest.TestCase):
    def test_boundary_is_u_equals_x_on_all_four_edges(self):
        config = make_config(iter_max=1)
        U = config.U0.copy()
        bc.update_BCs(U, config)

        np.testing.assert_allclose(U[0, :], config.x[0])
        np.testing.assert_allclose(U[-1, :], config.x[-1])
        np.testing.assert_allclose(U[:, 0], config.x)
        np.testing.assert_allclose(U[:, -1], config.x)

    def test_constant_state_has_zero_flux_residual(self):
        config = make_config(iter_max=1)
        U = np.full_like(config.U0, 0.5)
        np.testing.assert_allclose(nonlinear_flux_residual(U, config), 0.0)

    def test_stopping_uses_residual_with_update_stagnation_safeguard(self):
        settings = StoppingConfig(tol_update=1e-10, tol_residual=1e-5)
        self.assertEqual(
            stopping_reason(2e-4, 1e-6, settings),
            "tol_residual",
        )
        self.assertEqual(
            stopping_reason(1e-12, 2e-5, settings),
            "stagnated_update",
        )
        self.assertIsNone(stopping_reason(1e-6, 2e-5, settings))

    def test_two_norm_is_applied_to_flattened_grid(self):
        values = np.array([[3.0, 0.0], [0.0, 4.0]])
        self.assertAlmostEqual(normalized_grid_norm(values, 2), 2.5)

    def test_coefficient_rejects_singularity(self):
        with self.assertRaisesRegex(FloatingPointError, "u=-1"):
            regularized_coefficient(np.array([0.0, -1.0]))

    def test_configured_regularization_matches_experiment_formula(self):
        values = np.linspace(0.0, 1.0, 7)
        for epsilon in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
            with self.subTest(epsilon=epsilon):
                config = make_config(
                    iter_max=1,
                    regularization_epsilon=epsilon,
                )
                actual = config.problem_instance.coefficient(values, config)
                expected = regularized_coefficient(values, epsilon)
                np.testing.assert_allclose(actual, expected)
                self.assertAlmostEqual(float(actual[-1]), 0.5)

    def test_regularization_epsilon_is_validated(self):
        for epsilon in (-1e-3, np.inf, np.nan):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "regularization_epsilon"):
                    NonlinearCoefficientConfig(regularization_epsilon=epsilon)

    def test_sample_config_runs_to_flux_residual_tolerance(self):
        config = make_config()
        _, t, iterations, histories = nonlinear_diffusion.solve_all(config)

        self.assertTrue(histories["converged"])
        self.assertEqual(histories["reason"], "tol_residual")
        self.assertEqual(iterations, len(histories["residual"]))
        self.assertLess(
            histories["residual"][-1],
            config.stopping_config.tol_residual,
        )
        self.assertGreater(t, 0.0)

    def test_nonlinear_json_loads_modular_problem_and_tolerances(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "config_nonlinear.json"
        )
        config = initialize_simulation(load_config(path))
        self.assertEqual(config.problem_instance.name, "unit_square_linear_x")
        self.assertEqual(config.stopping_config.tol_residual, 1e-5)
        self.assertEqual(config.timestep_mode, "fixed")
        self.assertEqual(
            config.settings.problem.coefficient.regularization_epsilon,
            0.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            saved_path = Path(directory) / "saved_config.json"
            saved = config.to_dict()
            saved["problem"]["coefficient"]["regularization_epsilon"] = 0.1
            saved_path.write_text(json.dumps(saved))
            reloaded = initialize_simulation(load_config(saved_path))
        self.assertEqual(reloaded.problem_instance.name, "unit_square_linear_x")
        self.assertEqual(
            reloaded.settings.problem.coefficient.regularization_epsilon,
            0.1,
        )

    def test_regularized_problem_runs_to_flux_residual_tolerance(self):
        config = make_config(regularization_epsilon=0.1)
        _, _, _, histories = nonlinear_diffusion.solve_all(config)

        self.assertTrue(histories["converged"])
        self.assertEqual(histories["reason"], "tol_residual")
        self.assertLess(
            histories["residual"][-1],
            config.stopping_config.tol_residual,
        )

    def test_fixed_and_adaptive_timesteps_are_available_for_both_methods(self):
        for method in ("p_unsplit", "h_unsplit"):
            with self.subTest(method=method):
                fixed = make_config(method, iter_max=1, timestep_mode="fixed")
                _, _, _, fixed_histories = nonlinear_diffusion.solve_all(fixed)

                adaptive = make_config(method, iter_max=1, timestep_mode="adaptive")
                _, _, _, adaptive_histories = nonlinear_diffusion.solve_all(adaptive)

                self.assertGreater(adaptive_histories["dt"][0], fixed_histories["dt"][0])

    def test_invalid_timestep_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestep_mode"):
            RunConfig(
                method="h_unsplit",
                cfl=0.8,
                omega=1.0,
                iter_max=1,
                norm="fro",
                timestep_mode="automatic",
            )


if __name__ == "__main__":
    unittest.main()
