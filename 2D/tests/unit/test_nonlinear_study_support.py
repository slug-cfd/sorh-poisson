"""Unit tests for shared nonlinear-study configuration and policies."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from configuration import compute_omega_opt
from problems.nonlinear import ellipticity_ratio
from problems.nonlinear.coefficients import regularized_coefficient
from tests.nonlinear_studies.common import (
    _face_stats,
    formula_reference,
    is_stagnated,
    make_config,
    run_case,
    save_study,
    select_first_converged_branch,
)


class NonlinearStudyVariantTests(unittest.TestCase):
    def test_epsilon_zero_is_the_original_coefficient(self):
        values = np.linspace(0.0, 1.0, 7)
        np.testing.assert_allclose(
            regularized_coefficient(values, 0.0),
            values / (1.0 + values),
        )

    def test_regularization_keeps_fixed_velocity_scale(self):
        for epsilon in (1e-3, 1e-2, 1e-1, 1.0):
            coefficient = lambda values: regularized_coefficient(values, epsilon)
            self.assertAlmostEqual(float(coefficient(1.0)), 0.5)
            self.assertAlmostEqual(
                float(coefficient(0.0)) / float(coefficient(1.0)),
                ellipticity_ratio(epsilon),
            )

    def test_study_formula_matches_production_config(self):
        values = np.linspace(0.0, 1.0, 7)
        for epsilon in (0.0, 1e-2, 1.0):
            config = make_config("h_unsplit", 5, 0.8, epsilon=epsilon)
            np.testing.assert_allclose(
                regularized_coefficient(values, epsilon),
                config.problem_instance.coefficient(values, config),
            )

    def test_constant_control_and_initial_guesses(self):
        constant = make_config(
            "h_unsplit",
            5,
            0.8,
            coefficient_variant="constant",
        )
        np.testing.assert_allclose(
            constant.problem_instance.coefficient(constant.U0),
            0.5,
        )

        linear = make_config("h_unsplit", 5, 0.8, initial_guess="linear_x")
        np.testing.assert_allclose(
            linear.U0,
            np.broadcast_to(linear.x[:, None], linear.U0.shape),
        )
        zero = make_config("h_unsplit", 5, 0.8, initial_guess="zero")
        np.testing.assert_allclose(zero.U0, 0.0)


class NonlinearStudyPolicyTests(unittest.TestCase):
    def test_formula_reference_uses_config_implementation(self):
        config = make_config("p_unsplit", 32, 1.0, cfl=0.8)
        self.assertAlmostEqual(
            formula_reference("p_unsplit", 32, 0.8),
            compute_omega_opt(config.settings, config.grid),
        )

    def test_selection_stops_at_first_failed_branch(self):
        rows = [
            {"omega": 0.7, "converged": False, "stable": True, "sweeps": 99,
             "elapsed_sec": 1.0, "omega_delta": -0.2},
            {"omega": 0.8, "converged": True, "stable": True, "sweeps": 20,
             "elapsed_sec": 1.0, "omega_delta": -0.1},
            {"omega": 0.9, "converged": True, "stable": True, "sweeps": 15,
             "elapsed_sec": 1.0, "omega_delta": 0.0},
            {"omega": 1.0, "converged": False, "stable": False, "sweeps": 10,
             "elapsed_sec": 1.0, "omega_delta": 0.1},
            {"omega": 1.1, "converged": True, "stable": True, "sweeps": 5,
             "elapsed_sec": 1.0, "omega_delta": 0.2},
        ]
        self.assertEqual(select_first_converged_branch(rows)["omega"], 0.9)

    def test_stagnation_policy(self):
        self.assertTrue(is_stagnated(np.ones(2_000)))
        self.assertFalse(is_stagnated(np.linspace(1.0, 0.5, 2_000)))

    def test_invalid_final_face_diagnostic_does_not_abort_case(self):
        config = make_config(
            "p_unsplit",
            4,
            2.0,
            timestep_mode="adaptive",
        )
        invalid = np.full_like(config.U0, -1.0)
        with self.assertRaises(FloatingPointError):
            _face_stats(invalid, config)

        histories = {
            "update": np.array([np.inf]),
            "residual": np.array([np.inf]),
            "dt": np.array([0.0]),
            "converged": False,
            "stable": False,
            "reason": "numerical_guard",
        }
        with patch(
            "tests.nonlinear_studies.common.nonlinear_diffusion.solve_all",
            return_value=(invalid, 0.0, 1, histories),
        ):
            row, _ = run_case(config, {}, max_sweeps=1)

        self.assertFalse(row["stable"])
        self.assertEqual(row["reason"], "numerical_guard")
        self.assertTrue(np.isnan(row["final_face_v_min"]))
        self.assertTrue(np.isnan(row["final_face_v_max"]))

    def test_study_can_omit_redundant_all_runs_file(self):
        rows = [{"case": "fixed", "sweeps": 12}]
        with TemporaryDirectory() as directory:
            output = save_study(
                "unused",
                rows,
                rows,
                {"description": "duplicate-output check"},
                output_dir=Path(directory),
                summary_name="comparison.csv",
                write_all_runs=False,
            )

            self.assertFalse((output / "all_runs.csv").exists())
            self.assertTrue((output / "comparison.csv").exists())

    def test_study_does_not_replace_existing_results(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            existing = output / "comparison.csv"
            existing.write_text("original\n")

            with self.assertRaises(FileExistsError):
                save_study(
                    "unused", [{"case": "new"}], [{"case": "new"}],
                    {}, output_dir=output, summary_name="comparison.csv",
                )

            self.assertEqual(existing.read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
