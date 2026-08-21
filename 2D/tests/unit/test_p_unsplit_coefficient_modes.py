import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import boundary_conditions as bc
from configuration import (
    ParabolicUnsplitConfig,
    initialize_simulation,
    load_config,
)
from methods.nonlinear_diffusion import p_unsplit
from tests.nonlinear_studies.common import make_config


def nonuniform_state(config):
    x = config.x[:, None]
    y = config.y[None, :]
    state = np.clip(
        x + 0.08 * np.sin(np.pi * x) * np.sin(np.pi * y),
        0.0,
        1.0,
    )
    bc.update_BCs(state, config)
    return state


def sweep_with_mode(state, config):
    return p_unsplit.full_sweep(
        state.copy(),
        p_unsplit.get_params(config),
        config,
    )


class SorPCoefficientModeConfigTests(unittest.TestCase):
    def test_default_mode_is_pointwise(self):
        self.assertEqual(
            ParabolicUnsplitConfig().coefficient_mode,
            "pointwise",
        )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coefficient_mode"):
            ParabolicUnsplitConfig(coefficient_mode="frozen")

    def test_json_loader_and_serialization_expose_mode(self):
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
                "p_unsplit": {
                    "coefficient_mode": "frozen_per_sweep",
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            config = initialize_simulation(load_config(path))

        self.assertEqual(
            config.p_unsplit_config.coefficient_mode,
            "frozen_per_sweep",
        )
        self.assertEqual(
            config.to_dict()["methods"]["p_unsplit"]["coefficient_mode"],
            "frozen_per_sweep",
        )


class SorPCoefficientKernelTests(unittest.TestCase):
    def test_modes_match_for_constant_coefficients(self):
        pointwise_config = make_config(
            "p_unsplit",
            6,
            1.3,
            coefficient_variant="constant",
            coefficient_mode="pointwise",
        )
        frozen_config = make_config(
            "p_unsplit",
            6,
            1.3,
            coefficient_variant="constant",
            coefficient_mode="frozen_per_sweep",
        )
        state = nonuniform_state(pointwise_config)

        pointwise = sweep_with_mode(state, pointwise_config)
        frozen = sweep_with_mode(state, frozen_config)
        np.testing.assert_allclose(frozen, pointwise, rtol=0.0, atol=1e-14)

    def test_modes_are_distinct_for_nonlinear_coefficients(self):
        pointwise_config = make_config(
            "p_unsplit", 6, 1.3, coefficient_mode="pointwise"
        )
        frozen_config = make_config(
            "p_unsplit", 6, 1.3, coefficient_mode="frozen_per_sweep"
        )
        state = nonuniform_state(pointwise_config)

        pointwise = sweep_with_mode(state, pointwise_config)
        frozen = sweep_with_mode(state, frozen_config)
        self.assertGreater(float(np.max(np.abs(frozen - pointwise))), 1e-12)

    def test_modes_use_the_same_sweep_start_adaptive_timestep(self):
        base = make_config("p_unsplit", 6, 1.3, timestep_mode="adaptive")
        state = nonuniform_state(base)
        timesteps = {}

        for mode in ("pointwise", "frozen_per_sweep"):
            with self.subTest(mode=mode):
                config = make_config(
                    "p_unsplit",
                    6,
                    1.3,
                    timestep_mode="adaptive",
                    coefficient_mode=mode,
                )
                params = p_unsplit.get_params(config)
                result = p_unsplit.full_sweep(state.copy(), params, config)
                self.assertTrue(np.all(np.isfinite(result)))
                timesteps[mode] = params["dt"]

        self.assertEqual(
            timesteps["pointwise"],
            timesteps["frozen_per_sweep"],
        )


if __name__ == "__main__":
    unittest.main()
