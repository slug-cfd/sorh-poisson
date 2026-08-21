import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configuration import (
    Config,
    GridConfig,
    ProblemConfig,
    RunConfig,
    compute_omega_opt,
    initialize_simulation,
    load_config,
)


def make_config(method, *, nx=64, ny=64, cfl=0.8, omega=0.75, use_omega_opt=True):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="linear_diffusion", case="default"),
            grid=GridConfig(Nx=nx, Ny=ny),
            run=RunConfig(
                method=method,
                cfl=cfl,
                omega=omega,
                iter_max=10,
                norm="fro",
                use_omega_opt=use_omega_opt,
            ),
        )
    )


class OptimalOmegaConfigTests(unittest.TestCase):
    def test_nx_ny_are_interior_counts(self):
        config = make_config("h_unsplit", nx=5, ny=3, use_omega_opt=False)

        self.assertEqual(config.U0.shape, (7, 5))
        self.assertEqual((config.ib, config.ie), (1, 5))
        self.assertEqual((config.jb, config.je), (1, 3))
        self.assertEqual(config.ie - config.ib + 1, config.Nx)
        self.assertEqual(config.je - config.jb + 1, config.Ny)
        self.assertAlmostEqual(config.dx, 1.0 / 6.0)
        self.assertAlmostEqual(config.dy, 1.0 / 4.0)

    def test_extra_ghost_layers_do_not_change_interior_count(self):
        config = initialize_simulation(
            Config(
                problem=ProblemConfig(kind="linear_diffusion", case="default"),
                grid=GridConfig(Nx=5, Ny=3, laplacian_stencil="wide"),
                run=RunConfig(
                    method="h_unsplit",
                    cfl=0.8,
                    omega=0.75,
                    iter_max=10,
                    norm="fro",
                ),
            )
        )

        self.assertEqual(config.ngc, 2)
        self.assertEqual(config.U0.shape, (9, 7))
        self.assertEqual(config.ie - config.ib + 1, config.Nx)
        self.assertEqual(config.je - config.jb + 1, config.Ny)
        self.assertAlmostEqual(config.x[config.ngc - 1], config.xa)
        self.assertAlmostEqual(config.x[-config.ngc], config.xb)

    def test_hyperbolic_square_grid_matches_1d_formula(self):
        config = make_config("h_unsplit", nx=64, ny=64)
        expected = 1.0 / (1.0 + math.sin(math.pi / 65))
        self.assertAlmostEqual(config.omega, expected)

    def test_parabolic_square_grid_matches_1d_formula(self):
        config = make_config("p_unsplit", nx=64, ny=64, cfl=0.8)
        expected = 2.0 / (0.8 * (1.0 + math.sin(math.pi / 65)))
        self.assertAlmostEqual(config.omega, expected)

    def test_rectangular_grid_uses_interval_counts(self):
        config = make_config("h_unsplit", nx=5, ny=3)
        nx_intervals = 6
        ny_intervals = 4
        c_max = (
            nx_intervals * math.cos(math.pi / nx_intervals)
            + ny_intervals * math.cos(math.pi / ny_intervals)
        ) / (nx_intervals + ny_intervals)
        expected = 1.0 / (1.0 + math.sqrt(1.0 - c_max**2))
        self.assertAlmostEqual(config.omega, expected)

    def test_disabled_option_preserves_configured_omega(self):
        config = make_config("h_unsplit", omega=0.73, use_omega_opt=False)
        self.assertEqual(config.omega, 0.73)

    def test_optimal_formula_remains_available_for_diagnostics(self):
        config = make_config(
            "h_unsplit",
            nx=64,
            ny=64,
            omega=0.73,
            use_omega_opt=False,
        )
        expected = 1.0 / (1.0 + math.sin(math.pi / 65))
        self.assertAlmostEqual(
            compute_omega_opt(config.settings, config.grid),
            expected,
        )
        self.assertEqual(config.omega, 0.73)

    def test_json_loader_accepts_option_without_fixed_omega(self):
        raw = {
            "problem": {"kind": "linear_diffusion", "case": "default"},
            "grid": {"Nx": 32, "Ny": 32},
            "run": {
                "method": "h_unsplit",
                "cfl": 0.8,
                "omega": 1.0,
                "use_omega_opt": True,
                "iter_max": 10,
                "norm": "fro",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            config = initialize_simulation(load_config(path))

        expected = 1.0 / (1.0 + math.sin(math.pi / 33))
        self.assertAlmostEqual(config.omega, expected)


if __name__ == "__main__":
    unittest.main()
