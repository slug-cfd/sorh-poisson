import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import boundary_conditions as bc
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
from methods.nonlinear_diffusion import h_unsplit as nonlinear_h_unsplit
from methods.nonlinear_diffusion import hu_methods as nonlinear_hu_methods
from methods.poisson import h_unsplit, hu_methods, p_unsplit
from problems import poisson


LEXICOGRAPHIC_ORDER = [(1, 1), (1, 2), (2, 1), (2, 2)]


def make_poisson_config(
    method: str,
    omega: float,
    *,
    nx: int = 5,
    ny: int = 5,
    h_mode: str = "red_black",
    p_ordering: str = "red_black",
):
    return initialize_simulation(
        Config(
            problem=ProblemConfig(kind="poisson", case="two_blob"),
            grid=GridConfig(Nx=nx, Ny=ny),
            run=RunConfig(
                method=method,
                cfl=1.0,
                omega=omega,
                iter_max=1,
                norm="fro",
            ),
            methods=MethodConfig(
                h_unsplit=HyperbolicUnsplitConfig(mode=h_mode),
                p_unsplit=ParabolicUnsplitConfig(gs_ordering=p_ordering),
            ),
        )
    )


def matching_square_sweeps(ordering):
    omega_h = 0.6
    h_config = make_poisson_config(
        "h_unsplit",
        omega_h,
        h_mode="red_black" if ordering == "red_black" else "default",
    )
    p_config = make_poisson_config(
        "p_unsplit",
        2.0 * omega_h,
        p_ordering=ordering,
    )

    h_boundaries = bc.get_bcs_poisson_2d(h_config, h_config.phi_exact)
    p_boundaries = bc.get_bcs_poisson_2d(p_config, p_config.phi_exact)
    U_h = h_config.U0.copy()
    U_p = p_config.U0.copy()
    bc.apply_bcs_poisson_2d(U_h, h_config, h_boundaries)
    bc.apply_bcs_poisson_2d(U_p, p_config, p_boundaries)

    U_h = h_unsplit.full_sweep(
        U_h,
        h_unsplit.get_params(h_config),
        h_boundaries,
        h_config,
    )
    U_p = p_unsplit.full_sweep(
        U_p,
        p_unsplit.get_params(p_config),
        p_boundaries,
        p_config,
    )
    return U_h, U_p


def make_nonlinear_config():
    return initialize_simulation(
        Config(
            problem=ProblemConfig(
                kind="nonlinear_diffusion",
                case="unit_square_linear_x",
            ),
            grid=GridConfig(Nx=5, Ny=4),
            run=RunConfig(
                method="h_unsplit",
                cfl=0.5,
                omega=0.5,
                iter_max=1,
                norm="fro",
            ),
        )
    )


class SorHOrderingConfigTests(unittest.TestCase):
    def test_default_ordering_is_lexicographic(self):
        self.assertEqual(HyperbolicUnsplitConfig().mode, "default")

    def test_invalid_ordering_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "h_unsplit mode"):
            HyperbolicUnsplitConfig(mode="diagonal")

    def test_json_loader_and_serialization_expose_ordering(self):
        raw = {
            "problem": {"kind": "poisson", "case": "two_blob"},
            "grid": {"Nx": 5, "Ny": 4},
            "run": {
                "method": "h_unsplit",
                "cfl": 0.5,
                "omega": 0.5,
                "iter_max": 1,
                "norm": "fro",
            },
            "methods": {"h_unsplit": {"mode": "red_black"}},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            config = initialize_simulation(load_config(path))

        self.assertEqual(config.h_unsplit_config.mode, "red_black")
        self.assertEqual(
            config.to_dict()["methods"]["h_unsplit"]["mode"],
            "red_black",
        )


class SorHDefaultOrderingTests(unittest.TestCase):
    def test_poisson_sweep_uses_lexicographic_order(self):
        config = make_poisson_config(
            "h_unsplit",
            omega=0.5,
            h_mode="default",
        )
        config.ib = config.jb = 1
        config.ie = config.je = 2
        visited = []

        def record_laplacian(_, i, j, *args):
            visited.append((i, j))
            return 0.0

        with (
            mock.patch.object(
                hu_methods,
                "laplacian_2D",
                side_effect=record_laplacian,
            ),
            mock.patch.object(hu_methods.bc, "apply_bcs_poisson_2d"),
        ):
            hu_methods.unsplit_update(
                np.zeros((4, 4)),
                config,
                boundaries=[],
                update_coeffs=(1.0 / (config.dx + config.dy), 0.1),
                f_all=np.zeros((4, 4)),
            )

        self.assertEqual(visited, LEXICOGRAPHIC_ORDER)

    def test_nonlinear_sweep_uses_lexicographic_order(self):
        config = make_nonlinear_config()
        config.ib = config.jb = 1
        config.ie = config.je = 2
        visited = []

        class TrackedVelocity:
            def __init__(self):
                self.values = np.zeros((2, 2, 2))

            def __getitem__(self, index):
                local_i, local_j, side = index
                if side == 0:
                    visited.append((local_i + config.ib, local_j + config.jb))
                return self.values[index]

        with (
            mock.patch.object(
                nonlinear_hu_methods.nlh,
                "get_vx",
                return_value=TrackedVelocity(),
            ),
            mock.patch.object(
                nonlinear_hu_methods.nlh,
                "get_vy",
                return_value=np.zeros((2, 2, 2)),
            ),
            mock.patch.object(nonlinear_hu_methods.bc, "update_BCs"),
        ):
            nonlinear_h_unsplit.full_sweep(
                np.zeros((4, 4)),
                {"fixed_dt": 0.1},
                config,
            )

        self.assertEqual(visited, LEXICOGRAPHIC_ORDER)


class SorHRedBlackFormulaTests(unittest.TestCase):
    def test_uses_frozen_residual_and_distinct_x_y_weights(self):
        params = SimpleNamespace(
            dx=2.0,
            dy=3.0,
            omega=0.4,
            ib=1,
            ie=2,
            jb=1,
            je=2,
            laplacian_stencil="centered",
        )
        c1 = 1.0 / (params.dx + params.dy)
        c2 = 7.0
        U = np.arange(16, dtype=float).reshape(4, 4)
        f_all = np.zeros_like(U)
        residual = np.zeros_like(U)
        residual[1, 1] = 1.0
        residual[2, 2] = 2.0
        residual[1, 2] = 3.0
        residual[2, 1] = 4.0

        with (
            mock.patch.object(
                hu_methods,
                "get_all_laplace",
                return_value=f_all + residual,
            ) as get_all_laplace,
            mock.patch.object(hu_methods.bc, "apply_bcs_poisson_2d"),
        ):
            result = hu_methods.red_black_rectangular(
                U,
                params,
                boundaries=[],
                update_coeffs=(c1, c2),
                f_all=f_all,
            )

        residual_coeff = params.omega * c1 * c2
        x_coeff = params.omega * c1 * params.dy
        y_coeff = params.omega * c1 * params.dx
        expected_delta = np.zeros_like(U)
        expected_delta[1, 1] = residual_coeff * residual[1, 1]
        expected_delta[2, 2] = residual_coeff * residual[2, 2]
        expected_delta[1, 2] = (
            x_coeff * expected_delta[2, 2]
            + y_coeff * expected_delta[1, 1]
            + residual_coeff * residual[1, 2]
        )
        expected_delta[2, 1] = (
            x_coeff * expected_delta[1, 1]
            + y_coeff * expected_delta[2, 2]
            + residual_coeff * residual[2, 1]
        )

        np.testing.assert_allclose(result, U + expected_delta)
        get_all_laplace.assert_called_once()
        np.testing.assert_array_equal(get_all_laplace.call_args.args[0], U)
        self.assertIs(get_all_laplace.call_args.args[1], params)
        self.assertEqual(get_all_laplace.call_args.args[2], "centered")

    def test_rejects_non_bipartite_laplacian_stencils(self):
        params = SimpleNamespace(laplacian_stencil="isotropic_9pt")

        with self.assertRaisesRegex(ValueError, "requires.*centered"):
            hu_methods.red_black_rectangular(
                np.zeros((4, 4)),
                params,
                boundaries=[],
                update_coeffs=(1.0, 1.0),
                f_all=np.zeros((4, 4)),
            )


class SorHOrderingIntegrationTests(unittest.TestCase):
    def test_square_cfl_one_matches_classical_red_black_sor(self):
        U_h, U_p = matching_square_sweeps("red_black")
        np.testing.assert_allclose(U_h, U_p, rtol=1e-13, atol=1e-13)

    def test_square_cfl_one_matches_classical_lexicographic_sor(self):
        U_h, U_p = matching_square_sweeps("lexicographic")
        np.testing.assert_allclose(U_h, U_p, rtol=1e-13, atol=1e-13)

    def test_orderings_produce_distinct_first_sweeps(self):
        solutions = {}
        for mode in ("default", "red_black"):
            with self.subTest(mode=mode):
                config = make_poisson_config(
                    "h_unsplit",
                    omega=0.5,
                    h_mode=mode,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    solution, _, _, _ = poisson.solve_all(config)
                solutions[mode] = solution

        difference = np.max(
            np.abs(solutions["default"] - solutions["red_black"])
        )
        self.assertGreater(float(difference), 0.0)

if __name__ == "__main__":
    unittest.main()
