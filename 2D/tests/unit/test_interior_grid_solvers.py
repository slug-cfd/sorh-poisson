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
    MethodConfig,
    MultiGridConfig,
    ProblemConfig,
    RunConfig,
    compute_omega_opt,
    initialize_simulation,
)
from methods.poisson import multigrid
from problems import poisson


def make_poisson_config(
    method: str,
    *,
    nx: int = 5,
    ny: int = 4,
    multigrid_config=None,
    use_omega_opt=False,
):
    settings = Config(
        problem=ProblemConfig(kind="poisson", case="two_blob"),
        grid=GridConfig(Nx=nx, Ny=ny),
        run=RunConfig(
            method=method,
            cfl=0.5,
            omega=0.5,
            iter_max=1,
            norm="fro",
            multigrid=multigrid_config is not None,
            use_omega_opt=use_omega_opt,
        ),
        methods=MethodConfig(
            multigrid=multigrid_config or MultiGridConfig(),
        ),
    )
    return initialize_simulation(settings)


class InteriorGridSolverTests(unittest.TestCase):
    def test_multigrid_rebuilds_solver_parameters_for_each_interior_size(self):
        for method in ("p_unsplit", "h_unsplit", "p_ADI", "h_ADI"):
            with self.subTest(method=method):
                config = make_poisson_config(
                    method,
                    nx=7,
                    ny=5,
                    multigrid_config=MultiGridConfig(
                        solver="rep_v_cycles",
                        max_cycles=1,
                        nu1=1,
                        nu2=1,
                        min_coarse_n=2,
                        coarse_max_sweeps=1,
                    ),
                )

                with contextlib.redirect_stdout(io.StringIO()):
                    solution, _, cycles, _ = poisson.solve_all(config)

                self.assertEqual(solution.shape, (9, 7))
                self.assertEqual(cycles, 1)

    def test_multigrid_transfer_shapes_preserve_interior_convention(self):
        for ngc in (1, 2):
            with self.subTest(ngc=ngc):
                fine = np.arange(
                    (7 + 2 * ngc) * (5 + 2 * ngc),
                    dtype=float,
                ).reshape(7 + 2 * ngc, 5 + 2 * ngc)
                coarse = multigrid.restrict_injection(fine, ngc)

                self.assertEqual(coarse.shape, (4 + 2 * ngc, 3 + 2 * ngc))
                prolonged = multigrid.prolong_bilinear(coarse, fine.shape, ngc)
                self.assertEqual(prolonged.shape, fine.shape)

    def test_coarse_optimal_omega_uses_coarse_interior_counts(self):
        fine_config = make_poisson_config(
            "h_unsplit",
            nx=7,
            ny=5,
            use_omega_opt=True,
        )
        coarse_rhs = multigrid.restrict_injection(
            fine_config.f_all,
            fine_config.ngc,
        )
        coarse_config = multigrid.build_coarse_config(fine_config, coarse_rhs)

        self.assertEqual((coarse_config.Nx, coarse_config.Ny), (4, 3))
        self.assertEqual(coarse_config.U0.shape, coarse_rhs.shape)
        self.assertAlmostEqual(
            coarse_config.omega,
            compute_omega_opt(coarse_config.settings, coarse_config.grid),
        )


if __name__ == "__main__":
    unittest.main()
