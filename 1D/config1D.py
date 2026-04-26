from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


problem_name_map = {
    "linear": "Linear",
    "nonlinear": "Nonlinear",
}

method_name_map = {
    "jacobi_linear_parabolic": "Jacobi/Linear/Parabolic",
    "jacobi_nonlinear_parabolic": "Jacobi/Nonlinear/Parabolic",
    "gs_linear_parabolic": "Gauss-Seidel/Linear/Parabolic",
    "gs_nonlinear_parabolic": "Gauss-Seidel/Nonlinear/Parabolic",
    "sor_linear_parabolic": "SOR/Linear/Parabolic",
    "sor_nonlinear_parabolic": "SOR/Nonlinear/Parabolic",
    "sor_linear_hyperbolic": "SOR/Linear/Hyperbolic",
    "sor_nonlinear_hyperbolic": "SOR/Nonlinear/Hyperbolic",
}


@dataclass
class Solver1DConfig:
    tol_update: float = 1e-7
    tol_exact: float | None = None
    res_max: float = 1e8
    stop_on_exact: bool = False


@dataclass
class Config1D:
    # user-set
    problem: str
    method: str

    N: int
    cfl: float
    omega: float
    iter_max: int
    norm: Any

    solver_config: Solver1DConfig = field(default_factory=Solver1DConfig)
    dt_factor: float = 1.0

    xa: float = 0.0
    xb: float = 1.0
    left_bc: float = 0.0
    right_bc: float = 1.0

    linear_v: float = 0.5
    nonlinear_vmax: float = 0.5
    hyperbolic_linear_v: float = 1.0

    # derived
    dx: float = field(init=False)
    ngc: int = field(init=False)
    ib: int = field(init=False)
    ie: int = field(init=False)
    x: np.ndarray = field(init=False)
    U0: np.ndarray = field(init=False)
    U_exact: np.ndarray | None = field(init=False)

    def __post_init__(self):
        if self.problem not in problem_name_map:
            raise ValueError(
                f"Unknown 1D problem '{self.problem}'. "
                f"Valid options: {list(problem_name_map)}"
            )
        if self.method not in method_name_map:
            raise ValueError(
                f"Unknown 1D method '{self.method}'. "
                f"Valid options: {list(method_name_map)}"
            )
        if self.N < 2:
            raise ValueError("N must be at least 2 so the grid has interior points.")

        self.ngc = 0
        self.ib = 1
        self.ie = self.N - 1
        self.x = np.linspace(self.xa, self.xb, self.N + 1)
        self.dx = (self.xb - self.xa) / self.N

        self.U0 = np.zeros(self.N + 1)
        self.apply_bcs(self.U0)

        self.U_exact = self.x.copy() if self.problem == "linear" else None

    @property
    def interior_size(self) -> int:
        return self.ie - self.ib + 1

    def apply_bcs(self, U: np.ndarray) -> np.ndarray:
        U[0] = self.left_bc
        U[-1] = self.right_bc
        return U

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "method": self.method,
            "N": self.N,
            "cfl": self.cfl,
            "omega": self.omega,
            "iter_max": self.iter_max,
            "norm": "inf" if self.norm == np.inf else self.norm,
            "solver_config": asdict(self.solver_config),
            "dt_factor": self.dt_factor,
            "xa": self.xa,
            "xb": self.xb,
            "left_bc": self.left_bc,
            "right_bc": self.right_bc,
            "linear_v": self.linear_v,
            "nonlinear_vmax": self.nonlinear_vmax,
            "hyperbolic_linear_v": self.hyperbolic_linear_v,
        }
