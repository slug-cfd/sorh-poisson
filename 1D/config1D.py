from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


# Accepted problem names mapped to their display labels.
problem_name_map = {
    "linear": "Linear",
    "nonlinear": "Nonlinear",
}

# History plotted beside the final solution: exact diagnostic or update residual.
plot_history_norm_options = {"exact", "update"}

# Accepted solver names mapped to their display labels.
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
    # Stop when the norm of one iteration's update falls below this value.
    tol_update: float = 1e-7
    # Optional threshold for the problem-specific exact diagnostic.
    tol_exact: float | None = None
    # Treat a nonfinite or larger update residual as numerical instability.
    res_max: float = 1e8
    # Allow tol_exact, when set, to terminate the solve.
    stop_on_exact: bool = False


@dataclass
class Config1D:
    # user-set
    problem: str    # Options: "linear", "nonlinear".
    method: str     # Options: see `method_name_map` (keys) above.

    N: int                  # Grid size (N + 1 total pts and N - 1 interior unknowns)
    cfl: float              # stability factor used to construct the pseudo-time step.
    omega: float            # Relaxation parameter for SOR methods
    use_omega_opt: bool     # use analytical estimate for omega
    iter_max: int           # max iteration count
    norm: Any               # Residual norm: 1, 2, or "inf".

    solver_config: Solver1DConfig = field(default_factory=Solver1DConfig)
    plot_enabled: bool = False          # Whether to plot results after the solver run
    plot_history_norm: str = "exact"    # Options: "exact" or "update".
    dt_factor: float = 1.0              # UNUSED extra dt multiplier

    xa: float = 0.0
    xb: float = 1.0
    left_bc: float = 0.0
    right_bc: float = 1.0

    linear_v: float = 0.5               # Diffusivity parameter for linear parabolic methods.
    nonlinear_vmax: float = 0.5         # Initial velocity bound for nonlinear time steps.
    hyperbolic_linear_v: float = 1.0    # Velocity scale for linear hyperbolic SOR.

    # derived
    dx: float = field(init=False)  # Uniform grid spacing.
    ngc: int = field(init=False)  # Ghost-cell count; currently always zero.
    ib: int = field(init=False)  # First interior-node index (inclusive).
    ie: int = field(init=False)  # Last interior-node index (inclusive).
    x: np.ndarray = field(init=False)  # Spatial grid (N+1)
    U0: np.ndarray = field(init=False)  # Initial iterate with boundary values applied.
    U_exact: np.ndarray | None = field(init=False)  # Linear exact solution, if available.

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
        if self.plot_history_norm not in plot_history_norm_options:
            raise ValueError(
                f"Unknown plot_history_norm '{self.plot_history_norm}'. "
                f"Valid options: {sorted(plot_history_norm_options)}"
            )

        self.ngc = 0
        self.ib = 1
        self.ie = self.N - 1
        self.x = np.linspace(self.xa, self.xb, self.N + 1)
        self.dx = (self.xb - self.xa) / self.N

        if self.use_omega_opt:
            self.omega = self.compute_omega_opt()

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

    def compute_omega_opt(self):
        omega_opt = self.omega  # default to manually-set omega
        if "parabolic" in self.method:
            # this formula is really only accurate for the linear problem.
            # for nonlinear problem, use this only for debugging or unofficial tests.
            omega_opt = 2.0 / (self.cfl * (1.0 + np.sin(np.pi / self.N)))
        elif "hyperbolic" in self.method:
            # hyperbolic formula works reasonably well for both linear and nonlinear problems
            omega_opt = 1.0 / (1.0 + np.sin(np.pi / self.N))
        return omega_opt

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
            "plot_enabled": self.plot_enabled,
            "plot_history_norm": self.plot_history_norm,
            "dt_factor": self.dt_factor,
            "xa": self.xa,
            "xb": self.xb,
            "left_bc": self.left_bc,
            "right_bc": self.right_bc,
            "linear_v": self.linear_v,
            "nonlinear_vmax": self.nonlinear_vmax,
            "hyperbolic_linear_v": self.hyperbolic_linear_v,
        }
