from dataclasses import asdict, dataclass, field
from typing import Any
import warnings

import numpy as np

from problem_helpers import MIXED_MODE_PROFILE_OPTIONS, mixed_mode_profile


# Accepted problem names mapped to their display labels.
problem_name_map = {
    "linear": "Linear affine BVP",
    "linear_mixed_mode": "Linear mixed-mode error",
    "nonlinear": "Nonlinear",
}

# History plotted beside the final solution.
plot_history_norm_options = {"equation", "exact", "update"}

# Accepted solver names mapped to their display labels.
method_name_map = {
    "jacobi_linear_parabolic": "Jacobi/Linear/Parabolic",
    "jacobi_nonlinear_parabolic": "Jacobi/Nonlinear/Parabolic",
    "gs_linear_parabolic": "Gauss-Seidel/Linear/Parabolic",
    "gs_nonlinear_parabolic": "Gauss-Seidel/Nonlinear/Parabolic",
    "sor_linear_parabolic": "SOR/Linear/Parabolic",
    "sor_linear_parabolic_fourth": "SORp/Linear/4th-order",
    "sor_nonlinear_parabolic": "SOR/Nonlinear/Parabolic",
    "sor_linear_hyperbolic": "SOR/Linear/Hyperbolic",
    "sor_linear_hyperbolic_fourth_comp": "SORh/Linear/4th-order/P-comp",
    "sor_linear_hyperbolic_fourth_bdf2": "SORh/Linear/4th-order/BDF2",
    "sor_nonlinear_hyperbolic": "SOR/Nonlinear/Hyperbolic",
}

FOURTH_ORDER_METHODS = {
    "sor_linear_parabolic_fourth",
    "sor_linear_hyperbolic_fourth_comp",
    "sor_linear_hyperbolic_fourth_bdf2",
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
    plot_history_norm: str = "exact"    # Options: "exact", "update", or "equation".
    initial_profile: str = "mixed"      # Profile used by the mixed-mode linear problem.
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
    ngc: int = field(init=False)  # Exterior guard-node count on each side.
    left_boundary: int = field(init=False)
    right_boundary: int = field(init=False)
    ib: int = field(init=False)  # First interior-node index (inclusive).
    ie: int = field(init=False)  # Last interior-node index (inclusive).
    x: np.ndarray = field(init=False)  # Full grid, including any guard nodes.
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
        if (
            self.problem == "linear_mixed_mode"
            and self.initial_profile not in MIXED_MODE_PROFILE_OPTIONS
        ):
            raise ValueError(
                f"Unknown mixed-mode profile '{self.initial_profile}'. "
                f"Valid options: {sorted(MIXED_MODE_PROFILE_OPTIONS)}"
            )
        if self.problem == "linear_mixed_mode" and not (
            np.isclose(self.left_bc, 0.0) and np.isclose(self.right_bc, 0.0)
        ):
            raise ValueError(
                "The linear_mixed_mode problem requires homogeneous Dirichlet "
                "conditions: left_bc=right_bc=0."
            )
        if (
            self.plot_history_norm == "equation"
            and self.problem != "linear_mixed_mode"
        ):
            raise ValueError(
                "plot_history_norm='equation' is currently available only for "
                "the linear_mixed_mode problem."
            )
        if self.method in FOURTH_ORDER_METHODS and self.use_omega_opt:
            warnings.warn(
                "The existing analytical omega estimate has not been established "
                "for the fourth-order methods; applying it as an experimental "
                "value.",
                UserWarning,
                stacklevel=2,
            )

        self.dx = (self.xb - self.xa) / self.N
        self.ngc = 1 if self.method in FOURTH_ORDER_METHODS else 0
        self.left_boundary = self.ngc
        self.right_boundary = self.ngc + self.N
        self.ib = self.left_boundary + 1
        self.ie = self.right_boundary - 1
        self.x = np.linspace(
            self.xa - self.ngc * self.dx,
            self.xb + self.ngc * self.dx,
            self.N + 1 + 2 * self.ngc,
        )

        if self.use_omega_opt:
            self.omega = self.compute_omega_opt()

        self.U0 = np.zeros_like(self.x)
        if self.problem == "linear_mixed_mode":
            self.U0[self.physical_slice] = mixed_mode_profile(
                self.x_physical,
                self.initial_profile,
            )
        self.apply_bcs(self.U0)

        if self.problem == "linear":
            slope = (self.right_bc - self.left_bc) / (self.xb - self.xa)
            self.U_exact = self.left_bc + slope * (self.x - self.xa)
        elif self.problem == "linear_mixed_mode":
            self.U_exact = np.zeros_like(self.x)
        else:
            self.U_exact = None

    @property
    def interior_size(self) -> int:
        return self.ie - self.ib + 1

    @property
    def physical_slice(self) -> slice:
        return slice(self.left_boundary, self.right_boundary + 1)

    @property
    def interior_slice(self) -> slice:
        return slice(self.ib, self.ie + 1)

    @property
    def x_physical(self) -> np.ndarray:
        return self.x[self.physical_slice]

    @property
    def uses_fourth_order_laplacian(self) -> bool:
        return self.method in FOURTH_ORDER_METHODS

    @property
    def problem_display_name(self) -> str:
        label = problem_name_map[self.problem]
        if self.problem == "linear_mixed_mode":
            return f"{label} ({self.initial_profile})"
        return label

    @property
    def run_display_name(self) -> str:
        return f"{method_name_map[self.method]} — {self.problem_display_name}"

    def apply_bcs(self, U: np.ndarray) -> np.ndarray:
        U[self.left_boundary] = self.left_bc
        U[self.right_boundary] = self.right_bc

        for layer in range(1, self.ngc + 1):
            U[self.left_boundary - layer] = (
                2.0 * self.left_bc - U[self.left_boundary + layer]
            )
            U[self.right_boundary + layer] = (
                2.0 * self.right_bc - U[self.right_boundary - layer]
            )
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
            "initial_profile": self.initial_profile,
            "dt_factor": self.dt_factor,
            "xa": self.xa,
            "xb": self.xb,
            "left_bc": self.left_bc,
            "right_bc": self.right_bc,
            "linear_v": self.linear_v,
            "nonlinear_vmax": self.nonlinear_vmax,
            "hyperbolic_linear_v": self.hyperbolic_linear_v,
            "ngc": self.ngc,
        }
