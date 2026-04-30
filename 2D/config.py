from dataclasses import dataclass, field, asdict
from typing import Any, Callable
import numpy as np
from problems.poisson_tests import get_poisson_problem


problem_name_map = {
    'poisson': "Poisson",
    'nonlinear': "Nonlinear",
    'linear': "Linear",
}

method_name_map = {
    'p_unsplit': "Parabolic/Unsplit",
    'p_ADI': "Parabolic/ADI",
    'h_unsplit': "Hyperbolic/Unsplit",
    'h_ADI': "Hyperbolic/ADI",
}

@dataclass
class HyperbolicUnsplitConfig:
    # mode selection: "default", "default2", "directional", "weighted_hybrid"
    mode: str = "directional"

    # directional mode controls
    use_theta: bool = True
    theta: float = 0.25 * np.pi

    # normalize (bx, by) to sqrt(2)
    normalize_direction: bool = True
    direction_norm: float = np.sqrt(2.0)

    # weighted hybrid controls
    alpha: float = 0.8
    beta: float = 0.8
    eps_alpha: float = 0.05


@dataclass
class MultiGridConfig:
    solver: str = "fmg"     # "rep_v_cycles" or "fmg"
    max_cycles: int = 200
    tol_update: float = 1e-5
    tol_residual: float = 1e-5
    res_max: float = 1e8
    nu1: int = 2
    nu2: int = 2
    min_coarse_n: int = 4
    coarse_max_sweeps: int = 100
    coarse_tol: float = 1e-6
    restriction: str = "full_weighting"
    prolongation: str = "bilinear"


@dataclass
class Config:
    # — user‐set —
    problem:    str     # 'poisson', 'toy_linear', 'toy_nonlinear'
    method:     str     # 'p_unsplit', 'h_unsplit', 'p_ADI', 'h_ADI', etc.
    bc_type:    str     # 'poisson', 'linear_in_x', etc.

    Nx:         int     # number of INTERIOR cells (x-direction)
    Ny:         int     # number of INTERIOR cells (y-direction)
    cfl:        float   # CFL number

    omega:      float   # SOR over/under-relaxation parameter
    iter_max:   int     # maximum allowed iterations

    norm:       Any     # norm type. options: [1, 2, "fro", np.inf]

    mg_config:  MultiGridConfig     # multigrid parameters
    h_unsplit_config: HyperbolicUnsplitConfig

    dt_factor: float = 1.0      # dt := dt_factor * [initial dt]

    f:          Callable = None     # RHS definition
    phi_exact:  Callable = None     # function for exact solution calculation

    multigrid:  bool = False        # use multigrid?
    poisson_problem: str = "two_blob"  # selected modular Poisson test problem

    # — derived —
    xa:         float   = field(init=False)  # left physical boundary (x)
    xb:         float   = field(init=False)  # right physical boundary (x)
    ya:         float   = field(init=False)  # left physical boundary (y)
    yb:         float   = field(init=False)  # right physical boundary (y)

    dx:         float   = field(init=False)
    dy:         float   = field(init=False)

    ngc:        int     = field(init=False)  # number of ghost cells per side
    ib:         int     = field(init=False)  # starting interior index (x)
    ie:         int     = field(init=False)  # ending interior index (x)
    jb:         int     = field(init=False)  # starting interior index (y)
    je:         int     = field(init=False)  # ending interior index (y)

    # Poisson-specific parameters
    poisson_problem_def: Any = field(init=False)
    R:          float = field(init=False)
    rho0:       float = field(init=False)
    G:          float = field(init=False)

    # — initial state arrays —
    x:          np.ndarray = field(init=False)  # includes ghosts
    y:          np.ndarray = field(init=False)  # "        "

    U0:         np.ndarray = field(init=False)  # initial state array

    f_all:      np.ndarray = field(init=False)  # exact RHS
    U_exact:    np.ndarray = field(init=False)  # exact solution (for Poisson problem)


    def __post_init__(self):
        self.f_all = None
        self.U_exact = None
        self.poisson_problem_def = None

        # 1st quadrant only for Poisson
        self.xa = 0.0
        self.ya = 0.0

        if self.problem == "poisson":
            self.poisson_problem_def = get_poisson_problem(self.poisson_problem)
            self.bc_type = "poisson"
            self.ngc = 1

            constants = self.poisson_problem_def.constants
            self.R = constants.get("R", 1.0)
            self.rho0 = constants.get("rho0", 1.0)
            self.G = constants.get("G", 0.25 / np.pi)
            self.epsilon = constants.get("epsilon", 1.0)

            domain = self.poisson_problem_def.domain
            self.xa = domain["xa"]
            self.xb = domain["xb"]
            self.ya = domain["ya"]
            self.yb = domain["yb"]

            if self.f is None:
                self.f = self.poisson_problem_def.rhs
            if self.phi_exact is None:
                self.phi_exact = self.poisson_problem_def.exact
        else:
            self.bc_type = "linear_in_x"    # toy problem setup
            self.ngc = 1
            self.xb = 1.0
            self.yb = 1.0

        self.ib = self.ngc
        self.jb = self.ngc
        self.ie = self.Nx - self.ngc - 1
        self.je = self.Ny - self.ngc - 1

        # np.linspace includes the endpoints; these will also
        # automatically include ghost cells depending on self.ngc
        self.x = np.linspace(self.xa, self.xb, self.Nx)
        self.y = np.linspace(self.ya, self.yb, self.Ny)

        self.dx = (self.xb - self.xa) / (self.Nx - 1)
        self.dy = (self.yb - self.ya) / (self.Ny - 1)

        # initialize U0
        self.U0 = np.zeros((self.Nx, self.Ny))

        if self.f:
            # apply f(x, y) over the whole 2D grid
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')  # shape (Nx+2*ngc, Ny+2*ngc)
            self.f_all = self._eval_rhs(X, Y)

        if self.phi_exact:
            # set the exact solution on full grid
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            self.U_exact = self._eval_exact(X, Y)

    def _eval_rhs(self, x, y):
        try:
            return self.f(x, y, self)
        except TypeError:
            return self.f(x, y, self.rho0, self.R)

    def _eval_exact(self, x, y):
        try:
            return self.phi_exact(x, y, self)
        except TypeError:
            return self.phi_exact(x, y, self.rho0, self.R, self.G)

    def to_dict(self) -> dict:
        """Return only the user-set fields, safe for JSON serialization."""
        return {
            "problem":    self.problem,
            "poisson_problem": self.poisson_problem,
            "method":     self.method,
            "bc_type":    self.bc_type,
            "Nx":         self.Nx,
            "Ny":         self.Ny,
            "cfl":        self.cfl,
            "omega":      self.omega,
            "iter_max":   self.iter_max,
            "norm":       self.norm if not isinstance(self.norm, float) else "inf",
            "dt_factor":  self.dt_factor,
            "multigrid":  self.multigrid,
            "f":          self.f.__name__ if self.f else None,
            "phi_exact":  self.phi_exact.__name__ if self.phi_exact else None,
            "mg_config":       asdict(self.mg_config),
            "h_unsplit_config": asdict(self.h_unsplit_config),
        }
