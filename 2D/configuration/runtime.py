"""Initialized grid and problem data consumed by 2D solvers."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .models import Config


@dataclass(frozen=True)
class Grid:
    xa: float
    xb: float
    ya: float
    yb: float
    dx: float
    dy: float
    ngc: int
    Nx: int
    Ny: int
    Nx_full: int
    Ny_full: int
    ib: int
    ie: int
    jb: int
    je: int
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ProblemInstance:
    name: str
    display_name: str
    definition: Any
    rhs: Callable | None
    exact: Callable | None
    boundary: Callable | None
    coefficient: Callable | None
    bc_type: str
    constants: dict[str, float]


class SimulationContext:
    """Fully initialized state and derived data for one simulation."""

    def __init__(
        self,
        settings: Config,
        grid: Grid,
        problem: ProblemInstance,
        effective_omega: float,
    ):
        self.settings = settings
        self.grid = grid
        self.problem_instance = problem
        self.omega = float(effective_omega)
        self.iter_max = settings.run.iter_max
        self.U0 = np.zeros((grid.Nx_full, grid.Ny_full), dtype=float)
        self.f_all = None
        self.U_exact = None
        self._nonlinear_boundaries = None

    @property
    def problem(self):
        return self.settings.problem.kind

    @property
    def method(self):
        return self.settings.run.method

    @property
    def cfl(self):
        return self.settings.run.cfl

    @property
    def norm(self):
        return self.settings.run.norm

    @property
    def use_omega_opt(self):
        return self.settings.run.use_omega_opt

    @property
    def dt_factor(self):
        return self.settings.run.dt_factor

    @property
    def timestep_mode(self):
        return self.settings.run.timestep_mode

    @property
    def multigrid(self):
        return self.settings.run.multigrid

    @property
    def stopping_config(self):
        return self.settings.run.stopping

    @property
    def h_unsplit_config(self):
        return self.settings.methods.h_unsplit

    @property
    def p_unsplit_config(self):
        return self.settings.methods.p_unsplit

    @property
    def mg_config(self):
        return self.settings.methods.multigrid

    @property
    def plot(self):
        return self.settings.output.plot

    @property
    def plot_exact(self):
        return self.settings.output.plot_exact

    @property
    def problem_case(self):
        return self.settings.problem.case

    @property
    def laplacian_stencil(self):
        return self.settings.grid.laplacian_stencil

    @property
    def bc_type(self):
        return self.problem_instance.bc_type

    @property
    def f(self):
        return self.problem_instance.rhs

    @property
    def phi_exact(self):
        return self.problem_instance.exact

    def __getattr__(self, name):
        if name in Grid.__dataclass_fields__:
            return getattr(self.grid, name)
        if name in self.problem_instance.constants:
            return self.problem_instance.constants[name]
        raise AttributeError(name)

    def _eval_rhs(self, x, y):
        if self.f is None:
            return None
        return self.f(x, y, self)

    def _eval_exact(self, x, y):
        if self.phi_exact is None:
            return None
        return self.phi_exact(x, y, self)

    def to_dict(self) -> dict:
        result = self.settings.to_dict()
        result["run"]["omega"] = self.omega
        return result
