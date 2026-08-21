"""Explicit construction of solver-ready 2D simulation contexts."""

import numpy as np

from problems.nonlinear import build_coefficient, get_nonlinear_case
from problems.poisson_tests import get_poisson_problem

from .models import Config
from .omega import compute_omega_opt
from .runtime import Grid, ProblemInstance, SimulationContext


def _resolve_problem(settings: Config) -> ProblemInstance:
    selection = settings.problem
    if selection.kind == "poisson":
        definition = get_poisson_problem(selection.case)
        unknown_parameters = sorted(
            set(selection.parameters) - set(definition.constants)
        )
        if unknown_parameters:
            raise ValueError(
                f"Unknown parameters for Poisson case '{selection.case}': "
                f"{unknown_parameters}. Expected: {sorted(definition.constants)}."
            )
        constants = dict(definition.constants)
        constants.update(selection.parameters)
        return ProblemInstance(
            name=definition.name,
            display_name=definition.name.replace("_", " ").title(),
            definition=definition,
            rhs=definition.rhs,
            exact=definition.exact,
            boundary=definition.exact,
            coefficient=None,
            bc_type="poisson",
            constants=constants,
        )
    if selection.kind == "nonlinear_diffusion":
        definition = get_nonlinear_case(selection.case)
        return ProblemInstance(
            name=definition.name,
            display_name=definition.display_name,
            definition=definition,
            rhs=definition.rhs,
            exact=None,
            boundary=definition.boundary,
            coefficient=build_coefficient(selection.coefficient),
            bc_type=definition.bc_type,
            constants={},
        )
    return ProblemInstance(
        name="linear_diffusion",
        display_name="Linear Diffusion",
        definition=None,
        rhs=None,
        exact=None,
        boundary=None,
        coefficient=None,
        bc_type="linear_in_x",
        constants={},
    )


def _build_grid(settings: Config, problem: ProblemInstance) -> Grid:
    domain = (
        problem.definition.domain
        if problem.definition is not None
        else {"xa": 0.0, "xb": 1.0, "ya": 0.0, "yb": 1.0}
    )
    grid_config = settings.grid
    minimum_ghosts = 2 if grid_config.laplacian_stencil == "wide" else 1
    ngc = grid_config.ghost_layers or minimum_ghosts
    Nx_full = grid_config.Nx + 2 * ngc
    Ny_full = grid_config.Ny + 2 * ngc
    dx = (domain["xb"] - domain["xa"]) / (grid_config.Nx + 1)
    dy = (domain["yb"] - domain["ya"]) / (grid_config.Ny + 1)
    if (
        grid_config.laplacian_stencil == "isotropic_9pt"
        and not np.isclose(dx, dy)
    ):
        raise ValueError(
            "laplacian_stencil='isotropic_9pt' requires isotropic grid spacing: "
            f"dx={dx}, dy={dy}."
        )
    x_offsets = np.arange(Nx_full) - ngc + 1
    y_offsets = np.arange(Ny_full) - ngc + 1
    x = domain["xa"] + x_offsets * dx
    y = domain["ya"] + y_offsets * dy
    return Grid(
        xa=domain["xa"],
        xb=domain["xb"],
        ya=domain["ya"],
        yb=domain["yb"],
        dx=dx,
        dy=dy,
        ngc=ngc,
        Nx=grid_config.Nx,
        Ny=grid_config.Ny,
        Nx_full=Nx_full,
        Ny_full=Ny_full,
        ib=ngc,
        ie=ngc + grid_config.Nx - 1,
        jb=ngc,
        je=ngc + grid_config.Ny - 1,
        x=x,
        y=y,
    )


def _initialize_arrays(context: SimulationContext):
    X, Y = np.meshgrid(context.x, context.y, indexing="ij")
    if context.settings.problem.initial_condition == "linear_x":
        context.U0[:, :] = context.x[:, None]
    if context.f is not None:
        context.f_all = context._eval_rhs(X, Y)
    if context.phi_exact is not None:
        context.U_exact = context._eval_exact(X, Y)


def resolve_omega(settings: Config, grid: Grid) -> float:
    """Resolve the configured or grid-derived relaxation parameter."""
    if settings.run.use_omega_opt:
        return compute_omega_opt(settings, grid)
    return settings.run.omega


def initialize_simulation(settings: Config) -> SimulationContext:
    """Resolve settings into one fully initialized solver context."""
    problem = _resolve_problem(settings)
    grid = _build_grid(settings, problem)
    effective_omega = resolve_omega(settings, grid)
    context = SimulationContext(settings, grid, problem, effective_omega)
    _initialize_arrays(context)
    return context
