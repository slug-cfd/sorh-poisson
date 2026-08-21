"""Public configuration and initialization interface for the 2D solvers."""

from .initialization import initialize_simulation, resolve_omega
from .loading import load_config
from .models import (
    Config,
    GridConfig,
    HyperbolicUnsplitConfig,
    MethodConfig,
    MultiGridConfig,
    NonlinearCoefficientConfig,
    OutputConfig,
    ParabolicUnsplitConfig,
    ProblemConfig,
    RunConfig,
    StoppingConfig,
)
from .runtime import Grid, ProblemInstance, SimulationContext
from .omega import compute_omega_opt

__all__ = [
    "Config",
    "Grid",
    "GridConfig",
    "HyperbolicUnsplitConfig",
    "MethodConfig",
    "MultiGridConfig",
    "NonlinearCoefficientConfig",
    "OutputConfig",
    "ParabolicUnsplitConfig",
    "ProblemConfig",
    "ProblemInstance",
    "RunConfig",
    "SimulationContext",
    "StoppingConfig",
    "compute_omega_opt",
    "initialize_simulation",
    "load_config",
    "resolve_omega",
]
