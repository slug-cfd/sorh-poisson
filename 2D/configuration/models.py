"""Declarative, JSON-serializable settings for 2D simulations."""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HyperbolicUnsplitConfig:
    mode: str = "default"
    theta: float = 0.25 * np.pi
    normalize_direction: bool = True
    direction_norm: float = np.sqrt(2.0)
    alpha: float = 0.8
    beta: float = 0.8
    eps_alpha: float = 0.05
    default2_blend_eta: float = 0.5

    def __post_init__(self):
        valid_modes = {
            "default",
            "red_black",
            "default2",
            "default2_blend",
            "directional",
            "weighted_hybrid",
        }
        if self.mode not in valid_modes:
            raise ValueError(
                f"Unknown h_unsplit mode='{self.mode}'. "
                f"Expected one of: {sorted(valid_modes)}."
            )


@dataclass(frozen=True)
class ParabolicUnsplitConfig:
    gs_ordering: str = "lexicographic"
    coefficient_mode: str = "pointwise"

    def __post_init__(self):
        valid_orderings = {"lexicographic", "red_black"}
        if self.gs_ordering not in valid_orderings:
            raise ValueError(
                f"Unknown p_unsplit gs_ordering='{self.gs_ordering}'. "
                f"Expected one of: {sorted(valid_orderings)}."
            )
        valid_coefficient_modes = {"pointwise", "frozen_per_sweep"}
        if self.coefficient_mode not in valid_coefficient_modes:
            raise ValueError(
                "Unknown p_unsplit coefficient_mode="
                f"'{self.coefficient_mode}'. Expected one of: "
                f"{sorted(valid_coefficient_modes)}."
            )


@dataclass(frozen=True)
class StoppingConfig:
    tol_update: float = 1e-10
    tol_residual: float = 1e-5
    res_max: float = 1e8

    def __post_init__(self):
        values = (self.tol_update, self.tol_residual, self.res_max)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Stopping tolerances and res_max must be finite.")
        if self.tol_update <= 0.0 or self.tol_residual <= 0.0:
            raise ValueError("Stopping tolerances must be positive.")
        if self.res_max <= max(self.tol_update, self.tol_residual):
            raise ValueError("res_max must exceed both stopping tolerances.")


@dataclass(frozen=True)
class MultiGridConfig:
    solver: str = "fmg"
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


@dataclass(frozen=True)
class NonlinearCoefficientConfig:
    model: str = "regularized"
    regularization_epsilon: float = 0.0
    constant_value: float = 0.5

    def __post_init__(self):
        if self.model not in {"regularized", "constant"}:
            raise ValueError(
                "nonlinear coefficient model must be 'regularized' or 'constant'."
            )
        if (
            not np.isfinite(self.regularization_epsilon)
            or self.regularization_epsilon < 0.0
        ):
            raise ValueError(
                "nonlinear regularization_epsilon must be finite and nonnegative."
            )
        if not np.isfinite(self.constant_value) or self.constant_value <= 0.0:
            raise ValueError(
                "nonlinear constant coefficient value must be finite and positive."
            )


@dataclass(frozen=True)
class ProblemConfig:
    kind: str
    case: str
    parameters: dict[str, float] = field(default_factory=dict)
    coefficient: NonlinearCoefficientConfig = field(
        default_factory=NonlinearCoefficientConfig
    )
    initial_condition: str = "zero"

    def __post_init__(self):
        valid_kinds = {"poisson", "nonlinear_diffusion", "linear_diffusion"}
        if self.kind not in valid_kinds:
            raise ValueError(
                f"Unknown 2D problem '{self.kind}'. "
                f"Expected one of: {sorted(valid_kinds)}."
            )
        if self.initial_condition not in {"zero", "linear_x"}:
            raise ValueError(
                "initial_condition must be 'zero' or 'linear_x'."
            )
        if self.parameters and self.kind != "poisson":
            raise ValueError(
                "problem parameters are currently supported only for Poisson cases."
            )
        for name, value in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("problem parameter names must be nonempty strings.")
            if not np.isfinite(value):
                raise ValueError(
                    f"problem parameter '{name}' must be finite; got {value}."
                )


@dataclass(frozen=True)
class GridConfig:
    Nx: int
    Ny: int
    laplacian_stencil: str = "centered"
    ghost_layers: int | None = None

    def __post_init__(self):
        if self.Nx < 1 or self.Ny < 1:
            raise ValueError(
                "Nx and Ny are interior counts and must both be positive: "
                f"Nx={self.Nx}, Ny={self.Ny}."
            )
        valid_stencils = {"centered", "wide", "isotropic_9pt"}
        if self.laplacian_stencil not in valid_stencils:
            raise ValueError(
                f"Unknown laplacian_stencil='{self.laplacian_stencil}'. "
                f"Expected one of: {sorted(valid_stencils)}."
            )
        minimum = 2 if self.laplacian_stencil == "wide" else 1
        if self.ghost_layers is not None and self.ghost_layers < minimum:
            raise ValueError(
                f"laplacian_stencil='{self.laplacian_stencil}' requires "
                f"at least {minimum} ghost layer(s)."
            )


@dataclass(frozen=True)
class RunConfig:
    method: str
    cfl: float
    omega: float
    iter_max: int
    norm: Any
    use_omega_opt: bool = False
    dt_factor: float = 1.0
    timestep_mode: str = "fixed"
    multigrid: bool = False
    stopping: StoppingConfig = field(default_factory=StoppingConfig)

    def __post_init__(self):
        if not np.isfinite(self.cfl) or self.cfl <= 0.0:
            raise ValueError(f"cfl must be finite and positive; got {self.cfl}.")
        if not np.isfinite(self.omega) or self.omega <= 0.0:
            raise ValueError(f"omega must be finite and positive; got {self.omega}.")
        if self.iter_max < 1:
            raise ValueError(f"iter_max must be at least 1; got {self.iter_max}.")
        if self.norm not in {1, 2, "fro", np.inf}:
            raise ValueError(
                f"Unknown norm '{self.norm}'. Expected one of: 1, 2, 'fro', inf."
            )
        if self.timestep_mode not in {"fixed", "adaptive"}:
            raise ValueError(
                f"Unknown nonlinear timestep_mode='{self.timestep_mode}'. "
                "Expected one of: ['adaptive', 'fixed']."
            )


@dataclass(frozen=True)
class MethodConfig:
    h_unsplit: HyperbolicUnsplitConfig = field(
        default_factory=HyperbolicUnsplitConfig
    )
    p_unsplit: ParabolicUnsplitConfig = field(
        default_factory=ParabolicUnsplitConfig
    )
    multigrid: MultiGridConfig = field(default_factory=MultiGridConfig)


@dataclass(frozen=True)
class OutputConfig:
    plot: bool = False
    plot_exact: bool = False


@dataclass(frozen=True)
class Config:
    problem: ProblemConfig
    grid: GridConfig
    run: RunConfig
    methods: MethodConfig = field(default_factory=MethodConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        if self.problem.kind == "nonlinear_diffusion":
            valid_methods = {"p_unsplit", "h_unsplit", "DS1", "DS2"}
            if self.run.method not in valid_methods:
                raise ValueError(
                    f"Unknown nonlinear_diffusion method '{self.run.method}'. "
                    f"Expected one of: {sorted(valid_methods)}."
                )

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.run.norm == np.inf:
            result["run"]["norm"] = "inf"
        return result
