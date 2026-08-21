"""JSON loading for declarative 2D simulation settings."""

import json
from pathlib import Path
from typing import Any

import numpy as np

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


_NORM_MAP: dict[str, Any] = {
    "1": 1,
    "2": 2,
    "inf": np.inf,
    "fro": "fro",
}


def _parse_norm(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _NORM_MAP:
            return _NORM_MAP[key]
        raise ValueError(f"Unknown norm '{value}'. Valid options: {list(_NORM_MAP)}")
    return value


def _stopping_config(raw):
    return StoppingConfig(
        tol_update=raw.get("tol_update", 1e-10),
        tol_residual=raw.get("tol_residual", 1e-5),
        res_max=raw.get("res_max", 1e8),
    )


def _multigrid_config(raw):
    return MultiGridConfig(
        solver=raw.get("solver", "fmg"),
        max_cycles=raw.get("max_cycles", 200),
        tol_update=raw.get("tol_update", 1e-5),
        tol_residual=raw.get("tol_residual", 1e-5),
        res_max=raw.get("res_max", 1e8),
        nu1=raw.get("nu1", 2),
        nu2=raw.get("nu2", 2),
        min_coarse_n=raw.get("min_coarse_n", 4),
        coarse_max_sweeps=raw.get("coarse_max_sweeps", 100),
        coarse_tol=raw.get("coarse_tol", 1e-6),
        restriction=raw.get("restriction", "full_weighting"),
        prolongation=raw.get("prolongation", "bilinear"),
    )


def _h_unsplit_config(raw):
    return HyperbolicUnsplitConfig(
        mode=raw.get("mode", "default"),
        theta=raw.get("theta", 0.25 * np.pi),
        normalize_direction=raw.get("normalize_direction", True),
        direction_norm=raw.get("direction_norm", np.sqrt(2.0)),
        alpha=raw.get("alpha", 0.8),
        beta=raw.get("beta", 0.8),
        eps_alpha=raw.get("eps_alpha", 0.05),
        default2_blend_eta=raw.get("default2_blend_eta", 0.5),
    )


def _p_unsplit_config(raw):
    return ParabolicUnsplitConfig(
        gs_ordering=raw.get("gs_ordering", "lexicographic"),
        coefficient_mode=raw.get("coefficient_mode", "pointwise"),
    )


def _config_from_dict(raw):
    problem_raw = raw["problem"]
    grid_raw = raw["grid"]
    run_raw = raw["run"]
    methods_raw = raw.get("methods", {})
    output_raw = raw.get("output", {})
    coefficient_raw = problem_raw.get("coefficient", {})
    return Config(
        problem=ProblemConfig(
            kind=problem_raw["kind"],
            case=problem_raw["case"],
            parameters=dict(problem_raw.get("parameters", {})),
            coefficient=NonlinearCoefficientConfig(
                model=coefficient_raw.get("model", "regularized"),
                regularization_epsilon=coefficient_raw.get(
                    "regularization_epsilon", 0.0
                ),
                constant_value=coefficient_raw.get("constant_value", 0.5),
            ),
            initial_condition=problem_raw.get("initial_condition", "zero"),
        ),
        grid=GridConfig(
            Nx=grid_raw["Nx"],
            Ny=grid_raw["Ny"],
            laplacian_stencil=grid_raw.get("laplacian_stencil", "centered"),
            ghost_layers=grid_raw.get("ghost_layers"),
        ),
        run=RunConfig(
            method=run_raw["method"],
            cfl=run_raw["cfl"],
            omega=run_raw.get("omega", 1.0),
            iter_max=run_raw["iter_max"],
            norm=_parse_norm(run_raw["norm"]),
            use_omega_opt=run_raw.get("use_omega_opt", False),
            dt_factor=run_raw.get("dt_factor", 1.0),
            timestep_mode=run_raw.get("timestep_mode", "fixed"),
            multigrid=run_raw.get("multigrid", False),
            stopping=_stopping_config(run_raw.get("stopping", {})),
        ),
        methods=MethodConfig(
            h_unsplit=_h_unsplit_config(methods_raw.get("h_unsplit", {})),
            p_unsplit=_p_unsplit_config(methods_raw.get("p_unsplit", {})),
            multigrid=_multigrid_config(methods_raw.get("multigrid", {})),
        ),
        output=OutputConfig(
            plot=output_raw.get("plot", False),
            plot_exact=output_raw.get("plot_exact", False),
        ),
    )


def load_config(path: str | Path) -> Config:
    """Load declarative settings without initializing solver arrays."""
    with Path(path).open() as stream:
        raw = json.load(stream)
    if not isinstance(raw.get("problem"), dict):
        raise ValueError(
            "2D configs must use the nested problem/grid/run structure."
        )
    return _config_from_dict(raw)
