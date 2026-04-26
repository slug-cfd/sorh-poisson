import json
from pathlib import Path
from typing import Any

import numpy as np

from config1D import Config1D, Solver1DConfig


_NORM_MAP: dict[str, Any] = {
    "1": 1,
    "2": 2,
    "inf": np.inf,
}


def _parse_norm(value: str | int | float) -> Any:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _NORM_MAP:
            return _NORM_MAP[key]
        raise ValueError(f"Unknown norm '{value}'. Valid options: {list(_NORM_MAP)}")
    return value


def _load_solver_config(raw: dict) -> Solver1DConfig:
    return Solver1DConfig(
        tol_update=raw.get("tol_update", 1e-7),
        tol_exact=raw.get("tol_exact"),
        res_max=raw.get("res_max", 1e8),
        stop_on_exact=raw.get("stop_on_exact", False),
    )


def config_from_json(path: str | Path) -> Config1D:
    path = Path(path)
    with path.open() as fh:
        raw = json.load(fh)

    return Config1D(
        problem=raw["problem"],
        method=raw["method"],
        N=raw["N"],
        cfl=raw["cfl"],
        omega=raw.get("omega", 1.0),
        iter_max=raw["iter_max"],
        norm=_parse_norm(raw.get("norm", 2)),
        solver_config=_load_solver_config(raw.get("solver_config", {})),
        dt_factor=raw.get("dt_factor", 1.0),
        xa=raw.get("xa", 0.0),
        xb=raw.get("xb", 1.0),
        left_bc=raw.get("left_bc", 0.0),
        right_bc=raw.get("right_bc", 1.0),
        linear_v=raw.get("linear_v", 0.5),
        nonlinear_vmax=raw.get("nonlinear_vmax", 0.5),
        hyperbolic_linear_v=raw.get("hyperbolic_linear_v", 1.0),
    )


if __name__ == "__main__":
    cfg = config_from_json(Path(__file__).with_name("config_sample1D.json"))
    print(cfg)
