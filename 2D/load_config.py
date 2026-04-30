import json
from pathlib import Path

from config import *
from problems import poisson2D, nonlinear2D, linear2D
from problems.poisson_tests import get_poisson_problem

# ---------------------------------------------------------------------------
# Registry of callable functions referenced by name in JSON.
# ---------------------------------------------------------------------------
FUNCTION_REGISTRY: dict[str, Callable] = {
    "poisson_rhs":   poisson2D.f,
    "poisson_exact": poisson2D.phi_exact,
}

# ---------------------------------------------------------------------------
# Norm helper: JSON stores norms as strings or numbers; convert back.
# ---------------------------------------------------------------------------
_NORM_MAP: dict[str, Any] = {
    "1":   1,
    "2":   2,
    "inf": np.inf,
    "fro": "fro",
}

def _parse_norm(value: str | int | float) -> Any:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _NORM_MAP:
            return _NORM_MAP[key]
        raise ValueError(f"Unknown norm '{value}'. Valid options: {list(_NORM_MAP)}")
    return value  # already a number (1, 2, etc.)


# ---------------------------------------------------------------------------
# Sub-config loaders
# ---------------------------------------------------------------------------

def _load_mg_config(d: dict) -> MultiGridConfig:
    return MultiGridConfig(
        solver           = d.get("solver",            "fmg"),
        max_cycles       = d.get("max_cycles",         200),
        tol_update       = d.get("tol_update",         1e-5),
        tol_residual     = d.get("tol_residual",       1e-5),
        res_max          = d.get("res_max",            1e8),
        nu1              = d.get("nu1",                2),
        nu2              = d.get("nu2",                2),
        min_coarse_n     = d.get("min_coarse_n",       4),
        coarse_max_sweeps= d.get("coarse_max_sweeps",  100),
        coarse_tol       = d.get("coarse_tol",         1e-6),
        restriction      = d.get("restriction",       "full_weighting"),
        prolongation     = d.get("prolongation",      "bilinear"),
    )

def _load_h_unsplit_config(d: dict) -> HyperbolicUnsplitConfig:
    return HyperbolicUnsplitConfig(
        mode                = d.get("mode",               "directional"),
        use_theta           = d.get("use_theta",           True),
        theta               = d.get("theta",               0.25 * np.pi),
        normalize_direction = d.get("normalize_direction", True),
        direction_norm      = d.get("direction_norm",      np.sqrt(2.0)),
        alpha               = d.get("alpha",               0.8),
        beta                = d.get("beta",                0.8),
        eps_alpha           = d.get("eps_alpha",           0.05),
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def config_from_json(path: str | Path) -> Config:
    """
    Load a Config object from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON configuration file.

    Returns
    -------
    Config
        A fully initialized Config object ready to pass to your solver.

    Raises
    ------
    KeyError
        If a required field is missing from the JSON.
    ValueError
        If a callable name is not found in FUNCTION_REGISTRY,
        or if an unrecognised norm string is provided.
    """
    path = Path(path)
    with path.open() as fh:
        raw = json.load(fh)

    poisson_problem = raw.get("poisson_problem", "two_blob")
    if raw["problem"] == "poisson":
        get_poisson_problem(poisson_problem)

    # -- resolve callables from registry (None if key absent or null) --
    def _resolve(key: str) -> Callable | None:
        name = raw.get(key)
        if name is None:
            return None
        if name not in FUNCTION_REGISTRY:
            raise ValueError(
                f"'{name}' (field '{key}') not found in FUNCTION_REGISTRY. "
                f"Available: {list(FUNCTION_REGISTRY)}"
            )
        return FUNCTION_REGISTRY[name]

    has_modular_poisson_selection = raw["problem"] == "poisson" and "poisson_problem" in raw

    return Config(
        # required fields
        problem   = raw["problem"],
        poisson_problem = poisson_problem,
        method    = raw["method"],
        bc_type   = raw["bc_type"],
        Nx        = raw["Nx"],
        Ny        = raw["Ny"],
        cfl       = raw["cfl"],
        omega     = raw["omega"],
        iter_max  = raw["iter_max"],
        norm      = _parse_norm(raw["norm"]),
        mg_config       = _load_mg_config(raw.get("mg_config", {})),
        h_unsplit_config= _load_h_unsplit_config(raw.get("h_unsplit_config", {})),
        # optional fields (dataclass defaults apply when absent)
        dt_factor = raw.get("dt_factor", 1.0),
        multigrid = raw.get("multigrid", False),
        f         = None if has_modular_poisson_selection else _resolve("f"),
        phi_exact = None if has_modular_poisson_selection else _resolve("phi_exact"),
    )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = config_from_json(Path(__file__).resolve().parent / "config_sample.json")
    print(cfg)
