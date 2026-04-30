from . import directional_ridge, polar_star, polar_star_smooth, quadtree_poisson_1, two_blob
from .types import PoissonTestProblem


_REGISTRY: dict[str, PoissonTestProblem] = {
    two_blob.PROBLEM.name: two_blob.PROBLEM,
    quadtree_poisson_1.PROBLEM.name: quadtree_poisson_1.PROBLEM,
    directional_ridge.PROBLEM.name: directional_ridge.PROBLEM,
    polar_star.PROBLEM.name: polar_star.PROBLEM,
    polar_star_smooth.PROBLEM.name: polar_star_smooth.PROBLEM,
}


def get_poisson_problem(name: str) -> PoissonTestProblem:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(list_poisson_problems())
        raise ValueError(f"Unknown Poisson test problem '{name}'. Valid options: {valid}") from exc


def list_poisson_problems() -> list[str]:
    return sorted(_REGISTRY)
