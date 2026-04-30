from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class PoissonTestProblem:
    name: str
    domain: dict[str, float]
    rhs: Callable
    exact: Callable
    bc_type: str = "dirichlet_exact"
    constants: dict[str, float] = field(default_factory=dict)
