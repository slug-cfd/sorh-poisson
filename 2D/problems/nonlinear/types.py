from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class NonlinearCaseDefinition:
    """Spatial data that defines one nonlinear boundary-value problem."""

    name: str
    display_name: str
    domain: dict[str, float]
    rhs: Callable
    boundary: Callable
    bc_type: str = "linear_in_x"
