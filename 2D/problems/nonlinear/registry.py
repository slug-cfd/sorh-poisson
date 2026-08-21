from . import unit_square_linear_x
from .types import NonlinearCaseDefinition


_REGISTRY: dict[str, NonlinearCaseDefinition] = {
    unit_square_linear_x.CASE.name: unit_square_linear_x.CASE,
}

def get_nonlinear_case(name: str) -> NonlinearCaseDefinition:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(list_nonlinear_cases())
        raise ValueError(
            f"Unknown nonlinear case '{name}'. Valid options: {valid}"
        ) from exc


def list_nonlinear_cases() -> list[str]:
    return sorted(_REGISTRY)
