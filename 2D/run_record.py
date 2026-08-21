from dataclasses import dataclass

from configuration import SimulationContext


@dataclass
class RunResult:
    converged: bool
    stable: bool
    iter_count: int
    residual_final: float
    residual_min: float
    residual_max: float
    elapsed_sec: float
    reason: str


@dataclass
class RunRecord:
    config: SimulationContext
    results: RunResult
    meta: dict
