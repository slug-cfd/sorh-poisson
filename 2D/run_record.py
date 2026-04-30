from dataclasses import dataclass, asdict
# import json, time
from config import Config

@dataclass
class RunResult:
    converged: bool
    stable: bool
    iter_count: int
    residual_final: float
    residual_min: float
    residual_max: float
    elapsed_sec: float

@dataclass
class RunRecord:
    config: Config
    results: RunResult
    meta: dict
