"""Service-time injection with a strict historical adapter."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Protocol
import math
from fidelityloop.legacy.maxopt_v2.calibrated import service_seconds


class ServiceEstimator(Protocol):
    def seconds(self, row: Mapping, concurrency: int) -> float: ...


@dataclass(frozen=True)
class HistoricalServiceEstimator:
    """Calls the frozen V3/V2 service function without changing its arithmetic."""
    model: Mapping
    def seconds(self, row, concurrency):
        return service_seconds(self.model, row, concurrency)


@dataclass(frozen=True)
class ConstantServiceEstimator:
    """Synthetic estimator used only by bounded CPU extension examples."""
    seconds_per_job: float = 1.0
    max_concurrency: int = 8
    def __post_init__(self):
        if not math.isfinite(self.seconds_per_job) or self.seconds_per_job <= 0:
            raise ValueError("seconds_per_job must be positive")
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 8:
            raise ValueError("max_concurrency must be within 1..8")
    def seconds(self, row, concurrency):
        if type(concurrency) is not int or not 1 <= concurrency <= self.max_concurrency:
            raise ValueError("unsupported local concurrency")
        return float(self.seconds_per_job)
