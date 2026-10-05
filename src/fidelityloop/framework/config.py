"""Small, validated configuration surface for CPU framework examples."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Any
import math


@dataclass(frozen=True)
class FrameworkConfig:
    """Finite configuration; bounds are intentional, not a cluster contract."""
    window_seconds: int = 30
    drain_seconds: int = 20
    device_count: int = 2
    local_slots: int = 4
    startup_seconds: int = 0
    shutdown_seconds: int = 0
    api_capacity: int = 8
    api_latency_seconds: int = 5
    api_route_wait_seconds: int = 10
    initial_active_devices: int = 1
    gpu_second_price: float = 1.0
    startup_event_price: float = 1.0
    shutdown_event_price: float = 1.0
    api_input_price: float = 0.0
    api_output_price: float = 0.0
    offline_miss_price: float = 1.0
    online_sla_seconds: float = 60.0
    historical_contract: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    historical_model: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    historical_safety: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    historical_guard_model: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    historical_scale: bool = False
    historical_reservation: bool = False

    def __post_init__(self):
        ints = {
            "window_seconds": self.window_seconds, "drain_seconds": self.drain_seconds,
            "device_count": self.device_count, "local_slots": self.local_slots,
            "startup_seconds": self.startup_seconds, "shutdown_seconds": self.shutdown_seconds,
            "api_capacity": self.api_capacity, "api_latency_seconds": self.api_latency_seconds,
            "api_route_wait_seconds": self.api_route_wait_seconds,
            "initial_active_devices": self.initial_active_devices,
        }
        for name, value in ints.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if self.window_seconds < 1 or self.drain_seconds < 0:
            raise ValueError("window_seconds must be positive and drain_seconds nonnegative")
        if not 1 <= self.device_count <= 8:
            raise ValueError("bounded facade supports 1..8 devices")
        if not 1 <= self.local_slots <= 8:
            raise ValueError("bounded facade supports 1..8 local slots per device")
        if not 0 <= self.initial_active_devices <= self.device_count:
            raise ValueError("initial_active_devices exceeds device_count")
        if self.api_capacity < 1 or self.api_latency_seconds < 1 or self.api_route_wait_seconds < 0:
            raise ValueError("API capacity/latency/wait bounds are invalid")
        for name in ("gpu_second_price", "startup_event_price", "shutdown_event_price",
                     "api_input_price", "api_output_price", "offline_miss_price",
                     "online_sla_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @classmethod
    def from_historical(cls, contract, model, *, safety=None, guard_model=None,
                        scale=False, reservation=False):
        """Bind the accepted two-device contract for exact historical delegation."""
        revision = contract["workload_revision"]
        return cls(window_seconds=revision["window_seconds"],
                   drain_seconds=revision["drain_seconds"], device_count=2, local_slots=4,
                   historical_contract=contract, historical_model=model,
                   historical_safety=safety or {}, historical_guard_model=guard_model,
                   historical_scale=scale, historical_reservation=reservation)

    @classmethod
    def synthetic(cls, *, device_count=2, local_slots=2, **kwargs):
        """Construct a deliberately small synthetic CPU scenario."""
        return cls(device_count=device_count, local_slots=local_slots, **kwargs)
