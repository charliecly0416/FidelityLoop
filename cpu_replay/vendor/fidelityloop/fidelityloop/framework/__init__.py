"""Bounded, CPU-only facade for the frozen MaxOpt lifecycle semantics.

The facade deliberately keeps the historical V3 runtime as the default path and
provides a small generic event loop for synthetic finite-device examples.
"""
from .config import FrameworkConfig
from .estimators import HistoricalServiceEstimator, ConstantServiceEstimator
from .policies import (
    TargetPolicyAdapter, GuardPolicyAdapter, PPOActionAdapter,
)
from .runtime import FrameworkSimulator, run_historical

__all__ = [
    "FrameworkConfig", "HistoricalServiceEstimator", "ConstantServiceEstimator",
    "TargetPolicyAdapter", "GuardPolicyAdapter", "PPOActionAdapter",
    "FrameworkSimulator", "run_historical",
]
