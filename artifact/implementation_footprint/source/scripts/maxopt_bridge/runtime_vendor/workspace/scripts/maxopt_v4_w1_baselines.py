"""CPU-only W1 unguarded baseline contracts for V4.

The module deliberately contains only target-rule state machines.  It does not
import the V3 simulator or the deadline guard, so a W1 replay cannot silently
change the frozen V3 contract.  ``State`` is the small adapter used by the
development harness; a production/mock-schedule adapter may populate it from
the existing one-second observation records.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Mapping, Optional, Sequence


UPDATE_SECONDS = 60
MAX_TARGET = 2
HPA_Q_TARGETS = (4, 8, 16)
HPA_STABILIZATIONS = (120, 300, 600)
PRED_ALPHAS = (0.2, 0.4, 0.6)
PRED_BETAS = (0.1, 0.3, 0.5)
PRED_HORIZONS = (60, 120, 300)
PRED_MARGINS = (0, 1, 2)
MMC_RHO_MAX = (0.70, 0.80, 0.85, 0.90)


@dataclass(frozen=True)
class State:
    """Causal state visible to a W1 controller at one-second resolution.

    ``queue_history`` and ``arrivals_last_60s`` contain only observations up
    to ``elapsed_seconds``.  ``service_seconds`` is the already locked E
    service lookup value used by B-MMC; it is never inferred from future rows.
    """

    elapsed_seconds: int
    queue: float
    queue_history: tuple[float, ...] = ()
    arrivals_last_60s: float = 0.0
    service_seconds: Optional[float] = 1.0
    service_lookup_supported: bool = True
    max_target: int = MAX_TARGET

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.queue < 0 or self.arrivals_last_60s < 0:
            raise ValueError("queue and arrivals must be non-negative")
        if self.max_target != MAX_TARGET:
            raise ValueError("W1 is locked to a two-device target ceiling")
        if self.service_seconds is not None and self.service_seconds <= 0:
            raise ValueError("service_seconds must be positive when present")


@dataclass(frozen=True)
class Decision:
    baseline_id: str
    target: int
    update_applied: bool
    unguarded: bool = True
    infeasible_capacity: bool = False
    reason: str = ""
    qbar: Optional[float] = None
    qhat: Optional[float] = None
    lambda_hat: Optional[float] = None
    utilization: Optional[float] = None

    def __post_init__(self) -> None:
        if self.target < 0 or self.target > MAX_TARGET:
            raise ValueError("target must be clipped to [0, 2]")
        if not self.unguarded:
            raise ValueError("W1 contracts must remain unguarded")


def _clip_target(value: float) -> int:
    return max(0, min(MAX_TARGET, int(math.ceil(value))))


def _is_update(state: State) -> bool:
    return state.elapsed_seconds % UPDATE_SECONDS == 0


class HPAController:
    """B-HPA: HPA-inspired reactive queue target with down stabilization."""

    baseline_id = "B-HPA"

    def __init__(
        self,
        q_target: int = 8,
        stabilization_seconds: int | None = None,
        *,
        stabilization: int | None = None,
    ) -> None:
        # ``stabilization`` is the preregistered manifest key; the longer
        # spelling remains accepted for callers adapting existing code.
        if stabilization_seconds is not None and stabilization is not None:
            raise ValueError("pass only one stabilization spelling")
        if stabilization_seconds is None:
            stabilization_seconds = 300 if stabilization is None else int(stabilization)
        if q_target not in HPA_Q_TARGETS:
            raise ValueError(f"q_target must be one of {HPA_Q_TARGETS}")
        if stabilization_seconds not in HPA_STABILIZATIONS:
            raise ValueError(f"stabilization must be one of {HPA_STABILIZATIONS}")
        self.q_target = int(q_target)
        self.stabilization_seconds = int(stabilization_seconds)
        self.target = 0
        self._down_since: Optional[int] = None

    def decide(self, state: State) -> Decision:
        if not _is_update(state):
            return Decision(self.baseline_id, self.target, False, reason="60s_update_cadence")
        history = state.queue_history[-UPDATE_SECONDS:] or (state.queue,)
        qbar = sum(history) / len(history)
        desired = _clip_target(qbar / self.q_target)
        if desired > self.target:
            self.target = desired
            self._down_since = None
            reason = "scale_up_immediate"
        elif desired < self.target:
            if self._down_since is None:
                self._down_since = state.elapsed_seconds
            if state.elapsed_seconds - self._down_since >= self.stabilization_seconds:
                self.target = desired
                self._down_since = None
                reason = "scale_down_after_stabilization"
            else:
                reason = "scale_down_stabilizing"
        else:
            self._down_since = None
            reason = "target_unchanged"
        return Decision(self.baseline_id, self.target, True, reason=reason, qbar=qbar)


class PredictiveController:
    """B-PRED: causal Holt queue forecast over the locked 81-cell grid."""

    baseline_id = "B-PRED"

    def __init__(self, alpha: float = 0.4, beta: float = 0.3, horizon: int = 120, margin: int = 1) -> None:
        if alpha not in PRED_ALPHAS or beta not in PRED_BETAS:
            raise ValueError("alpha/beta are outside the registered grid")
        if horizon not in PRED_HORIZONS or margin not in PRED_MARGINS:
            raise ValueError("horizon/margin are outside the registered grid")
        self.alpha, self.beta = float(alpha), float(beta)
        self.horizon, self.margin = int(horizon), int(margin)
        self.target = 0
        self.level: Optional[float] = None
        self.trend = 0.0

    def decide(self, state: State) -> Decision:
        if not _is_update(state):
            return Decision(self.baseline_id, self.target, False, reason="60s_update_cadence")
        q = float(state.queue)
        if self.level is None:
            self.level, self.trend = q, 0.0
        else:
            previous_level = self.level
            self.level = self.alpha * q + (1.0 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - previous_level) + (1.0 - self.beta) * self.trend
        steps = max(1, self.horizon // UPDATE_SECONDS)
        qhat = max(self.level + step * self.trend for step in range(1, steps + 1))
        self.target = _clip_target(qhat + self.margin)
        return Decision(self.baseline_id, self.target, True, reason="holt_forecast", qhat=qhat)


class MMCController:
    """B-MMC: causal marginal capacity rule with explicit zero/insufficient cases."""

    baseline_id = "B-MMC"

    def __init__(self, rho_max: float = 0.80) -> None:
        if rho_max not in MMC_RHO_MAX:
            raise ValueError(f"rho_max must be one of {MMC_RHO_MAX}")
        self.rho_max = float(rho_max)
        self.target = 0

    def decide(self, state: State) -> Decision:
        if not _is_update(state):
            return Decision(self.baseline_id, self.target, False, reason="60s_update_cadence")
        # arrivals_last_60s / 60 is requests per second.  A zero rate is a
        # first-class state and therefore never reaches the division below.
        lambda_hat = float(state.arrivals_last_60s) / UPDATE_SECONDS
        if lambda_hat == 0:
            self.target = 0
            return Decision(self.baseline_id, 0, True, reason="zero_arrival", lambda_hat=0.0)
        if state.service_seconds is None or not state.service_lookup_supported:
            self.target = MAX_TARGET
            return Decision(self.baseline_id, MAX_TARGET, True, infeasible_capacity=True,
                            reason=("missing_service_lookup" if state.service_seconds is None
                                    else "unsupported_service_lookup"), lambda_hat=lambda_hat)
        for c in (1, 2):
            rho = lambda_hat * float(state.service_seconds) / c
            if rho < self.rho_max:
                self.target = c
                return Decision(self.baseline_id, c, True, reason="minimum_feasible_capacity",
                                lambda_hat=lambda_hat, utilization=rho)
        # The contract intentionally does not return c=3: the two-device
        # ceiling is a hard boundary and must be surfaced as a diagnostic.
        rho = lambda_hat * float(state.service_seconds) / MAX_TARGET
        self.target = MAX_TARGET
        return Decision(self.baseline_id, MAX_TARGET, True, infeasible_capacity=True,
                        reason="insufficient_two_device_capacity", lambda_hat=lambda_hat,
                        utilization=rho)


def pred_grid() -> tuple[dict[str, object], ...]:
    """Return the complete deterministic 3×3×3×3 = 81-cell grid."""
    return tuple(
        {"alpha": a, "beta": b, "horizon": h, "margin": m}
        for a, b, h, m in itertools.product(PRED_ALPHAS, PRED_BETAS, PRED_HORIZONS, PRED_MARGINS)
    )


def all_parameter_cells() -> dict[str, tuple[dict[str, object], ...]]:
    return {
        "B-HPA": tuple({"q_target": q, "stabilization": s} for q, s in itertools.product(HPA_Q_TARGETS, HPA_STABILIZATIONS)),
        "B-PRED": pred_grid(),
        "B-MMC": tuple({"rho_max": rho} for rho in MMC_RHO_MAX),
    }


def validate_contracts() -> dict[str, object]:
    cells = all_parameter_cells()
    if len(cells["B-PRED"]) != 81 or len({tuple(sorted(c.items())) for c in cells["B-PRED"]}) != 81:
        raise AssertionError("B-PRED must contain 81 unique cells")
    zero = MMCController().decide(State(60, queue=0, arrivals_last_60s=0))
    boundary = MMCController().decide(State(60, queue=0, arrivals_last_60s=60, service_seconds=10.0))
    if zero.target != 0 or boundary.target != 2 or not boundary.infeasible_capacity:
        raise AssertionError("B-MMC zero-arrival/capacity boundary contract failed")
    return {"baseline_ids": tuple(cells), "cell_counts": {k: len(v) for k, v in cells.items()}, "unguarded": True}


def decision_dict(decision: Decision) -> dict[str, object]:
    return asdict(decision)
