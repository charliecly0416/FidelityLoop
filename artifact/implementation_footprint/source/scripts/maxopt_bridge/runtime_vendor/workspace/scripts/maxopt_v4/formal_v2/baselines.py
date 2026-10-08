"""Frozen-parameter W1 baseline adapters with explicit causal windows."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable

from scripts.maxopt_v4.w1_baselines import (
    HPA_Q_TARGETS,
    HPA_STABILIZATIONS,
    MMC_RHO_MAX,
    PRED_ALPHAS,
    PRED_BETAS,
    PRED_HORIZONS,
    PRED_MARGINS,
)

UPDATE_SECONDS = 60
MAX_TARGET = 2


def _clip(value: float) -> int:
    return max(0, min(MAX_TARGET, math.ceil(value)))


@dataclass(frozen=True)
class BaselineDecision:
    baseline_id: str
    target: int
    update_applied: bool
    reason: str
    qbar: float | None = None
    qhat: float | None = None
    lambda_hat: float | None = None
    utilization: float | None = None
    infeasible_capacity: bool = False


class HPA:
    """Queue target with a same-desired-value downscale stabilization clock."""

    baseline_id = "B-HPA"

    def __init__(self, *, q_target: int, stabilization: int) -> None:
        if q_target not in HPA_Q_TARGETS or stabilization not in HPA_STABILIZATIONS:
            raise ValueError("HPA parameters are outside the frozen grid")
        self.q_target = q_target
        self.stabilization = stabilization
        self.target = 0
        self._down_desired: int | None = None
        self._down_seconds = 0

    def snapshot(self) -> dict[str, int | None]:
        return {"target": self.target, "down_desired": self._down_desired,
                "down_seconds": self._down_seconds}

    def decide(self, tick: int, past_queue: Iterable[float]) -> BaselineDecision:
        if tick % UPDATE_SECONDS:
            return BaselineDecision(self.baseline_id, self.target, False, "60s_update_cadence")
        values = tuple(float(value) for value in past_queue)
        qbar = sum(values) / len(values) if values else 0.0
        desired = _clip(qbar / self.q_target)
        if desired > self.target:
            self.target = desired
            self._down_desired, self._down_seconds = None, 0
            reason = "scale_up_immediate"
        elif desired == self.target:
            self._down_desired, self._down_seconds = None, 0
            reason = "target_unchanged"
        else:
            if self._down_desired == desired:
                self._down_seconds += UPDATE_SECONDS
            else:
                # The first matching desired tick counts as the first 60-second interval.
                self._down_desired, self._down_seconds = desired, UPDATE_SECONDS
            if self._down_seconds >= self.stabilization:
                self.target = desired
                self._down_desired, self._down_seconds = None, 0
                reason = "scale_down_after_stabilization"
            else:
                reason = "scale_down_stabilizing"
        return BaselineDecision(self.baseline_id, self.target, True, reason, qbar=qbar)


class Predictor:
    baseline_id = "B-PRED"

    def __init__(self, *, alpha: float, beta: float, horizon: int, margin: int) -> None:
        if (alpha not in PRED_ALPHAS or beta not in PRED_BETAS or horizon not in PRED_HORIZONS
                or margin not in PRED_MARGINS):
            raise ValueError("predictor parameters are outside the frozen grid")
        self.alpha, self.beta, self.horizon, self.margin = alpha, beta, horizon, margin
        self.target, self.level, self.trend = 0, None, 0.0

    def snapshot(self) -> dict[str, float | int | None]:
        return {"target": self.target, "level": self.level, "trend": self.trend}

    def decide(self, tick: int, queue: float) -> BaselineDecision:
        if tick % UPDATE_SECONDS:
            return BaselineDecision(self.baseline_id, self.target, False, "60s_update_cadence")
        if self.level is None:
            self.level, self.trend = float(queue), 0.0
        else:
            old_level = self.level
            self.level = self.alpha * queue + (1.0 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - old_level) + (1.0 - self.beta) * self.trend
        steps = self.horizon // UPDATE_SECONDS
        qhat = max(self.level + step * self.trend for step in range(1, steps + 1))
        self.target = _clip(qhat + self.margin)
        return BaselineDecision(self.baseline_id, self.target, True, "holt_forecast", qhat=qhat)


class MMC:
    baseline_id = "B-MMC"

    def __init__(self, *, rho_max: float) -> None:
        if rho_max not in MMC_RHO_MAX:
            raise ValueError("MMC parameter is outside the frozen grid")
        self.rho_max, self.target = rho_max, 0

    def snapshot(self) -> dict[str, float | int]:
        return {"target": self.target, "rho_max": self.rho_max}

    def decide(self, tick: int, arrival_count: int, service_seconds: float | None) -> BaselineDecision:
        if tick % UPDATE_SECONDS:
            return BaselineDecision(self.baseline_id, self.target, False, "60s_update_cadence")
        rate = arrival_count / UPDATE_SECONDS
        if rate == 0:
            self.target = 0
            return BaselineDecision(self.baseline_id, 0, True, "zero_arrival", lambda_hat=0.0)
        if service_seconds is None or service_seconds <= 0:
            self.target = MAX_TARGET
            return BaselineDecision(self.baseline_id, MAX_TARGET, True, "missing_service_lookup",
                                    lambda_hat=rate, infeasible_capacity=True)
        for target in (1, 2):
            utilization = rate * service_seconds / target
            if utilization < self.rho_max:
                self.target = target
                return BaselineDecision(self.baseline_id, target, True, "minimum_feasible_capacity",
                                        lambda_hat=rate, utilization=utilization)
        utilization = rate * service_seconds / MAX_TARGET
        self.target = MAX_TARGET
        return BaselineDecision(self.baseline_id, MAX_TARGET, True, "insufficient_two_device_capacity",
                                lambda_hat=rate, utilization=utilization, infeasible_capacity=True)


class BaselineAdapter:
    """Supplies W1 controllers only causal, half-open histories.

    ``decide`` consumes the current observation but does not place it in either
    60-second history until its decision has been made. Thus tick zero sees an
    empty history and tick ``t`` sees exactly ``[t-60, t)``.
    """

    def __init__(self, plan: dict[str, Any], rows: list[dict[str, Any]],
                 service_lookup: Callable[[dict[str, Any]], float | None] | None = None) -> None:
        self.plan, self.rows = plan, rows
        params = dict(plan["policy_spec"]["parameters"])
        policy_id = plan["policy_id"]
        constructors = {"B-HPA": HPA, "B-PRED": Predictor, "B-MMC": MMC}
        if policy_id not in constructors:
            raise ValueError("unknown W1 baseline")
        self.controller = constructors[policy_id](**params)
        self.policy_id = policy_id
        self.queue_by_tick: dict[int, float] = {}
        self.arrival_ticks = Counter(int(row["arrival_s"]) for row in rows)
        self.service_lookup = service_lookup

    @property
    def state(self) -> dict[str, Any]:
        return self.controller.snapshot()

    def _past_queue(self, tick: int) -> tuple[float, ...]:
        return tuple(self.queue_by_tick.get(second, 0.0)
                     for second in range(max(0, tick - UPDATE_SECONDS), tick))

    def _past_arrivals(self, tick: int) -> int:
        return sum(self.arrival_ticks[second] for second in range(max(0, tick - UPDATE_SECONDS), tick))

    def _service(self) -> float | None:
        if self.service_lookup is None or not self.rows:
            return None
        return self.service_lookup(self.rows[0])

    def decide(self, observation: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        tick = int(observation["time"])
        queue = float(observation["queue_online"] + observation["queue_offline"])
        before = self.state
        if self.policy_id == "B-HPA":
            decision = self.controller.decide(tick, self._past_queue(tick))
        elif self.policy_id == "B-PRED":
            decision = self.controller.decide(tick, queue)
        else:
            decision = self.controller.decide(tick, self._past_arrivals(tick), self._service())
        # Record after deciding so it belongs to future ticks only.
        self.queue_by_tick[tick] = queue
        return decision.target, {
            "adapter": "w1_baseline_v2",
            "decision": asdict(decision),
            "policy_state_before": before,
            "policy_state_after": self.state,
            "past_window": {"start_inclusive": max(0, tick - UPDATE_SECONDS), "end_exclusive": tick},
        }
