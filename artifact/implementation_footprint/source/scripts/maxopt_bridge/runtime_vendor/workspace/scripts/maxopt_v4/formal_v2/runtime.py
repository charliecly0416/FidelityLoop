"""Physical formal_v2 controller with V3 policy semantics.

This module deliberately reuses the accepted lifecycle and worker ownership
machinery.  It supplies only the plan-neutral policy adapters, causal W1
histories, and V3 guard admission behavior that the old all2-only controller
cannot express.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
from pathlib import Path
import time
from typing import Any

from scripts.maxopt_v2.calibrated import service_seconds
from scripts.maxopt_v2.engine import TargetPolicy
from scripts.maxopt_v3.policy import DeadlinePolicy, SCHEMA as OBSERVATION_SCHEMA, public_job
from scripts.maxopt_v4.formal_v2.backend import AsyncWorkerBackend, expected_effective_runtime
from scripts.maxopt_v4.formal_v2.baselines import BaselineAdapter
from scripts.maxopt_v4.formal_v2.worker import validate_runtime
from scripts.maxopt_v4.w2_gpu_all2 import gpu_runner as gate


W1_POLICIES = {"B-HPA", "B-PRED", "B-MMC"}
GUARD_POLICY = "hysteresis_u8_d0_c60__guard"


class TargetAdapter:
    """Expose the frozen V2 target rule through the common adapter surface."""

    def __init__(self, policy_spec: dict[str, Any]) -> None:
        self.policy = TargetPolicy(policy_spec)
        self.policy.state["target"] = 2

    @property
    def state(self) -> dict[str, Any]:
        return self.policy.state

    def decide(self, observation: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        before = dict(self.policy.state)
        target = self.policy.decide(observation)
        return int(target), {
            "adapter": "v3_target_policy",
            "policy_state_before": before,
            "policy_state_after": dict(self.policy.state),
        }


class GuardAdapter:
    """Use the frozen V3 guard directly, including its per-slot selector."""

    def __init__(self, plan: dict[str, Any]) -> None:
        self.policy = DeadlinePolicy(
            plan["policy_spec"]["base_policy"], plan["service_model"], plan["guard_safety"],
            scale=True, reservation=True, margin=5.0,
        )

    @property
    def state(self) -> dict[str, Any]:
        return self.policy.state

    def decide(self, observation: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        decision = self.policy.decide(observation)
        return int(decision["target"]), {
            "adapter": "v3_deadline_guard",
            "decision": decision,
            "policy_state_before": decision["policy_state_before"],
            "policy_state_after": decision["policy_state_after"],
        }

    def dispatch_kind(self, observation: dict[str, Any], reserve: bool) -> str | None:
        return self.policy.dispatch_kind(observation, reserve)


def _service_lookup(plan: dict[str, Any]):
    """Bind MMC to its own frozen service model, never to another package."""
    model = plan.get("service_model")

    def lookup(row: dict[str, Any]) -> float | None:
        if not isinstance(model, dict):
            return None
        try:
            return service_seconds(model, row, 1)
        except (KeyError, TypeError, ValueError):
            # The controller records the missing lookup as an explicit MMC
            # infeasibility decision.  Substitution with another model is
            # prohibited by the formal contract.
            return None

    return lookup


def _validate_runtime_plan(plan: dict[str, Any]) -> None:
    runtime = validate_runtime(plan.get("runtime"))
    expected = plan.get("effective_runtime_expected")
    if expected != expected_effective_runtime(runtime):
        raise ValueError("effective runtime expectation differs from frozen runtime")
    horizon = plan.get("horizon_seconds")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("invalid observation horizon")
    if plan.get("formal") is True:
        if horizon != gate.HORIZON:
            raise ValueError("formal cells require the complete 2100-second observation")
    else:
        smoke = plan.get("smoke")
        if (not isinstance(smoke, dict) or smoke.get("nonformal") is not True
                or smoke.get("clock_scale") != 1 or not 20 <= horizon <= 600):
            raise ValueError("short observation requires an explicit uncompressed smoke contract")


def _adapter(plan: dict[str, Any], rows: list[dict[str, Any]]):
    policy_id = plan.get("policy_id")
    if policy_id in W1_POLICIES:
        return BaselineAdapter(plan, rows, _service_lookup(plan))
    if policy_id == GUARD_POLICY:
        return GuardAdapter(plan)
    return TargetAdapter(plan["policy_spec"])


class FormalV2Controller(gate.Controller):
    """One-second physical controller for all V4 work packages.

    The inherited controller owns raw ledgers, worker callbacks, setup and
    cleanup.  This subclass changes only decisions and admission semantics.
    """

    def __init__(self, plan: dict[str, Any], out: str | Path, *, backend_factory=AsyncWorkerBackend) -> None:
        _validate_runtime_plan(plan)
        rows = json.loads((Path(out) / "expected_requests.json").read_text(encoding="utf-8"))
        self.adapter = _adapter(plan, rows)
        self._device_since = [0.0, 0.0]
        self._current_tick = 0
        self._reserve = False
        super().__init__(plan, out, backend_factory=backend_factory, policy=self.adapter, formal=False)

    def emit(self, kind: str, *, observed_ns: int | None = None, **fields: Any):
        event = super().emit(kind, observed_ns=observed_ns, **fields)
        if kind == "lifecycle_state" and self.origin_ns is not None:
            gpu = fields["gpu"]
            self._device_since[gpu] = max(0.0, (event["controller_monotonic_ns"] - self.origin_ns) / 1e9)
        return event

    def start_window(self, seconds: int | None = None) -> None:
        horizon = self.plan["horizon_seconds"] if seconds is None else seconds
        if horizon != self.plan["horizon_seconds"]:
            raise ValueError("runtime/start_window horizon mismatch")
        super().start_window(horizon)
        # Both setup-ready instances form the observable time-zero state.
        self._device_since = [0.0, 0.0]

    def _simple_observation(self, tick: int) -> dict[str, int]:
        return self.actuator.observation(tick, len(self.queues["online"]), len(self.queues["offline"]))

    def _rich_observation(self, tick: int) -> dict[str, Any]:
        queues = {
            kind: [public_job(self.by_id[rid]) for rid in self.queues[kind]]
            for kind in ("online", "offline")
        }
        devices = []
        for gpu, device in enumerate(self.actuator.devices):
            jobs = []
            for rid in sorted(device["jobs"]):
                dispatched = self.local.get(rid, {}).get("dispatched_at")
                if dispatched is None:
                    raise ValueError("active local request lacks immutable dispatch timestamp")
                jobs.append(public_job(self.by_id[rid], dispatched))
            devices.append({"state": device["state"], "since": self._device_since[gpu], "jobs": jobs})
        return {"schema": OBSERVATION_SCHEMA, "time": tick,
                "queue_online": queues["online"], "queue_offline": queues["offline"], "devices": devices}

    def _dispatch_one(self, kind: str) -> bool:
        if kind not in self.queues or not self.queues[kind] or not self.actuator.available():
            return False
        if time.monotonic_ns() >= self.cutoff_ns:
            return False
        gpu = min(self.actuator.available(), key=lambda index: (len(self.actuator.devices[index]["jobs"]), index))
        rid = self.queues[kind].popleft()
        row = self.by_id[rid]
        identity = self.actuator.dispatch(gpu, rid)
        # Policy observations expose the actual controller dispatch instant,
        # not the scheduled integer tick.  A one-second tick may run late and
        # guard ETA calculations must retain that observable elapsed time.
        dispatch_ns = time.monotonic_ns()
        identity["dispatched_at"] = (dispatch_ns - self.origin_ns) / 1e9
        self.local[rid] = dict(identity)
        raw = self.emit("local_dispatch", request_id=rid, input_tokens=row["input_tokens"],
                        output_budget=row["max_output_tokens"], observed_ns=dispatch_ns, **identity)
        self.record("dispatch", rid, raw, route="gpu" + str(gpu), **identity)
        self.backend.submit(gpu, identity["generation"], row)
        return True

    def dispatch_local(self, kind: str) -> None:
        while self._dispatch_one(kind):
            pass

    def _admit_api(self) -> None:
        queue = self.queues["online"]
        while queue and len(self.api_active) < 8:
            now_ns = time.monotonic_ns()
            age = (now_ns - self.origin_ns) / 1e9 - self.by_id[queue[0]]["arrival_s"]
            if now_ns >= self.cutoff_ns or age < 10:
                break
            self.accept_api(queue.popleft())

    def admit(self) -> None:
        if not isinstance(self.adapter, GuardAdapter):
            super().admit()
            return
        # A V3 guard decides every local slot from a fresh observable state.
        while self.actuator.available():
            kind = self.adapter.dispatch_kind(self._rich_observation(self._current_tick), self._reserve)
            if kind is None or not self._dispatch_one(kind):
                break
        self._admit_api()

    def tick(self, tick: int, *, target_override: int | None = None) -> None:
        if target_override is not None:
            raise ValueError("scripted targets are forbidden in formal_v2")
        self._current_tick = tick
        self.check_health()
        if self.window_closed or time.monotonic_ns() >= self.cutoff_ns:
            raise ValueError("no admission after observation cutoff")
        while self.cursor < len(self.expected) and self.expected[self.cursor]["arrival_s"] <= tick:
            row = self.expected[self.cursor]
            rid = row["request_id"]
            self.queues[row["job_type"]].append(rid)
            self.released.add(rid)
            raw = self.emit("request_release", request_id=rid, scheduled_arrival_s=row["arrival_s"])
            self.record("release", rid, raw)
            self.cursor += 1
        simple = self._simple_observation(tick)
        before = dict(self.adapter.state)
        if isinstance(self.adapter, GuardAdapter):
            target, metadata = self.adapter.decide(self._rich_observation(tick))
            self._reserve = bool(metadata["decision"]["reserve"])
        else:
            target, metadata = self.adapter.decide(simple)
            self._reserve = False
        after = dict(self.adapter.state)
        self.emit("policy_tick", tick=tick, observation=simple, target=target,
                  policy_state_before=before, policy_state_after=after,
                  policy_metadata=metadata, guard_reservation=self._reserve, scripted=False)
        self.actuator.apply(target)
        self.admit()
        self.emit("admission_complete", tick=tick)

    async def observe(self) -> None:
        horizon = self.plan["horizon_seconds"]
        for tick in range(horizon):
            due_ns = self.origin_ns + tick * 10**9
            while time.monotonic_ns() < due_ns:
                await asyncio.sleep(min(0.01, (due_ns - time.monotonic_ns()) / 1e9))
            if time.monotonic_ns() >= due_ns + 10**9:
                raise RuntimeError("one-second policy tick missed")
            self.tick(tick)
        while time.monotonic_ns() < self.cutoff_ns:
            self.check_health()
            await asyncio.sleep(min(0.01, (self.cutoff_ns - time.monotonic_ns()) / 1e9))
        self.close_window()


async def run(plan: dict[str, Any], out: str | Path, *, backend_factory=AsyncWorkerBackend) -> dict[str, Any]:
    """Run one cell and classify its terminal state without campaign side effects."""
    controller = FormalV2Controller(plan, out, backend_factory=backend_factory)
    result = await gate.supervised_run(controller)
    technical = bool(result.get("technical_valid"))
    result.update({
        "schema": "maxopt-v4-formal-v2-runtime-result-v1",
        "work_package": plan["work_package"], "policy_id": plan["policy_id"],
        "window_id": plan["window_id"], "repeat": plan["repeat"],
        "model_id": plan["model_id"], "model_path": plan["model_path"],
        "effective_runtime_expected": plan["effective_runtime_expected"],
        "classification": ("TECHNICAL_INVALID" if not technical else
                           "PASS" if result.get("gate", {}).get("pass") else "NEGATIVE_RESULT"),
    })
    return result
