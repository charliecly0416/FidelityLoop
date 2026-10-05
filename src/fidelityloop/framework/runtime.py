"""Bounded CPU runtime facade.

Historical two-device runs are delegated to the frozen V3 simulator.  The small
fallback loop exists for synthetic finite-device/slot demonstrations only.
"""
from __future__ import annotations
from collections import deque
import copy
import math
from typing import Mapping
from fidelityloop.legacy.maxopt_v2.workload import FIELDS
from .config import FrameworkConfig
from .estimators import HistoricalServiceEstimator, ConstantServiceEstimator
from .policies import TargetPolicyAdapter, GuardPolicyAdapter


def run_historical(rows, contract, model, policy, *, safety, guard_model=None, scale=False, reservation=False):
    """Run the frozen V3 path, preserving its event/request/ledger semantics exactly."""
    from fidelityloop.legacy.maxopt_v3.simulator import Simulator
    policy = getattr(policy, "spec", policy)
    return Simulator(rows, contract, model, policy, safety=safety,
                     guard_model=guard_model, scale=scale, reservation=reservation).run()


class FrameworkSimulator:
    """CPU-only bounded facade with injectable estimator and policy.

    ``FrameworkSimulator.historical`` is the default compatibility route.  A
    directly constructed instance runs the intentionally small synthetic loop.
    """
    def __init__(self, rows, config: FrameworkConfig, *, estimator=None, policy=None):
        self.rows = [dict(r) for r in rows]
        self.config = config
        self.estimator = estimator or ConstantServiceEstimator(1.0, config.local_slots)
        self.policy = policy or TargetPolicyAdapter("all1", max_devices=config.device_count)
        self._validate_rows()
        self.events = []

    @classmethod
    def historical(cls, rows, contract, model, policy, *, safety, guard_model=None,
                   scale=False, reservation=False):
        config = FrameworkConfig.from_historical(contract, model, safety=safety,
                                                 guard_model=guard_model, scale=scale,
                                                 reservation=reservation)
        obj = cls.__new__(cls)
        obj.rows, obj.config, obj.estimator, obj.policy = [dict(r) for r in rows], config, HistoricalServiceEstimator(model), policy
        obj._validate_rows(); obj.events = []
        return obj

    def _validate_rows(self):
        seen, previous = set(), -1
        for row in self.rows:
            if set(row) != FIELDS: raise ValueError("unknown/missing workload fields")
            if row["request_id"] in seen: raise ValueError("duplicate request ID")
            seen.add(row["request_id"])
            if type(row["arrival_s"]) is not int or row["arrival_s"] < previous:
                raise ValueError("rows must be ordered by arrival")
            if not 0 <= row["arrival_s"] < self.config.window_seconds:
                raise ValueError("arrival outside half-open window")
            if row["sim_tick"] != row["arrival_s"] or row["wall_monotonic_s"] is not None:
                raise ValueError("invalid simulation clock")
            if row["split"] not in ("train", "validation"):
                raise ValueError("locked-test access is not authorized")
            if row["job_type"] not in ("online", "offline") or row["deadline_s"] < row["arrival_s"]:
                raise ValueError("invalid request class/deadline")
            if len(row["prompt_token_ids"]) != row["input_tokens"]: raise ValueError("token count mismatch")
            previous = row["arrival_s"]

    def run(self):
        if self.config.historical_contract is not None:
            policy = getattr(self.policy, "spec", self.policy)
            result = run_historical(self.rows, self.config.historical_contract,
                                    self.config.historical_model, policy,
                                    safety=self.config.historical_safety,
                                    guard_model=self.config.historical_guard_model,
                                    scale=self.config.historical_scale,
                                    reservation=self.config.historical_reservation)
            self.events = copy.deepcopy(result["events"])
            return result
        return self._run_bounded()

    def _run_bounded(self):
        cfg = self.config
        now, cursor = 0, 0
        end = cfg.window_seconds + cfg.drain_seconds
        devices = [{"id": i, "state": "active" if i < cfg.initial_active_devices else "off",
                     "due": None, "jobs": []} for i in range(cfg.device_count)]
        jobs = {r["request_id"]: dict(r, status="unreleased", started_at=None,
                                      completed_at=None, device_id=None, remaining=None)
                for r in self.rows}
        queues = {"online": deque(), "offline": deque()}
        api = {}
        events = []
        counters = {"arrived": 0, "completed": 0, "censored": 0, "local_completed": 0,
                    "synthetic_api_accepted": 0, "synthetic_api_completed": 0,
                    "offline_deadline_misses": 0, "startup_events": 0, "shutdown_events": 0,
                    "gpu_occupied_seconds": 0, "gpu_active_seconds": 0, "queue_seconds": 0}
        costs = {k: 0.0 for k in ("gpu", "startup", "shutdown", "synthetic_api", "offline_miss")}

        def emit(kind, **fields): events.append(dict(seq=len(events), time=now, kind=kind, **fields))
        def state(d, new, due=None):
            old = d["state"]; d.update(state=new, due=due); emit("state", device_id=d["id"], previous=old, state=new, due=due)
        def active_count(): return sum(d["state"] in ("active", "starting") for d in devices)
        def actuate(target):
            target = max(0, min(cfg.device_count, int(target)))
            for d in reversed(devices):
                if active_count() > target and d["state"] == "active":
                    state(d, "draining")
                    if not d["jobs"]: counters["shutdown_events"] += 1; costs["shutdown"] += cfg.shutdown_event_price; state(d, "stopping", now + cfg.shutdown_seconds)
            for d in devices:
                if active_count() < target and d["state"] == "draining": state(d, "active")
            for d in devices:
                if active_count() < target and d["state"] == "off":
                    counters["startup_events"] += 1; costs["startup"] += cfg.startup_event_price
                    state(d, "starting", now + cfg.startup_seconds)
                    if cfg.startup_seconds == 0: state(d, "active")

        def public_job(rid):
            j = jobs[rid]
            return {k: j[k] for k in ("request_id", "job_type", "arrival_s", "deadline_s",
                                      "input_tokens", "max_output_tokens", "started_at")}
        def observation(t):
            return {"schema": "bounded-framework-observation-v1", "time": t,
                    "queue_online": [public_job(r) for r in queues["online"]],
                    "queue_offline": [public_job(r) for r in queues["offline"]],
                    "devices": [{"state": d["state"], "since": 0.0,
                                 "jobs": [public_job(r) for r in d["jobs"]]} for d in devices]}
        def dispatch_kind(obs, reserve):
            method = getattr(self.policy, "dispatch_kind", None)
            if method is None: return "online" if obs["queue_online"] else ("offline" if obs["queue_offline"] else None)
            return method(obs, reserve)
        def admit(obs, reserve):
            while True:
                available = [d for d in devices if d["state"] == "active" and len(d["jobs"]) < cfg.local_slots]
                if not available: break
                kind = dispatch_kind(obs, reserve)
                if kind is None: break
                queue = queues[kind]
                if not queue: break
                rid = queue.popleft(); d = min(available, key=lambda x: (len(x["jobs"]), x["id"]))
                j = jobs[rid]; j.update(status="local", started_at=now, device_id=d["id"], remaining=1.0); d["jobs"].append(rid)
                emit("dispatch", request_id=rid, route=d["id"])

        for now in range(end):
            for d in devices:
                if d["state"] == "starting" and d["due"] <= now: state(d, "active")
                elif d["state"] == "stopping" and d["due"] <= now: state(d, "off")
            while cursor < len(self.rows) and self.rows[cursor]["arrival_s"] <= now:
                row = self.rows[cursor]; cursor += 1; rid = row["request_id"]; jobs[rid]["status"] = "waiting"
                queues[row["job_type"]].append(rid); counters["arrived"] += 1; emit("release", request_id=rid)
            obs = observation(now)
            decision = self.policy.decide(obs)
            target = decision["target"] if isinstance(decision, Mapping) else int(decision)
            emit("decision", target=target, observation=obs, guard=decision)
            actuate(target)
            admit(observation(now), bool(decision.get("reserve", False)) if isinstance(decision, Mapping) else False)
            while queues["online"] and len(api) < cfg.api_capacity:
                rid = queues["online"][0]; row = jobs[rid]
                if now - row["arrival_s"] < cfg.api_route_wait_seconds: break
                queues["online"].popleft(); row.update(status="api", started_at=now); api[rid] = now + cfg.api_latency_seconds
                counters["synthetic_api_accepted"] += 1; costs["synthetic_api"] += row["input_tokens"] * cfg.api_input_price + row["max_output_tokens"] * cfg.api_output_price
                emit("dispatch", request_id=rid, route="synthetic_api")
            # one-second progress and accounting
            for d in devices:
                if d["state"] != "off":
                    counters["gpu_occupied_seconds"] += 1; counters["gpu_active_seconds"] += d["state"] == "active"; costs["gpu"] += cfg.gpu_second_price
                counters["queue_seconds"] += len(queues["online"]) + len(queues["offline"])
                if d["jobs"]:
                    n = len(d["jobs"])
                    for rid in list(d["jobs"]):
                        jobs[rid]["remaining"] -= 1.0 / self.estimator.seconds(jobs[rid], n)
                        if jobs[rid]["remaining"] <= 1e-12:
                            d["jobs"].remove(rid); jobs[rid].update(status="completed", completed_at=now + 1); counters["local_completed"] += 1; counters["completed"] += 1; emit("complete", request_id=rid, route=d["id"])
            now += 1
            for rid, due in list(api.items()):
                if due <= now:
                    del api[rid]; jobs[rid].update(status="completed", completed_at=now); counters["synthetic_api_completed"] += 1; counters["completed"] += 1; emit("complete", request_id=rid, route="synthetic_api")
            for d in devices:
                if d["state"] == "draining" and not d["jobs"]:
                    counters["shutdown_events"] += 1; costs["shutdown"] += cfg.shutdown_event_price; state(d, "stopping", now + cfg.shutdown_seconds)
        for rid, j in jobs.items():
            if j["status"] not in ("completed",):
                if j["job_type"] == "offline": counters["offline_deadline_misses"] += 1; costs["offline_miss"] += cfg.offline_miss_price
                j["status"] = "censored"; counters["censored"] += 1; emit("censor", request_id=rid)
        queues["online"].clear(); queues["offline"].clear(); api.clear()
        for d in devices:
            if d["state"] != "off": counters["shutdown_events"] += 1; costs["shutdown"] += cfg.shutdown_event_price; state(d, "off")
        records = {rid: {"dispatched_at": j["started_at"], "completed_at": j["completed_at"],
                         "route": j["device_id"] if j["device_id"] is not None else ("synthetic_api" if j["status"] == "completed" else None)} for rid, j in jobs.items()}
        for kind in ("online", "offline"): counters[kind + "_arrived"] = sum(r["job_type"] == kind for r in self.rows)
        ontime = {k: sum(j["completed_at"] is not None and j["completed_at"] <= (r["arrival_s"] + cfg.online_sla_seconds if k == "online" else r["deadline_s"]) for r, j in [(r, jobs[r["request_id"]]) for r in self.rows if r["job_type"] == k]) for k in ("online", "offline")}
        total_cost = sum(costs.values())
        expected_cost = (counters["gpu_occupied_seconds"] * cfg.gpu_second_price
                         + counters["startup_events"] * cfg.startup_event_price
                         + counters["shutdown_events"] * cfg.shutdown_event_price
                         + costs["synthetic_api"]
                         + counters["offline_deadline_misses"] * cfg.offline_miss_price)
        state_ok = all(j["status"] in ("completed", "censored") for j in jobs.values()) and not any(d["jobs"] for d in devices) and not api
        summary = {**counters, "costs": {**costs, "operating": total_cost - costs["offline_miss"], "total": total_cost},
                   "total_cost": total_cost, "pending": 0, "empty_workload": not self.rows,
                   "online_sla_met": ontime["online"], "offline_deadline_met": ontime["offline"],
                   "offline_deadline_completion_rate": ontime["offline"] / counters["offline_arrived"] if counters["offline_arrived"] else None,
                   "online_violation_rate": 1 - ontime["online"] / counters["online_arrived"] if counters["online_arrived"] else None,
                   "elapsed_seconds": end, "finished": True, "state_conservation": state_ok,
                   "request_conservation": counters["arrived"] == counters["completed"] + counters["censored"],
                   "cost_conservation": math.isclose(total_cost, expected_cost, abs_tol=1e-12)}
        self.events = events
        return {"summary": summary, "requests": records, "events": events}
