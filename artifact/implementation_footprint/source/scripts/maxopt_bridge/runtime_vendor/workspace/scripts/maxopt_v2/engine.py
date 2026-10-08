"""Discrete-time, uncalibrated V2 lifecycle simulator with explicit completions.

No filesystem writes, model inference, network calls, or learned parameters.
The durable runtime owns events and snapshots; this module owns tick semantics.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .workload import FIELDS

OBSERVATION_FIELDS = frozenset({"time", "queue_online", "queue_offline", "local_running",
                               "active", "starting", "draining", "off"})
MODEL_DEFAULTS = {"identity": "uncalibrated_equal_share_pilot_v1", "prefill_tps": 1024.0,
                  "decode_tps": 128.0, "local_slots": 4, "startup_seconds": 30,
                  "max_devices": 2}
COST_KEYS = ("gpu", "startup", "shutdown", "synthetic_api", "offline_miss")
COUNT_KEYS = ("arrived", "online_arrived", "offline_arrived", "local_completed",
              "synthetic_api_accepted", "synthetic_api_completed", "censored",
              "online_sla_met", "offline_deadline_met", "offline_deadline_misses",
              "gpu_occupied_seconds", "gpu_active_seconds", "gpu_starting_seconds",
              "gpu_draining_seconds", "gpu_idle_seconds", "startup_events", "shutdown_events")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def normalize_config(config):
    config = copy.deepcopy(config)
    integer(config["window_seconds"], "window_seconds", 1)
    integer(config["drain_seconds"], "drain_seconds")
    config["model"] = {**MODEL_DEFAULTS, **config.get("model", {})}
    model = config["model"]
    if set(model) != set(MODEL_DEFAULTS):
        raise ValueError("unknown service model fields")
    if model["max_devices"] != 2:
        raise ValueError("V2 contract requires two identical independent TP=1 instances")
    for field in ("prefill_tps", "decode_tps"):
        if type(model[field]) not in (float, int) or not math.isfinite(model[field]) or model[field] <= 0:
            raise ValueError(f"invalid {field}")
    integer(model["local_slots"], "local_slots", 1)
    integer(model["startup_seconds"], "startup_seconds")
    prices = config["accounting"]
    for field in ("gpu_second", "startup_event", "shutdown_event", "api_input_token", "api_output_token", "offline_deadline_miss"):
        if type(prices[field]) not in (int, float) or not math.isfinite(prices[field]) or prices[field] < 0:
            raise ValueError(f"invalid price {field}")
    sink = config["synthetic_api"]
    integer(sink["capacity"], "API capacity")
    integer(sink["latency_seconds"], "API latency", 1)
    integer(sink["route_online_after_wait_seconds"], "API route delay")
    if sink["offline_route"] is not False or sink["cost_on_accept"] is not True or sink["completion_requires_elapsed_delay"] is not True or sink["virtual_settlement_counts_as_completion"] is not False:
        raise ValueError("unsupported synthetic API semantics")
    sla = config["feasibility"]["online_e2e_sla_seconds"]
    if type(sla) not in (int, float) or not math.isfinite(sla) or sla <= 0:
        raise ValueError("invalid online SLA")
    return config


class TargetPolicy:
    """Policy controls target count only; admission/accounting are shared."""

    def __init__(self, spec):
        spec = {"name": spec} if isinstance(spec, str) else dict(spec)
        name = spec.get("name")
        defaults = {"all1": {}, "all2": {},
                    "hpa": {"target_pressure": 4, "cooldown_seconds": 30, "min_devices": 0},
                    "hysteresis": {"up_threshold": 6, "down_threshold": 1,
                                   "cooldown_seconds": 30, "min_devices": 0}}
        if name not in defaults or set(spec) - {"name", *defaults[name]}:
            raise ValueError("unknown policy or policy fields")
        self.spec = {"name": name, **defaults[name], **spec}
        for key, value in self.spec.items():
            if key != "name":
                integer(value, key, 1 if key in {"target_pressure", "up_threshold"} else 0)
        if self.spec.get("min_devices", 0) > 2:
            raise ValueError("min_devices > max_devices")
        if name == "hysteresis" and self.spec["down_threshold"] >= self.spec["up_threshold"]:
            raise ValueError("hysteresis thresholds must be separated")
        self.state = {"target": 1, "last_change_at": -self.spec.get("cooldown_seconds", 0)}

    def decide(self, observation):
        if set(observation) != OBSERVATION_FIELDS:
            raise ValueError("unknown/missing observation fields; future outcomes forbidden")
        for key, value in observation.items():
            integer(value, key)
        name = self.spec["name"]
        if name in {"all1", "all2"}:
            self.state["target"] = int(name[-1])
            return self.state["target"]
        now, old = observation["time"], self.state["target"]
        if now - self.state["last_change_at"] < self.spec["cooldown_seconds"]:
            return old
        pressure = observation["queue_online"] + observation["queue_offline"] + observation["local_running"]
        if name == "hpa":
            target = math.ceil(pressure / self.spec["target_pressure"])
        elif pressure > self.spec["up_threshold"] * max(1, old):
            target = old + 1
        elif pressure <= self.spec["down_threshold"] * max(1, old - 1):
            target = old - 1
        else:
            target = old
        target = min(2, max(self.spec["min_devices"], target))
        if target != old:
            self.state = {"target": target, "last_change_at": now}
        return target


class Simulator:
    def __init__(self, rows, config, policy="all1", emit=None, *,
                 mode="independent_closed_loop", target_sequence=None):
        self.config = normalize_config(config)
        self.policy = TargetPolicy(policy)
        self.emit_callback = emit
        self.mode = mode
        self.end = self.config["window_seconds"] + self.config["drain_seconds"]
        if mode not in {"independent_closed_loop", "conditional_action_replay_diagnostic"}:
            raise ValueError("unknown simulation mode")
        if mode == "independent_closed_loop" and target_sequence is not None:
            raise ValueError("external targets are forbidden in independent closed-loop prediction")
        if mode == "conditional_action_replay_diagnostic":
            if target_sequence is None or len(target_sequence) != self.end:
                raise ValueError("conditional replay needs exactly one target per tick")
            for target in target_sequence:
                if integer(target, "target") > 2:
                    raise ValueError("target > 2")
        self.target_sequence = list(target_sequence) if target_sequence is not None else None
        digest, ids, self.rows = hashlib.sha256(), set(), []
        previous = -1
        membership = set()
        for row in rows:
            if set(row) != FIELDS:
                raise ValueError("unknown/missing workload fields, including future outcomes")
            if row["split"] not in {"train", "validation"} and not (
                    self.config.get("allow_locked_test", False) and row["split"] == "locked_test"):
                raise ValueError("locked test access is not authorized")
            membership.add((row["split"], row["window_id"]))
            if len(membership) > 1:
                raise ValueError("one simulator run consumes one split/window")
            t = integer(row["arrival_s"], "arrival")
            if t < previous or row["sim_tick"] != t or row["wall_monotonic_s"] is not None:
                raise ValueError("unsorted arrivals or inconsistent clocks")
            previous = t
            rid = row["request_id"]
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError("duplicate/invalid request ID")
            ids.add(rid)
            if row["job_type"] not in {"online", "offline"}:
                raise ValueError("invalid request class")
            integer(row["input_tokens"], "input tokens")
            integer(row["max_output_tokens"], "output budget", 1)
            integer(row["deadline_s"], "deadline", t)
            if len(row["prompt_token_ids"]) != row["input_tokens"]:
                raise ValueError("input token count mismatch")
            digest.update(canonical(row) + b"\n")
            self.rows.append({key: row[key] for key in ("request_id", "job_type", "arrival_s", "deadline_s", "input_tokens", "max_output_tokens")})
        self.input_sha256 = digest.hexdigest()
        self.config_sha256 = fingerprint(self.config)
        self.now, self.cursor, self.next_seq = 0, 0, 0
        self.started, self.finished, self.poisoned = False, False, False
        self._in_step = False
        self.jobs, self.online_queue, self.offline_queue, self.sink = {}, [], [], {}
        self.devices = [{"id": i, "state": "off", "ready_at": None, "jobs": []} for i in range(2)]
        self.counts = dict.fromkeys(COUNT_KEYS, 0)
        self.costs = dict.fromkeys(COST_KEYS, 0.0)

    def _emit(self, kind, **fields):
        event = {"event_id": str(self.next_seq), "seq": self.next_seq,
                 "time": self.now, "kind": kind, **fields}
        self.next_seq += 1
        self._events.append(event)
        if self.emit_callback is not None:
            self.emit_callback(copy.deepcopy(event))

    def _shutdown(self, device, reason):
        assert not device["jobs"]
        device.update(state="off", ready_at=None)
        self.counts["shutdown_events"] += 1
        self.costs["shutdown"] += self.config["accounting"]["shutdown_event"]
        self._emit("shutdown", device_id=device["id"], reason=reason)

    def _actuate(self, target):
        active = lambda: sum(d["state"] in {"active", "starting"} for d in self.devices)
        # Starting instances cannot be cancelled; the target is reconsidered at
        # their observable ready event. Repeated targets never duplicate starts.
        for device in reversed(self.devices):
            if active() > target and device["state"] == "active":
                device["state"] = "draining"
                self._emit("drain", device_id=device["id"])
                if not device["jobs"]:
                    self._shutdown(device, "drain_complete")
        for device in self.devices:
            if active() < target and device["state"] == "draining":
                device["state"] = "active"
                self._emit("drain_cancel", device_id=device["id"])
        for device in self.devices:
            if active() < target and device["state"] == "off":
                device.update(state="starting", ready_at=self.now + self.config["model"]["startup_seconds"])
                self.counts["startup_events"] += 1
                self.costs["startup"] += self.config["accounting"]["startup_event"]
                self._emit("startup", device_id=device["id"], ready_at=device["ready_at"], tp=1)
                if device["ready_at"] == self.now:
                    device.update(state="active", ready_at=None)
                    self._emit("ready", device_id=device["id"])

    def _admit_local(self, queue):
        slots = self.config["model"]["local_slots"]
        while queue:
            available = [d for d in self.devices if d["state"] == "active" and len(d["jobs"]) < slots]
            if not available:
                break
            device = min(available, key=lambda d: (len(d["jobs"]), d["id"]))
            rid = queue.pop(0)
            job = self.jobs[rid]
            job.update(status="local", started_at=self.now, device_id=device["id"],
                       work_remaining=job["input_tokens"] / self.config["model"]["prefill_tps"] + job["max_output_tokens"] / self.config["model"]["decode_tps"])
            device["jobs"].append(rid)
            self._emit("local_start", request_id=rid, device_id=device["id"])

    def _complete(self, rid, kind):
        job = self.jobs[rid]
        expected = "local" if kind == "local_complete" else "sink"
        if job["status"] != expected:
            raise ValueError("duplicate or invalid completion")
        job.update(status="completed", finished_at=self.now)
        self.counts["local_completed" if expected == "local" else "synthetic_api_completed"] += 1
        if job["job_type"] == "online":
            self.counts["online_sla_met"] += self.now - job["arrival_s"] <= self.config["feasibility"]["online_e2e_sla_seconds"]
        else:
            self.counts["offline_deadline_met"] += self.now <= job["deadline_s"]
        fields = {"request_id": rid, "output_tokens": job["max_output_tokens"],
                  "service_seconds": self.now - job["started_at"],
                  "wait_seconds": job["started_at"] - job["arrival_s"],
                  "end_to_end_seconds": self.now - job["arrival_s"],
                  "evidence": "constructed_service" if expected == "local" else "synthetic_api"}
        if expected == "local":
            fields["device_id"] = job["device_id"]
        self._emit(kind, **fields)

    def _deadlines(self):
        for rid, job in self.jobs.items():
            if job["job_type"] == "offline" and not job.get("deadline_miss", False) and self.now >= job["deadline_s"] and not (job["status"] == "completed" and job["finished_at"] <= job["deadline_s"]):
                job["deadline_miss"] = True
                self.counts["offline_deadline_misses"] += 1
                self.costs["offline_miss"] += self.config["accounting"]["offline_deadline_miss"]
                self._emit("offline_miss", request_id=rid, deadline_s=job["deadline_s"])

    def _finish(self):
        for rid, job in self.jobs.items():
            if job["status"] != "completed":
                self._emit("censor", request_id=rid, previous_status=job["status"], reason="finite_horizon_drain_exhausted")
                job["status"] = "censored"
                self.counts["censored"] += 1
        self.online_queue.clear()
        self.offline_queue.clear()
        self.sink.clear()
        for device in self.devices:
            device["jobs"].clear()
            if device["state"] != "off":
                self._shutdown(device, "run_end")
        self.finished = True
        self._emit("run_end", arrived=self.counts["arrived"], empty_workload=self.counts["arrived"] == 0)

    def step(self):
        if self.finished:
            return []
        if self.poisoned or self._in_step:
            raise RuntimeError("failed or reentrant step; restore last durable checkpoint")
        self._in_step, self._events = True, []
        try:
            if not self.started:
                self._emit("run_start", mode=self.mode, input_sha256=self.input_sha256,
                           config_sha256=self.config_sha256, policy=self.policy.spec,
                           conditional_target_sha256=fingerprint(self.target_sequence) if self.target_sequence is not None else None,
                           model_identity=self.config["model"]["identity"], real_backend=False)
                self.started = True
            for device in self.devices:
                if device["state"] == "starting" and device["ready_at"] <= self.now:
                    device.update(state="active", ready_at=None)
                    self._emit("ready", device_id=device["id"])
            while self.cursor < len(self.rows) and self.rows[self.cursor]["arrival_s"] <= self.now and self.now < self.config["window_seconds"]:
                row = self.rows[self.cursor]
                self.cursor += 1
                rid = row["request_id"]
                self.jobs[rid] = dict(row, status="waiting")
                (self.online_queue if row["job_type"] == "online" else self.offline_queue).append(rid)
                self.counts["arrived"] += 1
                self.counts[row["job_type"] + "_arrived"] += 1
                self._emit("arrival", **row)
            obs = {"time": self.now, "queue_online": len(self.online_queue), "queue_offline": len(self.offline_queue),
                   "local_running": sum(len(d["jobs"]) for d in self.devices),
                   **{state: sum(d["state"] == state for d in self.devices) for state in ("active", "starting", "draining", "off")}}
            target = self.policy.decide(obs) if self.target_sequence is None else self.target_sequence[self.now]
            self._emit("policy_decision", mode=self.mode, observation=obs, target=target)
            self._actuate(target)
            self._admit_local(self.online_queue)
            sink_config = self.config["synthetic_api"]
            while self.online_queue and len(self.sink) < sink_config["capacity"]:
                rid = self.online_queue[0]
                job = self.jobs[rid]
                if self.now - job["arrival_s"] < sink_config["route_online_after_wait_seconds"]:
                    break
                self.online_queue.pop(0)
                job.update(status="sink", started_at=self.now)
                self.sink[rid] = self.now + sink_config["latency_seconds"]
                self.counts["synthetic_api_accepted"] += 1
                prices = self.config["accounting"]
                self.costs["synthetic_api"] += job["input_tokens"] * prices["api_input_token"] + job["max_output_tokens"] * prices["api_output_token"]
                self._emit("sink_accept", request_id=rid, complete_at=self.sink[rid], input_tokens=job["input_tokens"], assumed_output_tokens=job["max_output_tokens"], evidence="synthetic_api")
            self._admit_local(self.offline_queue)
            completed = []
            for device in self.devices:
                state = device["state"]
                if state == "off":
                    continue
                idle = state == "active" and not device["jobs"]
                self.counts["gpu_occupied_seconds"] += 1
                self.counts["gpu_" + state + "_seconds"] += 1
                self.counts["gpu_idle_seconds"] += idle
                self.costs["gpu"] += self.config["accounting"]["gpu_second"]
                self._emit("gpu_interval", device_id=device["id"], state=state, duration_seconds=1, idle=idle)
                count = len(device["jobs"])
                for rid in device["jobs"]:
                    self.jobs[rid]["work_remaining"] -= 1 / count
                    if self.jobs[rid]["work_remaining"] <= 1e-12:
                        completed.append((device, rid))
            self.now += 1
            for device, rid in completed:
                device["jobs"].remove(rid)
                self._complete(rid, "local_complete")
            for rid, complete_at in list(self.sink.items()):
                if complete_at <= self.now:
                    del self.sink[rid]
                    self._complete(rid, "sink_complete")
            for device in self.devices:
                if device["state"] == "draining" and not device["jobs"]:
                    self._shutdown(device, "drain_complete")
            self._deadlines()
            self._emit("tick", target=target, action={"target_devices": target},
                       active=sum(d["state"] == "active" for d in self.devices),
                       starting=sum(d["state"] == "starting" for d in self.devices),
                       draining=sum(d["state"] == "draining" for d in self.devices),
                       queue_online=len(self.online_queue), queue_offline=len(self.offline_queue),
                       local_running=sum(len(d["jobs"]) for d in self.devices), sink_running=len(self.sink))
            if self.now == self.end:
                self._finish()
            return self._events
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self._in_step = False

    def run(self):
        while not self.finished:
            self.step()
        return self.summary()

    def summary(self):
        c = dict(self.counts)
        c.update(completed=c["local_completed"] + c["synthetic_api_completed"],
                 pending=c["arrived"] - c["local_completed"] - c["synthetic_api_completed"] - c["censored"],
                 queue_pending=len(self.online_queue) + len(self.offline_queue),
                 local_pending=sum(len(d["jobs"]) for d in self.devices), sink_pending=len(self.sink),
                 virtual_settlement=0, cost_components=dict(self.costs), total_cost=sum(self.costs.values()),
                 online_violation_rate=1 - c["online_sla_met"] / c["online_arrived"] if c["online_arrived"] else None,
                 offline_deadline_completion_rate=c["offline_deadline_met"] / c["offline_arrived"] if c["offline_arrived"] else None,
                 elapsed_seconds=self.now, finished=self.finished, mode=self.mode,
                 empty_workload=c["arrived"] == 0, policy_result_eligible=False)
        return c

    def snapshot(self):
        if self._in_step or self.poisoned:
            raise RuntimeError("snapshot requires a complete, successful tick boundary")
        state = {key: copy.deepcopy(getattr(self, key)) for key in
                 ("now", "cursor", "next_seq", "started", "finished", "jobs", "online_queue", "offline_queue", "sink", "devices", "counts", "costs")}
        state["policy_state"] = copy.deepcopy(self.policy.state)
        snapshot = {"schema": "maxopt-simulator-checkpoint-v1", "input_sha256": self.input_sha256,
                    "config_sha256": self.config_sha256, "policy": copy.deepcopy(self.policy.spec),
                    "mode": self.mode, "target_sequence": copy.deepcopy(self.target_sequence),
                    "state": state, "state_sha256": fingerprint(state)}
        snapshot["snapshot_sha256"] = fingerprint(snapshot)
        return snapshot

    @classmethod
    def restore(cls, rows, config, snapshot, emit=None):
        body = {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
        if snapshot.get("schema") != "maxopt-simulator-checkpoint-v1" or fingerprint(body) != snapshot.get("snapshot_sha256") or fingerprint(snapshot["state"]) != snapshot["state_sha256"]:
            raise ValueError("checkpoint schema/hash mismatch")
        simulator = cls(rows, config, snapshot["policy"], emit,
                        mode=snapshot["mode"], target_sequence=snapshot["target_sequence"])
        if simulator.input_sha256 != snapshot["input_sha256"] or simulator.config_sha256 != snapshot["config_sha256"]:
            raise ValueError("checkpoint input/config mismatch")
        state = copy.deepcopy(snapshot["state"])
        simulator.policy.state = state.pop("policy_state")
        expected = {"now", "cursor", "next_seq", "started", "finished", "jobs", "online_queue", "offline_queue", "sink", "devices", "counts", "costs"}
        if set(state) != expected:
            raise ValueError("checkpoint state fields mismatch")
        for key, value in state.items():
            setattr(simulator, key, value)
        integer(simulator.now, "checkpoint time")
        integer(simulator.cursor, "checkpoint cursor")
        integer(simulator.next_seq, "checkpoint event sequence")
        if type(simulator.started) is not bool or type(simulator.finished) is not bool or simulator.started != (simulator.now > 0):
            raise ValueError("checkpoint started/finished flags mismatch")
        if simulator.now > simulator.end or simulator.cursor > len(simulator.rows) or simulator.finished != (simulator.now == simulator.end):
            raise ValueError("checkpoint time/cursor mismatch")
        if (simulator.next_seq == 0) != (simulator.now == 0) or simulator.next_seq < 2 * simulator.now:
            raise ValueError("checkpoint event sequence inconsistent with ticks")
        if set(simulator.counts) != set(COUNT_KEYS) or set(simulator.costs) != set(COST_KEYS):
            raise ValueError("checkpoint accounting fields mismatch")
        for key, value in simulator.counts.items():
            integer(value, "checkpoint counter " + key)
        for key, value in simulator.costs.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("checkpoint cost must be finite and nonnegative: " + key)
        policy_state = simulator.policy.state
        if set(policy_state) != {"target", "last_change_at"}:
            raise ValueError("checkpoint policy state fields mismatch")
        integer(policy_state["target"], "checkpoint target")
        cooldown = simulator.policy.spec.get("cooldown_seconds", 0)
        if policy_state["target"] > 2 or type(policy_state["last_change_at"]) is not int or not -cooldown <= policy_state["last_change_at"] <= max(0, simulator.now - 1):
            raise ValueError("checkpoint policy cooldown/target mismatch")
        if simulator.policy.spec["name"] in {"all1", "all2"} and simulator.started and simulator.target_sequence is None and policy_state["target"] != int(simulator.policy.spec["name"][-1]):
            raise ValueError("checkpoint fixed policy target mismatch")
        expected_cursor = sum(row["arrival_s"] < min(simulator.now, simulator.config["window_seconds"]) for row in simulator.rows)
        if simulator.cursor != expected_cursor or set(simulator.jobs) != {r["request_id"] for r in simulator.rows[:expected_cursor]} or simulator.counts["arrived"] != expected_cursor:
            raise ValueError("checkpoint arrival cursor/count mismatch")
        if [d["id"] for d in simulator.devices] != [0, 1]:
            raise ValueError("checkpoint device identity mismatch")
        for device in simulator.devices:
            if set(device) != {"id", "state", "ready_at", "jobs"}:
                raise ValueError("checkpoint device fields mismatch")
            if device["state"] not in {"off", "active", "starting", "draining"} or len(device["jobs"]) > simulator.config["model"]["local_slots"]:
                raise ValueError("checkpoint device state/capacity mismatch")
            if device["state"] in {"off", "starting"} and device["jobs"]:
                raise ValueError("checkpoint jobs on unavailable device")
            if device["state"] == "starting":
                integer(device["ready_at"], "checkpoint ready time", simulator.now)
                if device["ready_at"] >= simulator.now + simulator.config["model"]["startup_seconds"]:
                    raise ValueError("checkpoint ready time exceeds startup bound")
            elif device["ready_at"] is not None:
                raise ValueError("checkpoint spurious ready time")
        locations = simulator.online_queue + simulator.offline_queue + list(simulator.sink)
        locations += [rid for device in simulator.devices for rid in device["jobs"]]
        pending = {rid for rid, job in simulator.jobs.items() if job["status"] in {"waiting", "local", "sink"}}
        if len(locations) != len(set(locations)) or set(locations) != pending:
            raise ValueError("checkpoint queue conservation mismatch")
        if any(simulator.jobs[r]["status"] != "waiting" or simulator.jobs[r]["job_type"] != kind
               for queue, kind in ((simulator.online_queue, "online"), (simulator.offline_queue, "offline")) for r in queue):
            raise ValueError("checkpoint waiting-class mismatch")
        for device in simulator.devices:
            for rid in device["jobs"]:
                job = simulator.jobs[rid]
                if job["status"] != "local" or job["device_id"] != device["id"] or not math.isfinite(job["work_remaining"]) or job["work_remaining"] <= 0:
                    raise ValueError("checkpoint local work mismatch")
        if len(simulator.sink) > simulator.config["synthetic_api"]["capacity"] or any(simulator.jobs[r]["status"] != "sink" or due <= simulator.now for r, due in simulator.sink.items()):
            raise ValueError("checkpoint sink state mismatch")
        # Hashes bind bytes, while these checks reject semantically impossible
        # snapshots even if a corrupt producer recalculates its own hashes.
        model, prices = simulator.config["model"], simulator.config["accounting"]
        expected_counts = dict.fromkeys(("arrived", "online_arrived", "offline_arrived", "local_completed",
                                        "synthetic_api_accepted", "synthetic_api_completed", "censored",
                                        "online_sla_met", "offline_deadline_met", "offline_deadline_misses"), 0)
        expected_api_cost = 0.0
        for row in simulator.rows[:expected_cursor]:
            job = simulator.jobs[row["request_id"]]
            if any(job.get(key) != value or type(job.get(key)) is not type(value) for key, value in row.items()):
                raise ValueError("checkpoint immutable request fields mismatch")
            status = job["status"]
            if status not in {"waiting", "local", "sink", "completed", "censored"} or (status == "censored" and not simulator.finished):
                raise ValueError("checkpoint job status mismatch")
            expected_counts["arrived"] += 1
            expected_counts[row["job_type"] + "_arrived"] += 1
            local = "device_id" in job
            admitted = "started_at" in job
            allowed = set(row) | {"status"}
            if admitted:
                allowed.add("started_at")
                integer(job["started_at"], "checkpoint admission time", row["arrival_s"])
                if job["started_at"] >= simulator.now:
                    raise ValueError("checkpoint admission has not occurred")
            if local:
                allowed |= {"device_id", "work_remaining"}
                if not admitted or type(job["device_id"]) is not int or job["device_id"] not in {0, 1}:
                    raise ValueError("checkpoint local identity mismatch")
                if type(job["work_remaining"]) not in (int, float) or not math.isfinite(job["work_remaining"]):
                    raise ValueError("checkpoint nonfinite local work")
                original_work = row["input_tokens"] / model["prefill_tps"] + row["max_output_tokens"] / model["decode_tps"]
                if not -1 <= job["work_remaining"] <= original_work or (status == "completed" and job["work_remaining"] > 1e-12):
                    raise ValueError("checkpoint impossible remaining local work")
            elif admitted:
                if row["job_type"] != "online" or job["started_at"] - row["arrival_s"] < simulator.config["synthetic_api"]["route_online_after_wait_seconds"]:
                    raise ValueError("checkpoint invalid synthetic API admission")
                expected_counts["synthetic_api_accepted"] += 1
                expected_api_cost += row["input_tokens"] * prices["api_input_token"] + row["max_output_tokens"] * prices["api_output_token"]
            if status == "waiting" and admitted or status in {"local", "sink", "completed"} and not admitted or status == "local" and not local or status == "sink" and local:
                raise ValueError("checkpoint status/admission mismatch")
            if status == "completed":
                allowed.add("finished_at")
                integer(job["finished_at"], "checkpoint completion time", job["started_at"] + 1)
                if job["finished_at"] > simulator.now:
                    raise ValueError("checkpoint completion in future")
                if not local and job["finished_at"] != job["started_at"] + simulator.config["synthetic_api"]["latency_seconds"]:
                    raise ValueError("checkpoint synthetic API completion delay mismatch")
                expected_counts["local_completed" if local else "synthetic_api_completed"] += 1
                if row["job_type"] == "online":
                    expected_counts["online_sla_met"] += job["finished_at"] - row["arrival_s"] <= simulator.config["feasibility"]["online_e2e_sla_seconds"]
                else:
                    expected_counts["offline_deadline_met"] += job["finished_at"] <= row["deadline_s"]
            if status == "sink" and simulator.sink[row["request_id"]] != job["started_at"] + simulator.config["synthetic_api"]["latency_seconds"]:
                raise ValueError("checkpoint sink due time mismatch")
            expected_counts["censored"] += status == "censored"
            missed = row["job_type"] == "offline" and simulator.now >= row["deadline_s"] and not (status == "completed" and job["finished_at"] <= row["deadline_s"])
            if missed:
                allowed.add("deadline_miss")
                if job.get("deadline_miss") is not True:
                    raise ValueError("checkpoint missing deadline penalty")
                expected_counts["offline_deadline_misses"] += 1
            if set(job) != allowed:
                raise ValueError("checkpoint unknown/inconsistent job fields")
        if any(simulator.counts[key] != value for key, value in expected_counts.items()):
            raise ValueError("checkpoint request counters mismatch")
        counts = simulator.counts
        if counts["gpu_occupied_seconds"] != sum(counts["gpu_" + state + "_seconds"] for state in ("active", "starting", "draining")) or counts["gpu_occupied_seconds"] > 2 * simulator.now or counts["gpu_idle_seconds"] > counts["gpu_active_seconds"]:
            raise ValueError("checkpoint GPU counter conservation mismatch")
        if counts["startup_events"] - counts["shutdown_events"] != sum(d["state"] != "off" for d in simulator.devices):
            raise ValueError("checkpoint transition conservation mismatch")
        expected_costs = {"gpu": counts["gpu_occupied_seconds"] * prices["gpu_second"],
                          "startup": counts["startup_events"] * prices["startup_event"],
                          "shutdown": counts["shutdown_events"] * prices["shutdown_event"],
                          "synthetic_api": expected_api_cost,
                          "offline_miss": counts["offline_deadline_misses"] * prices["offline_deadline_miss"]}
        if any(not math.isclose(simulator.costs[key], cost, abs_tol=1e-10, rel_tol=1e-10) for key, cost in expected_costs.items()):
            raise ValueError("checkpoint costs do not match primitive counters")
        if simulator.now == 0 and (simulator.jobs or simulator.cursor or any(counts.values()) or any(simulator.costs.values()) or any(d["state"] != "off" for d in simulator.devices)):
            raise ValueError("checkpoint initial state mismatch")
        return simulator
