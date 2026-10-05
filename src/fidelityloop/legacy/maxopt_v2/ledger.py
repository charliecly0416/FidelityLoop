"""Independent event replay for V2 accounting and request conservation.

This module deliberately does not call Simulator or its accounting functions.
It derives charges from primitive events and the registered scenario prices.
"""
from __future__ import annotations

import math
import hashlib
import json
import copy


def recompute(events, config):
    normalized = copy.deepcopy(config)
    normalized["model"] = {"identity": "uncalibrated_equal_share_pilot_v1", "prefill_tps": 1024.0,
                           "decode_tps": 128.0, "local_slots": 4, "startup_seconds": 30,
                           "max_devices": 2, **config.get("model", {})}
    config_sha256 = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    prices, sink_config = config["accounting"], config["synthetic_api"]
    slots = config.get("model", {}).get("local_slots", 4)
    counts = dict.fromkeys(("arrived", "online_arrived", "offline_arrived", "local_completed",
                           "synthetic_api_accepted", "synthetic_api_completed", "censored",
                           "online_sla_met", "offline_deadline_met", "offline_deadline_misses",
                           "gpu_occupied_seconds", "gpu_active_seconds", "gpu_starting_seconds",
                           "gpu_draining_seconds", "gpu_idle_seconds", "startup_events", "shutdown_events"), 0)
    costs = dict.fromkeys(("gpu", "startup", "shutdown", "synthetic_api", "offline_miss"), 0.0)
    devices = {i: {"state": "off", "jobs": set(), "ready_at": None} for i in range(2)}
    jobs, accepted, deadline_misses = {}, {}, set()
    intervals, seen_events = set(), set()
    previous_time, elapsed, mode = 0, 0, None
    finished, started = False, False
    expected_seq = 0

    def fail(message):
        raise ValueError(message)

    for event in events:
        if finished:
            fail("events after run_end")
        seq = event.get("seq")
        if type(seq) is not int or seq != expected_seq or event.get("event_id") != str(seq) or event["event_id"] in seen_events:
            fail("duplicate/noncontiguous event identity")
        seen_events.add(event["event_id"])
        expected_seq += 1
        now = event["time"]
        if type(now) is not int or now < previous_time or now > previous_time + 1:
            fail("event time must be nondecreasing one-second steps")
        if now > previous_time:
            expected = {i for i, dev in devices.items() if dev["state"] != "off"}
            if intervals != expected:
                fail("missing/extra GPU interval charge")
            intervals.clear()
        previous_time = now
        kind = event["kind"]
        rid = event.get("request_id")
        if kind == "run_start":
            if started or seq != 0 or now != 0 or event["real_backend"] is not False:
                fail("invalid run start")
            if event["config_sha256"] != config_sha256 or event["model_identity"] != normalized["model"]["identity"]:
                fail("run configuration/model binding mismatch")
            mode = event["mode"]
            if mode not in {"independent_closed_loop", "conditional_action_replay_diagnostic"}:
                fail("unknown prediction mode")
            if (event["conditional_target_sha256"] is None) != (mode == "independent_closed_loop"):
                fail("conditional replay identity mismatch")
            started = True
            continue
        if not started:
            fail("missing run_start")
        if kind == "arrival":
            if rid in jobs or event["arrival_s"] != now or not 0 <= now < config["window_seconds"]:
                fail("duplicate/late arrival")
            if event["job_type"] not in {"online", "offline"}:
                fail("unknown request class")
            jobs[rid] = dict(event, status="waiting")
            counts["arrived"] += 1
            counts[event["job_type"] + "_arrived"] += 1
        elif kind in {"startup", "ready", "drain", "drain_cancel", "shutdown"}:
            device = devices[event["device_id"]]
            if kind == "startup":
                if device["state"] != "off" or event["tp"] != 1:
                    fail("duplicate start or unsupported tensor parallelism")
                if type(event["ready_at"]) is not int or event["ready_at"] != now + normalized["model"]["startup_seconds"]:
                    fail("startup ready time differs from registered delay")
                device.update(state="starting", ready_at=event["ready_at"])
                counts["startup_events"] += 1
                costs["startup"] += prices["startup_event"]
            elif kind == "ready":
                if device["state"] != "starting" or now != device["ready_at"]:
                    fail("ready differs from startup delay")
                device.update(state="active", ready_at=None)
            elif kind == "drain":
                if device["state"] != "active":
                    fail("only active instances may drain")
                device["state"] = "draining"
            elif kind == "drain_cancel":
                if device["state"] != "draining":
                    fail("only draining instances may reopen")
                device["state"] = "active"
            else:
                if device["state"] == "off" or device["jobs"]:
                    fail("duplicate shutdown or interrupted local jobs")
                if device["state"] == "starting" and event["reason"] != "run_end":
                    fail("startup cancellation is forbidden before run end")
                device.update(state="off", ready_at=None)
                counts["shutdown_events"] += 1
                costs["shutdown"] += prices["shutdown_event"]
        elif kind == "local_start":
            job, device = jobs[rid], devices[event["device_id"]]
            if job["status"] != "waiting" or device["state"] != "active" or len(device["jobs"]) >= slots:
                fail("invalid local admission")
            job.update(status="local", device_id=event["device_id"], started_at=now)
            device["jobs"].add(rid)
        elif kind == "sink_accept":
            job = jobs[rid]
            if job["status"] != "waiting" or job["job_type"] != "online" or len(accepted) >= sink_config["capacity"]:
                fail("invalid/overcapacity synthetic API admission")
            if now - job["arrival_s"] < sink_config["route_online_after_wait_seconds"]:
                fail("synthetic API routed before wait threshold")
            if event["input_tokens"] != job["input_tokens"] or event["assumed_output_tokens"] != job["max_output_tokens"] or event["complete_at"] != now + sink_config["latency_seconds"]:
                fail("synthetic API budget/delay changed")
            job.update(status="sink", started_at=now)
            accepted[rid] = event["complete_at"]
            counts["synthetic_api_accepted"] += 1
            costs["synthetic_api"] += prices["api_input_token"] * job["input_tokens"] + prices["api_output_token"] * job["max_output_tokens"]
        elif kind in {"local_complete", "sink_complete"}:
            job = jobs[rid]
            required = "local" if kind == "local_complete" else "sink"
            if job["status"] != required:
                fail("duplicate completion or completion without admission")
            if required == "sink":
                if now < accepted[rid]:
                    fail("synthetic API completion before elapsed delay")
                del accepted[rid]
                counts["synthetic_api_completed"] += 1
            else:
                if job["device_id"] != event["device_id"]:
                    fail("completion moved between instances")
                devices[job["device_id"]]["jobs"].remove(rid)
                counts["local_completed"] += 1
            if event["output_tokens"] != job["max_output_tokens"] or event["service_seconds"] != now - job["started_at"] or event["wait_seconds"] != job["started_at"] - job["arrival_s"] or event["end_to_end_seconds"] != now - job["arrival_s"]:
                fail("completion token/time fields inconsistent")
            job.update(status="completed", finished_at=now)
            if job["job_type"] == "online":
                counts["online_sla_met"] += now - job["arrival_s"] <= config["feasibility"]["online_e2e_sla_seconds"]
            else:
                counts["offline_deadline_met"] += now <= job["deadline_s"]
        elif kind == "offline_miss":
            job = jobs[rid]
            if rid in deadline_misses or job["job_type"] != "offline" or now < job["deadline_s"] or (job["status"] == "completed" and job["finished_at"] <= job["deadline_s"]):
                fail("invalid/duplicate offline deadline charge")
            deadline_misses.add(rid)
            counts["offline_deadline_misses"] += 1
            costs["offline_miss"] += prices["offline_deadline_miss"]
        elif kind == "gpu_interval":
            device_id = event["device_id"]
            device = devices[device_id]
            if device_id in intervals or event["state"] != device["state"] or device["state"] == "off" or event["duration_seconds"] != 1:
                fail("invalid/duplicate GPU interval")
            intervals.add(device_id)
            idle = device["state"] == "active" and not device["jobs"]
            if event["idle"] != idle:
                fail("idle accounting mismatch")
            counts["gpu_occupied_seconds"] += 1
            counts["gpu_" + device["state"] + "_seconds"] += 1
            counts["gpu_idle_seconds"] += idle
            costs["gpu"] += prices["gpu_second"]
        elif kind == "censor":
            job = jobs[rid]
            if now != config["window_seconds"] + config["drain_seconds"] or job["status"] in {"completed", "censored"} or event["previous_status"] != job["status"]:
                fail("invalid censoring")
            if job["status"] == "local":
                devices[job["device_id"]]["jobs"].remove(rid)
            if job["status"] == "sink":
                del accepted[rid]
            job["status"] = "censored"
            counts["censored"] += 1
        elif kind == "policy_decision":
            obs = event["observation"]
            expected_obs = {"time": now, "queue_online": sum(j["status"] == "waiting" and j["job_type"] == "online" for j in jobs.values()),
                            "queue_offline": sum(j["status"] == "waiting" and j["job_type"] == "offline" for j in jobs.values()),
                            "local_running": sum(len(d["jobs"]) for d in devices.values()),
                            **{s: sum(d["state"] == s for d in devices.values()) for s in ("active", "starting", "draining", "off")}}
            if obs != expected_obs or type(event["target"]) is not int or not 0 <= event["target"] <= 2 or event["mode"] != mode:
                fail("noncausal/mismatched policy observation")
        elif kind == "tick":
            if now != elapsed + 1:
                fail("noncontiguous tick")
            elapsed = now
            for s in ("active", "starting", "draining"):
                if event[s] != sum(d["state"] == s for d in devices.values()):
                    fail("tick lifecycle counts mismatch")
            if event["local_running"] != sum(len(d["jobs"]) for d in devices.values()) or event["sink_running"] != len(accepted):
                fail("tick pending counts mismatch")
        elif kind == "run_end":
            if now != config["window_seconds"] + config["drain_seconds"] or any(j["status"] not in {"completed", "censored"} for j in jobs.values()) or any(d["state"] != "off" for d in devices.values()):
                fail("incomplete terminal conservation")
            if event["arrived"] != counts["arrived"] or event["empty_workload"] != (counts["arrived"] == 0):
                fail("terminal count mismatch")
            finished = True
        else:
            fail("unknown event kind")
    if not started:
        fail("empty event stream is not a completed run")
    if intervals:
        fail("event prefix ends within an uncommitted tick")
    # Every observable missed deadline must have a penalty record, including
    # unfinished requests. A completed-only denominator is never used.
    overdue = {rid for rid, j in jobs.items() if j["job_type"] == "offline" and j["deadline_s"] <= elapsed and not (j["status"] == "completed" and j["finished_at"] <= j["deadline_s"])}
    if overdue != deadline_misses:
        fail("missing offline deadline charge")
    c = counts
    c.update(completed=c["local_completed"] + c["synthetic_api_completed"],
             pending=c["arrived"] - c["local_completed"] - c["synthetic_api_completed"] - c["censored"],
             queue_pending=sum(j["status"] == "waiting" for j in jobs.values()),
             local_pending=sum(j["status"] == "local" for j in jobs.values()), sink_pending=len(accepted),
             virtual_settlement=0, cost_components=costs, total_cost=sum(costs.values()),
             online_violation_rate=1 - c["online_sla_met"] / c["online_arrived"] if c["online_arrived"] else None,
             offline_deadline_completion_rate=c["offline_deadline_met"] / c["offline_arrived"] if c["offline_arrived"] else None,
             elapsed_seconds=elapsed, finished=finished, mode=mode,
             empty_workload=c["arrived"] == 0, policy_result_eligible=False)
    if c["pending"] != c["queue_pending"] + c["local_pending"] + c["sink_pending"] or any(not math.isfinite(value) or value < 0 for value in costs.values()):
        fail("final accounting/request conservation failed")
    return c
