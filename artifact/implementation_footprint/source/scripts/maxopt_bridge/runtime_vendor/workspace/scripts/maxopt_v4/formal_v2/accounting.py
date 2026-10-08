"""Read-only reconstruction of V4 request metrics and scenario cost.

This module consumes controller primitives, checks their request-ledger projection,
and never imports a GPU runtime or uses service/lifecycle predictions as costs.
Like V3 n5_acceptance, occupied time runs from launch_issued to verified_release;
active time is the integral of lifecycle state == active (not CUDA utilization).
All intervals are half-open; completion exactly at cutoff/deadline is accepted.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRICE_KEYS = ("gpu_second", "startup_event", "shutdown_event", "api_input_token",
              "api_output_token", "offline_deadline_miss")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _tails(values):
    return {"count": len(values), "p95_seconds": _quantile(values, .95),
            "p99_seconds": _quantile(values, .99), "max_seconds": max(values) if values else None}


def _integrate(intervals, left, right):
    return sum(max(0, min(stop, right) - max(start, left)) / 1e9 for start, stop in intervals)


def _contract(plan):
    if isinstance(plan.get("contract"), dict):
        contract = plan["contract"]
        binding = {"source": "embedded_plan_contract", "bound": True,
                   "canonical_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    else:
        path = Path(plan.get("contract_path", ROOT / "artifacts/max_optimization_v3_20260921/s4/contract.json"))
        digest = _sha(path)
        bound = plan.get("contract_sha256") == digest
        if "contract_sha256" in plan:
            _require(bound, "contract SHA256 mismatch")
        contract = _json(path)
        binding = {"source": str(path), "sha256": digest, "bound": bound}
    prices = contract["accounting"]
    for key in PRICE_KEYS:
        value = prices[key]
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid price: " + key)
    _require(prices.get("unit") == "scenario_USD", "price unit must be scenario_USD")
    return contract, binding


def analyze(plan, out):
    """Return derived evidence without writing files or granting formal acceptance.

    Missing end/cleanup evidence produces an incomplete, observed-only ledger.
    Contradictory identities, timestamps, projections or contracts raise ValueError.
    The caller may save this result to a *new* evidence path after raw files close.
    """
    out = Path(out)
    plan = _json(plan) if isinstance(plan, (str, Path)) else plan
    rows = _json(out / "expected_requests.json")
    controller = _lines(out / "controller_events.jsonl")
    projected = _lines(out / "request_events.jsonl")
    contract, binding = _contract(plan)
    prices = contract["accounting"]
    horizon = plan["horizon_seconds"]
    _require(type(horizon) is int and horizon > 0, "invalid horizon")
    package = plan.get("work_package")
    smoke = plan.get("formal") is False and plan.get("smoke", {}).get("nonformal") is True
    if smoke:
        _require(plan["smoke"].get("clock_scale", plan.get("clock_scale")) == 1, "smoke requires real clock_scale=1")
    defaults = (1500, 300, 300) if package == "W4" else (1800, 300, 0)
    lengths = tuple(plan.get(key, default) for key, default in zip(
        ("source_arrival_seconds", "drain_seconds", "padding_seconds"), defaults))
    _require(all(type(v) is int and v >= 0 for v in lengths) and
             (sum(lengths) == horizon or smoke and 0 < horizon <= sum(lengths)),
             "source/drain/padding must partition horizon")
    if package == "W4":
        _require(lengths == (1500, 300, 300), "W4 must retain 1500/300/300 segmentation")
    by_id = {r["request_id"]: r for r in rows}
    _require(len(by_id) == len(rows), "duplicate expected request ID")
    for row in rows:
        _require(row["job_type"] in ("online", "offline") and
                 0 <= row["arrival_s"] < horizon and
                 row["deadline_s"] == row["arrival_s"] + (60 if row["job_type"] == "online" else 300) and
                 row["deadline_s"] <= (sum(lengths) if smoke else horizon),
                 "invalid request class/arrival/deadline")
    expected_sha = _sha(out / "expected_requests.json")
    if plan.get("expected_requests_sha256"):
        _require(expected_sha == plan["expected_requests_sha256"], "expected requests SHA mismatch")
    _require(bool(controller), "empty controller ledger")
    for seq, event in enumerate(controller):
        _require(event["seq"] == seq, "controller sequence mismatch")
        stamp = event["controller_monotonic_ns"]
        _require(type(stamp) is int and (not seq or stamp >= controller[seq - 1]["controller_monotonic_ns"]),
                 "controller time is nonmonotonic")
    starts = [e for e in controller if e["kind"] == "observation_start"]
    _require(len(starts) == 1, "one observation_start is required")
    origin = starts[0]["origin_monotonic_ns"]
    _require(origin == starts[0]["controller_monotonic_ns"], "observation origin mismatch")
    _require(starts[0]["expected_observation_seconds"] == horizon, "observed horizon differs from plan")
    cutoff = origin + horizon * 10**9
    end_ns = controller[-1]["controller_monotonic_ns"]
    first_ns = controller[0]["controller_monotonic_ns"]
    issues = []
    if controller[0]["kind"] != "setup_start":
        issues.append("setup_start_missing")
    ends = [e for e in controller if e["kind"] == "observation_end"]
    _require(len(ends) <= 1, "duplicate observation_end")
    if not ends or ends[0]["controller_monotonic_ns"] < cutoff:
        issues.append("observation_incomplete")
    observed_end = min(cutoff, ends[0]["controller_monotonic_ns"] if ends else end_ns)
    clean = [e for e in controller if e["kind"] == "cleanup_complete"]
    if len(clean) != 1 or clean[0].get("complete") is not True:
        issues.append("cleanup_not_verified")
    failures = [e for e in controller if e["kind"] == "technical_failure"]
    if failures:
        issues.append("runtime_technical_failure")
    aborted = any(e.get("error_type") in ("CancelledError", "KeyboardInterrupt") or
                  e["kind"] in ("operator_abort", "run_aborted") for e in controller)

    generations, states, state_since = {}, {0: "off", 1: "off"}, {0: first_ns, 1: first_ns}
    state_intervals = {name: [] for name in ("off", "starting", "active", "draining", "stopping")}
    releases, dispatch, terminals, targets = {}, {}, {}, {}
    closed = False
    allowed = {"off": {"starting"}, "starting": {"active", "stopping"},
               "active": {"draining", "stopping"}, "draining": {"active", "stopping"}, "stopping": {"off"}}
    for event in controller:
        kind, stamp = event["kind"], event["controller_monotonic_ns"]
        if "at_s" in event:
            _require(math.isclose(event["at_s"], (stamp - origin) / 1e9, abs_tol=1e-8, rel_tol=0),
                     "controller relative timestamp mismatch")
        if kind == "observation_end":
            closed = True
        if kind in ("launch_issued", "ready", "shutdown_issued", "verified_release"):
            key = (event["gpu"], event["generation"])
            _require(key[0] in states, "unknown GPU")
            field = {"launch_issued": "launch", "ready": "ready", "shutdown_issued": "shutdown", "verified_release": "release"}[kind]
            if field == "launch":
                _require(key not in generations, "duplicate generation launch")
                _require(all(g["gpu"] != key[0] or "release" in g for g in generations.values()),
                         "overlapping generations on same GPU")
                generations[key] = {"gpu": key[0], "generation": key[1]}
            _require(key in generations and field not in generations[key], "missing/duplicate generation event")
            if field == "release":
                _require(event.get("release_verified") is True, "unverified release")
            generations[key][field] = stamp
        elif kind == "lifecycle_state":
            gpu, previous, new = event["gpu"], event["previous"], event["state"]
            _require(gpu in states and states[gpu] == previous and new in allowed[previous], "lifecycle transition mismatch")
            _require((gpu, event["generation"]) in generations, "lifecycle without generation")
            state_intervals[previous].append((state_since[gpu], stamp))
            states[gpu], state_since[gpu] = new, stamp
        elif kind in ("request_release", "local_dispatch", "api_accept"):
            rid = event["request_id"]
            _require(rid in by_id, "unknown raw request")
            _require(origin <= stamp < cutoff and not closed, "admission/release outside observation")
            row = by_id[rid]
            _require(stamp >= origin + int(row["arrival_s"] * 1e9), "request released before arrival")
            if kind == "request_release" and "scheduled_arrival_s" in event:
                _require(event["scheduled_arrival_s"] == row["arrival_s"], "scheduled arrival mismatch")
            target = releases if kind == "request_release" else dispatch
            _require(rid not in target, "duplicate release/dispatch")
            if kind != "request_release":
                _require(rid in releases, "dispatch before release")
                if kind == "api_accept":
                    _require(row["job_type"] == "online", "offline API acceptance forbidden")
                    _require(event["input_tokens"] == row["input_tokens"] and
                             event["output_budget"] == row["max_output_tokens"], "API token budget mismatch")
                    _require((stamp - origin) / 1e9 - row["arrival_s"] >= contract["synthetic_api"]["route_online_after_wait_seconds"],
                             "API acceptance before registered wait")
            target[rid] = event
        elif kind == "worker_receipt":
            worker = event["worker_event"]
            rid = worker.get("request_id")
            if rid in by_id and worker["kind"] in ("finished", "failed", "cancelled") and not closed and stamp <= cutoff:
                _require(rid in dispatch and dispatch[rid]["kind"] == "local_dispatch", "local terminal without local dispatch")
                d = dispatch[rid]
                _require(all(d[key] == event[key] for key in ("gpu", "generation")), "local terminal identity mismatch")
                _require(rid not in terminals, "duplicate raw terminal")
                if worker["kind"] == "finished":
                    _require(worker.get("backend_finished") is True and
                             len(worker.get("output_token_ids", [])) == by_id[rid]["max_output_tokens"] and
                             worker.get("prompt_token_ids") == by_id[rid].get("prompt_token_ids"),
                             "local completion token/evidence mismatch")
                terminals[rid] = (event, "completed" if worker["kind"] == "finished" else "failed")
        elif kind == "api_completed" and not closed and stamp <= cutoff:
            rid = event["request_id"]
            _require(rid in dispatch and dispatch[rid]["kind"] == "api_accept", "API terminal without acceptance")
            _require(rid not in terminals, "duplicate raw terminal")
            _require(stamp - dispatch[rid]["controller_monotonic_ns"] >= contract["synthetic_api"]["latency_seconds"] * 10**9,
                     "synthetic completion before elapsed delay")
            terminals[rid] = (event, "completed")
        elif kind == "policy_tick":
            _require(event["tick"] not in targets, "duplicate policy tick")
            targets[event["tick"]] = event["target"]
    for gpu, state in states.items():
        state_intervals[state].append((state_since[gpu], end_ns))
    if set(targets) != set(range(horizon)):
        issues.append("policy_tick_coverage_incomplete")
    if set(releases) != set(by_id):
        issues.append("unreleased_expected_requests")

    projection = {name: {} for name in ("release", "dispatch", "terminal")}
    for event in projected:
        kind, rid = event["kind"], event["request_id"]
        _require(kind in projection and rid in by_id and rid not in projection[kind], "unknown/duplicate request projection")
        seq = event["raw_controller_seq"]
        _require(type(seq) is int and 0 <= seq < len(controller), "bad raw controller reference")
        raw = controller[seq]
        _require(event["controller_monotonic_ns"] == raw["controller_monotonic_ns"] and
                 math.isclose(event["at_s"], (raw["controller_monotonic_ns"] - origin) / 1e9, abs_tol=1e-8, rel_tol=0),
                 "request projection timestamp mismatch")
        source = releases if kind == "release" else dispatch
        if kind != "terminal":
            _require(rid in source and source[rid]["seq"] == seq, "request projection source mismatch")
        elif rid in terminals:
            actual, status = terminals[rid]
            _require(seq == actual["seq"] and event["status"] == status, "terminal projection differs from raw")
        else:
            _require(event["status"] == "censored" and raw["kind"] == "observation_end", "unsupported projected terminal")
        if kind == "dispatch" or kind == "terminal" and "route" in event:
            d = dispatch[rid]
            route = "synthetic_api" if d["kind"] == "api_accept" else "gpu" + str(d["gpu"])
            _require(event["route"] == route, "request projection route mismatch")
        projection[kind][rid] = event
    _require(set(projection["release"]) == set(releases) and set(projection["dispatch"]) == set(dispatch),
             "request projection omitted raw release/dispatch")
    if set(projection["terminal"]) != set(by_id):
        issues.append("terminal_projection_incomplete")
    _require(set(terminals) <= set(projection["terminal"]), "request projection omitted raw terminal")

    requests = {}
    for rid, row in by_id.items():
        actual, status = terminals.get(rid, (None, "censored" if rid in projection["terminal"] else "unfinished"))
        d = dispatch.get(rid)
        completed = (actual["controller_monotonic_ns"] - origin) / 1e9 if status == "completed" else None
        dispatched = (d["controller_monotonic_ns"] - origin) / 1e9 if d else None
        _require(completed is None or dispatched is not None and completed >= dispatched, "completion precedes dispatch")
        route = ("synthetic_api" if d["kind"] == "api_accept" else "gpu" + str(d["gpu"])) if d else None
        requests[rid] = {"job_type": row["job_type"], "arrival_s": row["arrival_s"], "deadline_s": row["deadline_s"],
                         "released": rid in releases, "status": status, "route": route,
                         "dispatched_at": dispatched, "completed_at": completed,
                         "timely": completed is not None and completed <= row["deadline_s"],
                         "unfinished": status != "completed", "abort_affected": aborted and status != "completed",
                         "input_tokens": row["input_tokens"], "output_budget": row["max_output_tokens"]}
    populations = {}
    for kind in ("online", "offline"):
        subset = [r for r in requests.values() if r["job_type"] == kind]
        timely = sum(r["timely"] for r in subset)
        completed = [r for r in subset if r["status"] == "completed"]
        populations[kind] = {"arrivals": len(subset), "released": sum(r["released"] for r in subset),
            "completed": len(completed), "timely": timely, "not_on_time": len(subset) - timely,
            "timely_rate": timely / len(subset) if subset else None,
            "unfinished": len(subset) - len(completed), "censored": sum(r["status"] == "censored" for r in subset),
            "failed": sum(r["status"] == "failed" for r in subset), "abort_affected": sum(r["abort_affected"] for r in subset),
            "missing_terminal": sum(r["status"] == "unfinished" for r in subset),
            "late_completed": len(completed) - timely,
            "e2e": _tails([r["completed_at"] - r["arrival_s"] for r in completed]),
            "local_service": _tails([r["completed_at"] - r["dispatched_at"] for r in completed if r["route"] != "synthetic_api"])}

    occupied, generation_rows = [], []
    for generation in generations.values():
        g = dict(generation)
        ordered = [g[key] for key in ("launch", "ready", "shutdown", "release") if key in g]
        _require(ordered == sorted(ordered), "generation timestamps out of order")
        if "release" not in g:
            issues.append("generation_release_missing")
        else:
            _require("shutdown" in g, "release without shutdown")
        occupied.append((g["launch"], g.get("release", end_ns)))
        g["startup_seconds"] = (g["ready"] - g["launch"]) / 1e9 if "ready" in g else None
        g["shutdown_seconds"] = (g["release"] - g["shutdown"]) / 1e9 if "release" in g else None
        generation_rows.append(g)
    if not generations:
        issues.append("no_lifecycle_generations")
    if any(state != "off" for state in states.values()):
        issues.append("lifecycle_not_off_at_end")
    if any(p["failed"] for p in populations.values()):
        issues.append("failed_requests")
    api_events = [e for e in dispatch.values() if e["kind"] == "api_accept"]
    misses = [r for r in requests.values() if r["job_type"] == "offline" and not r["timely"] and r["deadline_s"] <= horizon]
    pending_deadlines = sum(r["job_type"] == "offline" and not r["timely"] and r["deadline_s"] > horizon
                            for r in requests.values())

    def phase(left, right, *, penalty=False):
        active = _integrate(state_intervals["active"], left, right)
        occ = _integrate(occupied, left, right)
        startup_count = sum(left <= g["launch"] < right for g in generation_rows)
        shutdown_count = sum(left <= g["shutdown"] < right for g in generation_rows if "shutdown" in g)
        accepted = [e for e in api_events if left <= e["controller_monotonic_ns"] < right]
        miss_count = sum(left <= origin + int(r["deadline_s"] * 1e9) < right or
                         (right == cutoff and origin + int(r["deadline_s"] * 1e9) == cutoff)
                         for r in misses) if penalty else 0
        # All expected-row misses are charged once at deadline, including abort
        # cases. These are scenario penalties, not observed cloud charges.
        components = {"gpu": occ * prices["gpu_second"], "startup": startup_count * prices["startup_event"],
            "shutdown": shutdown_count * prices["shutdown_event"],
            "synthetic_api": sum(by_id[e["request_id"]]["input_tokens"] * prices["api_input_token"] +
                                 by_id[e["request_id"]]["max_output_tokens"] * prices["api_output_token"] for e in accepted),
            "offline_miss": miss_count * prices["offline_deadline_miss"]}
        operating = sum(components[k] for k in ("gpu", "startup", "shutdown", "synthetic_api"))
        return {"start_s": (left - origin) / 1e9, "end_s": (right - origin) / 1e9,
                "occupied_seconds": occ, "active_seconds": active,
                "state_seconds": {k: _integrate(v, left, right) for k, v in state_intervals.items()},
                "startup_events": startup_count, "shutdown_events": shutdown_count,
                "api_accepted": len(accepted), "api_input_tokens_accepted": sum(by_id[e["request_id"]]["input_tokens"] for e in accepted),
                "api_output_budget_accepted": sum(by_id[e["request_id"]]["max_output_tokens"] for e in accepted),
                "offline_misses": miss_count, "cost_components": components,
                "operating_cost": operating, "penalty_cost": components["offline_miss"], "total_cost": operating + components["offline_miss"]}

    phases = {"setup": phase(first_ns, origin), "window": phase(origin, cutoff, penalty=True),
              "cleanup": phase(cutoff, max(cutoff, end_ns))}
    # On early abort cleanup may start within the nominal observation window.
    # Preserve the fixed window accounting but disclose actual runtime phase time.
    segments, left = {}, origin
    for name, seconds in zip(("source_arrival", "drain", "padding"), lengths):
        right = min(cutoff, left + seconds * 10**9)
        segments[name] = phase(left, right, penalty=True) if right > left else phase(left, right)
        segments[name]["declared_contract_seconds"] = seconds
        segments[name]["observed_contract_seconds"] = (right - left) / 1e9
        segments[name]["expected_arrivals"] = {kind: sum(r["job_type"] == kind and
            (left - origin) / 1e9 <= r["arrival_s"] < (right - origin) / 1e9 for r in rows) for kind in ("online", "offline")}
        left = right
    full = phase(first_ns, max(cutoff, end_ns), penalty=True)
    window = phases["window"]
    _require(math.isclose(sum(s["total_cost"] for s in segments.values()), window["total_cost"], abs_tol=1e-9), "segment cost conservation failed")
    _require(math.isclose(sum(s["total_cost"] for s in phases.values()), full["total_cost"], abs_tol=1e-9), "deployment cost conservation failed")
    limits = contract["feasibility"]
    feasible = bool(populations["online"]["arrivals"] and populations["offline"]["arrivals"] and
                    populations["online"]["timely_rate"] >= 1 - limits["max_online_violation_rate"] and
                    populations["offline"]["timely_rate"] >= limits["min_offline_deadline_completion_rate"])
    eligibility = []
    if not binding["bound"]:
        eligibility.append("contract_not_bound_in_historical_plan")
    if any(key not in plan for key in ("source_arrival_seconds", "drain_seconds", "padding_seconds")):
        eligibility.append("segment_lengths_not_explicit_in_plan")
    if package in ("W1", "W4") and "qwen" not in str(plan.get("model", "")).lower():
        eligibility.append("W1_W4_model_is_not_Qwen")
    return {"schema": "maxopt-v4-formal-accounting-v2", "run_id": plan.get("run_id"), "nonformal_smoke": smoke,
        "formal_accepted": False, "acceptance_scope": "derived_accounting_only_requires_separate_formal_CPU_acceptance",
        "input_sha256": {name: _sha(out / name) for name in ("controller_events.jsonl", "request_events.jsonl", "expected_requests.json")},
        "contract_binding": binding, "origin_monotonic_ns": origin, "cutoff_monotonic_ns": cutoff,
        "audit": {"accounting_complete": not issues, "issues": sorted(set(issues)), "formal_eligibility_issues": eligibility,
                  "raw_projection_consistent": True, "observed_window_seconds": max(0, (observed_end - origin) / 1e9),
                  "full_deployment_observed_to_s": (end_ns - origin) / 1e9,
                  "resource_cost_is_observed_lower_bound": "generation_release_missing" in issues,
                  "operator_abort": aborted, "runtime_technical_failures": failures},
        "metrics": {"populations": populations, "completed": sum(p["completed"] for p in populations.values()),
                    "unfinished": sum(p["unfinished"] for p in populations.values()),
                    "censored": sum(p["censored"] for p in populations.values()), "abort": aborted,
                    "api_accepted": window["api_accepted"], "gpu_occupied_seconds": window["occupied_seconds"],
                    "gpu_active_seconds": window["active_seconds"], "startup_events": window["startup_events"],
                    "shutdown_events": window["shutdown_events"], "contract_feasible": feasible,
                    "offline_all_timely": populations["offline"]["arrivals"] > 0 and populations["offline"]["not_on_time"] == 0,
                    "offline_penalties_assessed": len(misses), "offline_unfinished_deadline_after_smoke_cutoff": pending_deadlines},
        "accounting": {"unit": "scenario_USD", "prices": {k: prices[k] for k in PRICE_KEYS},
            "price_units": {"gpu_second": "scenario_USD/GPU-second", "startup_event": "scenario_USD/event",
                "shutdown_event": "scenario_USD/event", "api_input_token": "scenario_USD/accepted-input-token",
                "api_output_token": "scenario_USD/accepted-output-budget-token", "offline_deadline_miss": "scenario_USD/not-on-time-offline-request"},
            "occupied_definition": "actual launch_issued to verified_release; missing release clipped to last observed event",
            "active_definition": "actual lifecycle active state; not hardware compute utilization",
            "quantile_definition": "linear interpolation at (n-1)*q over completed requests only; failures retained in all-arrival denominator",
            "penalty_attribution": "offline deadline timestamp; cutoff deadline included in final nonempty segment",
            "phases": phases, "segments": segments, "full_deployment": full,
            "window_operating_cost": window["operating_cost"], "window_total_cost": window["total_cost"],
            "full_deployment_operating_cost": full["operating_cost"], "full_deployment_total_cost": full["total_cost"]},
        "requests": requests, "generations": generation_rows}
