"""Standard-library reader and accounting for the released contract-level traces."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read_index(root=ROOT):
    return json.loads((Path(root) / "INDEX.json").read_text())

def load_trace(entry, root=ROOT):
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("record path escapes corpus")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("record checksum mismatch: " + entry["record_id"])
    record = json.loads(gzip.decompress(data))
    if record["record_id"] != entry["record_id"] or record["schema"] != "fidelityloop-trace-v1":
        raise ValueError("record identity/schema mismatch")
    for key in ("policy", "window", "replay_model", "repeat"):
        if key in entry and record[key] != entry[key]:
            raise ValueError("index/record coordinate mismatch: " + key)
    return record

def measure(record):
    """Recompute window feasibility and scenario cost from projected primitives.

    Release verification and simulator event semantics are inherited from the
    archived executions. This function does not rerun a policy or serving engine.
    """
    horizon = record["horizon_s"]
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("invalid observation horizon")
    req = {x["request_id"]: x for x in record["requests"]}
    if len(req) != len(record["requests"]) or not req:
        raise ValueError("duplicate or empty request population")
    populations = {k: {"arrivals": 0, "on_time": 0, "completed": 0, "unfinished": 0} for k in ("online", "offline")}
    for x in req.values():
        if not 0 <= x["arrival_s"] <= x["deadline_s"] or x["arrival_s"] >= horizon:
            raise ValueError("invalid arrival/deadline")
        for k in ("input_tokens", "max_output_tokens"):
            if not isinstance(x[k], int) or x[k] < 0:
                raise ValueError("invalid token budget")
        end = x["completed_at_s"]
        if end is not None and (not math.isfinite(end) or end < x["arrival_s"]):
            raise ValueError("invalid completion timestamp")
        completed = x["status"] == "completed" and end is not None and end <= horizon
        on_time = completed and end <= x["deadline_s"]
        group = populations[x["job_type"]]
        group["arrivals"] += 1
        group["completed"] += int(completed)
        group["on_time"] += int(on_time)
        group["unfinished"] += int(not completed)
    occupied = 0.0
    intervals = {}
    for x in record["resource_intervals"]:
        start, end = x["start_s"], x["end_s"]
        if not all(math.isfinite(t) for t in (start, end)) or end < start:
            raise ValueError("invalid occupation interval")
        intervals.setdefault(x["gpu"], []).append((start, end))
        occupied += max(0.0, min(horizon, end) - max(0.0, start))
    for per_gpu in intervals.values():
        ordered = sorted(per_gpu)
        if any(a[1] > b[0] + 1e-8 for a, b in zip(ordered, ordered[1:])):
            raise ValueError("overlapping occupation intervals on one GPU")
    prices = record["prices"]
    for v in prices.values():
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError("invalid price")
    costs = {"gpu": occupied * prices["gpu_second"], "startup": 0.0, "shutdown": 0.0, "synthetic_api": 0.0}
    api_ids = set()
    for e in record["ledger_events"]:
        if not math.isfinite(e["time_s"]):
            raise ValueError("invalid ledger timestamp")
        if not 0 <= e["time_s"] < horizon:
            continue
        if e["kind"] == "start":
            costs["startup"] += prices["startup_event"]
        elif e["kind"] == "stop":
            costs["shutdown"] += prices["shutdown_event"]
        elif e["kind"] == "api_accept":
            rid = e["request_id"]
            if rid in api_ids or rid not in req:
                raise ValueError("duplicate or unknown API request")
            api_ids.add(rid)
            x = req[rid]
            costs["synthetic_api"] += x["input_tokens"] * prices["api_input_token"] + x["max_output_tokens"] * prices["api_output_token"]
        else:
            raise ValueError("unknown ledger event")
    off, online = populations["offline"], populations["online"]
    costs["offline_miss"] = (off["arrivals"] - off["on_time"]) * prices["offline_deadline_miss"]
    p1plus = all(x["on_time"] == x["arrivals"] for x in populations.values())
    p1 = off["on_time"] == off["arrivals"] and online["on_time"] >= 0.99 * online["arrivals"] and sum(x["unfinished"] for x in populations.values()) == 0
    return {"populations": populations, "P1plus": p1plus, "P1": p1, "gpu_occupied_seconds": occupied, "costs": costs, "cost_total": sum(costs.values())}

def check_reference(record, measured):
    for k in ("P1plus", "P1", "gpu_occupied_seconds", "cost_total"):
        expected = record["reference"].get(k)
        if expected is None:
            continue
        actual = measured[k]
        good = actual == expected if isinstance(expected, bool) else math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)
        if not good:
            raise ValueError(f"{record['record_id']}: {k}: {actual} != {expected}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_id", nargs="?")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    index = read_index(args.root)
    entries = [e for c in index["campaigns"] for k in ("physical", "predictions") for e in c[k]]
    if args.record_id is None:
        print(json.dumps({"campaigns": [c["campaign"] for c in index["campaigns"]], "records": len(entries)}, indent=2))
        return
    entry = next(e for e in entries if e["record_id"] == args.record_id)
    record = load_trace(entry, args.root)
    measured = measure(record)
    check_reference(record, measured)
    print(json.dumps(measured, indent=2))

if __name__ == "__main__":
    main()
