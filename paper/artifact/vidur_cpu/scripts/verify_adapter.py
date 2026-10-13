"""Map native Vidur request records to the unchanged FidelityLoop request evaluator.

All times are simulated seconds. No physical accuracy, lifecycle or cost claim.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "vendor/fidelityloop_metrics.py"
_spec = importlib.util.spec_from_file_location("fl_request_metrics", METRICS)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def identity(value):
    if type(value) not in (str, int) or value == "":
        raise ValueError("invalid native request identity")
    return str(value)

def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("nonfinite or nonnumeric " + name)
    return value

def token_count(value):
    value = finite(value, "token count")
    if value <= 0 or int(value) != value:
        raise ValueError("token count must be a positive integer")
    return int(value)

def index_unique(rows):
    indexed = {}
    for row in rows:
        rid = identity(row["id"])
        if rid in indexed:
            raise ValueError("duplicate native identity")
        indexed[rid] = row
    return indexed

def adapt(inputs, outputs, *, horizon, relative_deadline_seconds):
    finite(horizon, "horizon"); finite(relative_deadline_seconds, "deadline")
    if horizon <= 0 or relative_deadline_seconds < 0:
        raise ValueError("invalid horizon or deadline")
    expected, observed = index_unique(inputs), index_unique(outputs)
    if not expected or set(expected) != set(observed):
        raise ValueError("missing or extra native output identity")
    rows, records = [], {}
    excluded, clipped = 0, 0
    for rid, source in expected.items():
        record = observed[rid]
        arrival = finite(source["arrived_at"], "arrival")
        if arrival < 0 or finite(record["arrived_at"], "output arrival") != arrival:
            raise ValueError("arrival changed or negative")
        for key in ("num_prefill_tokens", "num_decode_tokens"):
            if token_count(source[key]) != token_count(record[key]):
                raise ValueError("token budget changed or invalid")
        if type(record["completed"]) is not bool:
            raise ValueError("explicit completion flag required")
        raw_done = finite(record["completed_at"], "completion")
        if record["completed"]:
            if raw_done < arrival:
                raise ValueError("completion precedes arrival")
            scheduled = finite(record["scheduled_at"], "scheduled time")
            if not arrival <= scheduled <= raw_done:
                raise ValueError("invalid native scheduling order")
            if record["num_processed_tokens"] != source["num_prefill_tokens"] + source["num_decode_tokens"]:
                raise ValueError("incomplete token count marked complete")
            done = raw_done
        else:
            if raw_done != 0:
                raise ValueError("unrecognized unfinished completion sentinel")
            done = None
        if arrival >= horizon:
            excluded += 1
            continue
        if done is not None and done > horizon:
            clipped += 1
            done = None
        rows.append({"request_id": rid, "job_type": "online", "arrival_s": arrival,
                     "deadline_s": arrival + relative_deadline_seconds,
                     "input_tokens": token_count(source["num_prefill_tokens"]),
                     "max_output_tokens": token_count(source["num_decode_tokens"])})
        records[rid] = {"completed_at": done}
    return rows, records, {"all_native_requests": len(expected), "included_arrivals": len(rows),
                           "excluded_at_or_after_cutoff": excluded,
                           "future_completions_redacted": clipped}

def independent_counts(rows, records, horizon):
    # Direct predicates, without calling or copying the evaluator implementation.
    completed = [r for r in rows if records[r["request_id"]]["completed_at"] is not None
                 and records[r["request_id"]]["completed_at"] <= horizon]
    timely = [r for r in completed if records[r["request_id"]]["completed_at"] <= r["deadline_s"]]
    return {"arrivals": len(rows), "completed": len(completed), "on_time": len(timely),
            "late": len(completed) - len(timely), "unfinished": len(rows) - len(completed)}

def evaluate(inputs, outputs, view):
    rows, records, mapping = adapt(inputs, outputs, horizon=view["horizon"],
                                  relative_deadline_seconds=view["relative_deadline_seconds"])
    metrics = _module.evaluate_requests(rows, records, horizon=view["horizon"])
    expected = independent_counts(rows, records, view["horizon"])
    if any(metrics["populations"]["online"][key] != value for key, value in expected.items()):
        raise AssertionError("independent deadline tally differs")
    return {"view": view, "mapping": mapping, "request_metrics": metrics,
            "P1plus_applicable": False,
            "P1plus_note": "Original paper requires 120 offline requests; this online-only demo is ineligible. False is not a failed policy test.",
            "independent_tally": expected, "agreement": True,
            "scope": "simulated request-output interoperability only",
            "lifecycle_occupied_seconds": None, "scenario_cost": None}, rows, records

def run(case_dir, out):
    case_dir, out = Path(case_dir), Path(out)
    source = json.loads((ROOT / "vendor/SOURCE.json").read_text())
    if hashlib.sha256(METRICS.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("vendored FidelityLoop evaluator changed")
    ip, op = case_dir / "input_requests.jsonl", case_dir / "request_records.jsonl"
    inputs, outputs = read_jsonl(ip), read_jsonl(op)
    protocol = json.loads((ROOT / "ADAPTER_PROTOCOL.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for view in protocol["views"]:
        summary, rows, records = evaluate(inputs, outputs, view)
        summaries.append(summary)
        (out / (view["name"] + ".json")).write_text(json.dumps({"summary":summary,"rows":rows,"records":records},indent=2,allow_nan=False)+"\n")
    result = {"status":"PASS", "input_sha256":hashlib.sha256(ip.read_bytes()).hexdigest(),
              "output_sha256":hashlib.sha256(op.read_bytes()).hexdigest(),
              "evaluator_sha256":source["sha256"],"views":summaries}
    (out / "ADAPTER_RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--case",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();result=run(a.case,a.out)
    print(json.dumps({"status":result["status"],"counts":[x["independent_tally"] for x in result["views"]]},indent=2))
