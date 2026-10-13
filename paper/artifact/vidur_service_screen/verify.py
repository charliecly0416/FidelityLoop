"""Recompute service-screen errors from reduced per-request measurements (stdlib only)."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent

def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)

def main():
    manifest = json.loads((ROOT / "FILES_SHA256.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    expected = json.loads((ROOT / "evaluation.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    frozen = {g["group_id"]: g for g in json.loads((ROOT / "service_cases_public.json").read_text())["groups"]}
    with (ROOT / "request_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 90 and len({r["request_id"] for r in rows}) == 90
    grouped, request_errors, condition_errors, spans = {}, [], {}, {}
    for row in rows:
        obs, pred = float(row["observed_latency_s"]), float(row["predicted_latency_s"])
        assert all(math.isfinite(x) and x > 0 for x in (obs, pred))
        request_errors.append(abs(pred / obs - 1))
        assert math.isclose(float(row["predicted_completion_s"]) - float(row["arrival_s"]), pred, abs_tol=1e-12)
        grouped.setdefault(row["group_id"], []).append(row)
    assert grouped.keys() == frozen.keys()
    batch_errors = []
    for group_id, group in grouped.items():
        assert len(group) == frozen[group_id]["request_count"]
        row = group[0]
        observed = float(row["observed_makespan_s"])
        predicted = float(row["predicted_makespan_s"])
        assert math.isclose(max(float(r["arrival_s"])+float(r["observed_latency_s"]) for r in group), observed, abs_tol=1e-9)
        assert math.isclose(max(float(r["predicted_completion_s"]) for r in group), predicted, abs_tol=1e-12)
        error = abs(predicted / observed - 1)
        batch_errors.append(error)
        condition_errors.setdefault(row["condition"], []).append(error)
        spans.setdefault(row["condition"], []).append(observed)
    assert len(grouped) == 36 and len(condition_errors) == 12
    assert all(len(v) == 3 for v in spans.values())
    metrics = {
        "request_latency_median_ape": median(request_errors),
        "request_latency_p90_ape": percentile(request_errors, .9),
        "batch_makespan_median_ape": median(batch_errors),
        "batch_makespan_p90_ape": percentile(batch_errors, .9),
        "max_condition_median_ape": max(median(v) for v in condition_errors.values()),
    }
    for key, value in metrics.items():
        assert math.isclose(value, expected["metrics"][key], rel_tol=0, abs_tol=1e-12), key
    gate = protocol["gate"]
    failed = [k for k,v in metrics.items() if v > gate[k if k == "max_condition_median_ape" else k+"_max"]]
    jitter = {k:max(v)/min(v) for k,v in spans.items() if max(v)/min(v)>gate["repeat_makespan_max_min_ratio_warn"]}
    status = "STOP_NOT_USEFUL_FOR_POLICY_VALIDATION" if failed else ("INCONCLUSIVE_STABILITY" if jitter else "PASS_SERVICE_SCREEN_ONLY")
    assert status == expected["status"] and failed == expected["failed_criteria"] and jitter == expected["jitter_conditions"]
    print(json.dumps({"status":status,"requests":len(rows),"groups":len(grouped),"metrics":metrics},indent=2))

if __name__ == "__main__":
    main()
