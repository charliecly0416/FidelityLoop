"""Verify the complete released corpus with the Python standard library."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from evaluate import evaluate
from load_trace import ROOT, load_trace, read_index

ROOT_KEYS = set("schema record_id campaign kind policy window replay_model repeat horizon_s prices requests resource_intervals ledger_events events decisions reference".split())
ROWS = {
    "requests": set("request_id job_type arrival_s deadline_s input_tokens max_output_tokens status completed_at_s route".split()),
    "resource_intervals": set("gpu generation start_s end_s".split()),
    "ledger_events": set("time_s kind request_id".split()),
    "events": set("time_s kind gpu generation request_id state target".split()),
    "decisions": set("time_s target".split()),
}
EVENT_KINDS = set("api_accept api_completed censored cleanup_complete complete decision dispatch end initial launch launch_issued lifecycle_state local_dispatch observation_end observation_start policy_tick ready release request_complete request_dispatch request_release request_terminal shutdown shutdown_issued state verified_release".split())
SENSITIVE = re.compile(r"/home/|/lustre/|/tmp/|chuliyang|charliecly|prompt_token_ids|raw_timestamp|hostname|controller_monotonic_ns|@gmail\.com|@pku\.edu", re.I)


def check_fields(record):
    if set(record) != ROOT_KEYS:
        raise ValueError("unexpected root fields")
    if record["horizon_s"] != 2100 or record["kind"] not in ("physical", "prediction"):
        raise ValueError("unexpected record scope")
    if set(record["prices"]) != set("gpu_second startup_event shutdown_event api_input_token api_output_token offline_deadline_miss".split()):
        raise ValueError("unexpected price fields")
    if set(record["reference"]) != set("P1 P1plus gpu_occupied_seconds cost_total".split()):
        raise ValueError("unexpected reference fields")
    for key, allowed in ROWS.items():
        for row in record[key]:
            if not set(row) <= allowed or (key != "events" and set(row) != allowed):
                raise ValueError("unexpected fields in " + key)
            if any(isinstance(v, (list, dict)) for v in row.values()):
                raise ValueError("nested payload in projected fields")
            rid = row.get("request_id")
            if rid is not None and re.fullmatch(r"r[0-9a-f]{24}", rid) is None:
                raise ValueError("unmapped request identity")
    for row in record["requests"]:
        if row["status"] not in ("completed", "censored", "failed", "unfinished") or row["route"] not in ("gpu0", "gpu1", "synthetic_api", "none"):
            raise ValueError("unknown request category")
    for row in record["events"]:
        if row["kind"] not in EVENT_KINDS:
            raise ValueError("unreviewed event kind")
        if "state" in row and row["state"] not in ("off", "starting", "active", "draining", "stopping"):
            raise ValueError("unknown lifecycle state")
    ticks = record["decisions"]
    if len(ticks) != 2100 or [x["time_s"] for x in ticks] != list(range(2100)):
        raise ValueError("decision sequence incomplete")
    if any(x["target"] not in (0, 1, 2) for x in ticks):
        raise ValueError("unexpected capacity target")
    if SENSITIVE.search(json.dumps(record)):
        raise ValueError("unexpected identifying/source fields")


def close(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(close(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
    return a == b


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected = {e["path"] for e in manifest["files"]} | {"MANIFEST.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("release inventory mismatch")
    for entry in manifest["files"]:
        p = (root / entry["path"]).resolve()
        if not p.is_relative_to(root) or p.stat().st_size != entry["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("release hash mismatch")
    index = read_index(root)
    if SENSITIVE.search(json.dumps(index)):
        raise ValueError("index exposes source identity")
    counts = {"physical_runs": 0, "unique_predictions": 0, "pairs": 0, "excluded_attempts": 0}
    seen = set()
    for campaign in index["campaigns"]:
        counts["physical_runs"] += len(campaign["physical"])
        counts["unique_predictions"] += len(campaign["predictions"])
        counts["pairs"] += len(campaign["pairs"])
        counts["excluded_attempts"] += len(campaign["excluded_attempts"])
        for entry in campaign["physical"] + campaign["predictions"]:
            if entry["record_id"] in seen:
                raise ValueError("duplicate record identity")
            seen.add(entry["record_id"])
            check_fields(load_trace(entry, root))
    if counts != {"physical_runs": 132, "unique_predictions": 140, "pairs": 372, "excluded_attempts": 6}:
        raise ValueError("unexpected corpus population")
    if not close(evaluate(root), json.loads((root / "SUMMARY.json").read_text())):
        raise ValueError("example summary changed")
    return {"status": "PASS_PAIRED_TRACE_CORPUS", **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))

if __name__ == "__main__": main()
