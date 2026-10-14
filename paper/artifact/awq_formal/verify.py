"""Recompute the AWQ startup contrast from frozen predictions and reduced ledgers.

Standard library only. This verifies reduced evidence, not raw GPU execution.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
MODELS = ("AWQ-Hist", "AWQ-30", "AWQ-Local")


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)


def strict_feasibility(account):
    """P1+ includes unfinished requests; contract_feasible alone does not."""
    populations = account["metrics"]["populations"]
    assert set(populations) == {"online", "offline"}
    for kind, pop in populations.items():
        requests = [r for r in account["requests"].values() if r["job_type"] == kind]
        assert len(requests) == pop["arrivals"] > 0
        timely = 0
        for request in requests:
            completed = request["completed_at"]
            on_time = (request["status"] == "completed" and completed is not None
                       and completed <= min(request["deadline_s"], 2100))
            assert on_time == request["timely"]
            timely += on_time
        assert timely == pop["timely"]
    return all(p["timely"] == p["arrivals"] for p in populations.values())


def recompute(root=ROOT):
    receipt = read(root / "CPU_PREDICTION_RECEIPT.json")
    summary = read(root / "SUMMARY.json")
    assert receipt["status"] == "PASS" and not receipt["selection_uses_outcome"]
    assert len(receipt["cells"]) == receipt["prediction_count"] == 18
    models = receipt["profile"]["models"]
    shared = [{k: v for k, v in models[m].items() if k != "startup_seconds"} for m in MODELS]
    assert shared[0] == shared[1] == shared[2]
    predictions = {}
    for p in receipt["cells"]:
        path = root / p["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == p["sha256"], path
        raw = json.loads(gzip.decompress(path.read_bytes()))
        assert p["P1plus"] == raw["request_metrics"]["P1plus"]
        assert p["P1plus"] == all(x["on_time"] == x["arrivals"]
                                  for x in raw["request_metrics"]["populations"].values())
        close(p["gpu_occupied_seconds"], raw["event_audit"]["gpu_occupied_seconds"])
        close(p["costs"]["total"], raw["costs"]["total"])
        key = (p["prediction_model"], p["policy_id"], p["window_id"])
        assert key not in predictions
        predictions[key] = p
    assert len(summary["cells"]) == 12
    pairs = {m: [] for m in MODELS}
    seen = set()
    anchors = 0
    for cell in summary["cells"]:
        run = cell["run_id"]
        assert run not in seen
        seen.add(run)
        assert cell["status"] == "VALID" and len(cell["attempts"]) == 1
        account = read(root / "accounting" / (run + "__attempt1.json"))
        assert account["run_id"] == run
        assert account["audit"]["accounting_complete"] and not account["audit"]["issues"]
        physical = strict_feasibility(account)
        assert physical == cell["request_metrics"]["P1plus"]
        occupied = account["accounting"]["phases"]["window"]["occupied_seconds"]
        cost = account["accounting"]["window_total_cost"]
        close(occupied, cell["gpu_occupied_seconds"])
        close(cost, cell["costs"]["total"])
        if cell["kind"] == "CAPACITY":
            assert cell["policy_id"] == "all2" and physical
            anchors += 1
            continue
        assert cell["kind"] == "COMPARISON"
        for model in MODELS:
            p = predictions[model, cell["policy_id"], cell["window_id"]]
            pairs[model].append({
                "run_id": run, "policy_id": cell["policy_id"], "window_id": cell["window_id"],
                "predicted_P1plus": p["P1plus"], "physical_P1plus": physical,
                "occupied_absolute_error_gpu_seconds": abs(p["gpu_occupied_seconds"] - occupied),
                "scenario_cost_absolute_error": abs(p["costs"]["total"] - cost),
            })
    assert anchors == 4
    results = {}
    for model, rows in pairs.items():
        assert len(rows) == 8
        windows = sorted({r["window_id"] for r in rows})
        assert len(windows) == 2 and all(sum(r["window_id"] == w for r in rows) == 4 for w in windows)
        results[model] = {
            "startup_seconds": models[model]["startup_seconds"],
            "unique_policy_window_predictions": len({(r["policy_id"], r["window_id"]) for r in rows}),
            "denominator": len(rows),
            "correct": sum(r["predicted_P1plus"] == r["physical_P1plus"] for r in rows),
            "false_accepts": sum(r["predicted_P1plus"] and not r["physical_P1plus"] for r in rows),
            "false_rejects": sum(not r["predicted_P1plus"] and r["physical_P1plus"] for r in rows),
            "occupied_mae_gpu_seconds": mean(r["occupied_absolute_error_gpu_seconds"] for r in rows),
            "cost_mae_scenario_usd": mean(r["scenario_cost_absolute_error"] for r in rows),
            "false_reject_run_ids": [r["run_id"] for r in rows if not r["predicted_P1plus"] and r["physical_P1plus"]],
        }
        published = summary["model_comparisons"][model]
        feasible = sum(r["physical_P1plus"] for r in rows)
        f = published["feasibility"]
        assert f["correct"] == results[model]["correct"]
        assert f["denominator"] == len(rows)
        assert f["false_pass"] == results[model]["false_accepts"]
        assert f["false_block"] == results[model]["false_rejects"]
        assert f["actual_feasible_denominator"] == feasible
        assert f["actual_infeasible_denominator"] == len(rows) - feasible
        assert f["accept_feasible"] == feasible - f["false_block"]
        assert f["block_infeasible"] == len(rows) - feasible - f["false_pass"]
        close(f["false_accept_rate"], f["false_pass"] / (len(rows) - feasible))
        close(f["false_reject_rate"], f["false_block"] / feasible)
        close(results[model]["occupied_mae_gpu_seconds"], published["equal_window_means"]["occupied_absolute_error_gpu_seconds"])
        close(results[model]["cost_mae_scenario_usd"], published["equal_window_means"]["scenario_cost_absolute_error"])
    local = summary["model_comparisons"]["AWQ-Local"]
    assert results["AWQ-Local"]["correct"] == local["feasibility"]["correct"]
    close(results["AWQ-Local"]["occupied_mae_gpu_seconds"], local["equal_window_means"]["occupied_absolute_error_gpu_seconds"])
    close(results["AWQ-Local"]["cost_mae_scenario_usd"], local["equal_window_means"]["scenario_cost_absolute_error"])
    return {"schema": "fidelityloop-awq-startup-comparison-v1", "status": "PASS",
            "scope": "Eight policy runs; four predictions per model reused over two repeats; four anchors excluded from means. Reduced evidence verification, not raw GPU execution.",
            "source_receipt_sha256": hashlib.sha256((root / "CPU_PREDICTION_RECEIPT.json").read_bytes()).hexdigest(),
            "model_comparisons": results, "pairs": pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write recomputed comparison JSON")
    args = parser.parse_args()
    result = recompute()
    output = ROOT / "AWQ_PREDICTOR_COMPARISON.json"
    if args.write:
        output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert read(output) == result, "Published comparison differs from recomputation"
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
