"""Examples: feasibility-direction agreement and scenario-cost MAE, by campaign/model."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from load_trace import ROOT, read_index, load_trace, measure, check_reference


def evaluate(root=ROOT):
    index = read_index(root)
    groups = []
    for campaign in index["campaigns"]:
        entries = {e["record_id"]: e for k in ("physical", "predictions") for e in campaign[k]}
        results = {}
        requests = {}
        for rid, entry in entries.items():
            record = load_trace(entry, root)
            measured = measure(record)
            check_reference(record, measured)
            results[rid] = measured
            requests[rid] = {r["request_id"]: (r["job_type"], r["arrival_s"], r["deadline_s"], r["input_tokens"], r["max_output_tokens"]) for r in record["requests"]}
        expected_pairs = {(p["record_id"], q["record_id"]) for p in campaign["physical"] for q in campaign["predictions"] if (p["policy"], p["window"]) == (q["policy"], q["window"])}
        actual_pairs = {(x["physical_id"], x["prediction_id"]) for x in campaign["pairs"]}
        if actual_pairs != expected_pairs:
            raise ValueError("pair index does not cover matching policy/window records")
        by_model = defaultdict(list)
        paired = set()
        for pair in campaign["pairs"]:
            phy, pred = pair["physical_id"], pair["prediction_id"]
            if (phy, pred) in paired:
                raise ValueError("duplicate physical/prediction pair")
            paired.add((phy, pred))
            if requests[phy] != requests[pred]:
                raise ValueError("paired request populations or budgets differ")
            # E3's frozen comparison excludes its two capacity anchors.
            if campaign["campaign"] == "E3" and entries[phy]["policy"] == "all2":
                continue
            by_model[entries[pred]["replay_model"]].append((results[phy], results[pred]))
        for model, pairs in sorted(by_model.items()):
            n = len(pairs)
            groups.append({"campaign": campaign["campaign"], "replay_model": model,
                "comparison_cells": n,
                "feasibility_agreement": sum(p["P1plus"] == q["P1plus"] for p, q in pairs),
                "false_acceptance_cells": sum(not p["P1plus"] and q["P1plus"] for p, q in pairs),
                "false_rejection_cells": sum(p["P1plus"] and not q["P1plus"] for p, q in pairs),
                "cost_mae_scenario_USD": sum(abs(p["cost_total"] - q["cost_total"]) for p, q in pairs) / n})
    return {"scope": "Per-campaign descriptive paired comparisons. All accepted physical cells except E3 anchors. Repeats reuse predictions; these are not independent prediction counts or population screening accuracy. V3 covers all released policies; the paper's fixed-H fidelity analyses use a subset.", "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--metric", choices=("feasibility", "cost", "both"), default="both")
    args = parser.parse_args()
    out = evaluate(args.root)
    if args.metric != "both":
        for row in out["groups"]:
            keys = [k for k in row if (k.startswith("cost_") if args.metric == "feasibility" else k in ("feasibility_agreement", "false_acceptance_cells", "false_rejection_cells"))]
            for key in keys:
                del row[key]
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
