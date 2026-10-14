"""Recompute the E5/E6 table from reduced per-cell comparisons (standard library)."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def main():
    bindings = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())
    for name, expected in bindings["files"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    table = []
    checks = []
    for campaign in ("e5", "e6"):
        with (ROOT / campaign / "POLICY_CELL_COMPARISONS.csv").open() as f:
            rows = list(csv.DictReader(f))
        with (ROOT / campaign / "MODEL_SUMMARY.csv").open() as f:
            summaries = list(csv.DictReader(f))
        for summary in summaries:
            model = summary["model"]
            group = [r for r in rows if r["model"] == model]
            if len({r["run_id"] for r in group}) != len(group):
                raise ValueError("Duplicated physical run within a model")
            if any(r[k] not in ("True", "False") for r in group
                   for k in ("physical_P1plus", "predicted_P1plus")):
                raise ValueError("Invalid boolean")
            passed = sum(r["physical_P1plus"] == "True" for r in group)
            failed = len(group) - passed
            correct = sum(r["physical_P1plus"] == r["predicted_P1plus"] for r in group)
            fa = sum(r["physical_P1plus"] == "False" and r["predicted_P1plus"] == "True" for r in group)
            fr = sum(r["physical_P1plus"] == "True" and r["predicted_P1plus"] == "False" for r in group)
            if len(group) != int(summary["denominator"]) or correct != int(summary.get("matches", summary.get("agreement"))):
                raise ValueError("Feasibility aggregation mismatch")
            if fr != int(summary["false_rejects"]) or ("false_accepts" in summary and fa != int(summary["false_accepts"])):
                raise ValueError("Error-direction mismatch")
            unique = {(r["policy"], r["window"]) for r in group}
            if len(unique) != (12 if campaign == "e5" else 4):
                raise ValueError("Unexpected prediction population")
            for policy, window in unique:
                pair = [r for r in group if (r["policy"], r["window"]) == (policy, window)]
                if len(pair) != 2 or len({r["predicted_P1plus"] for r in pair}) != 1:
                    raise ValueError("Repeats must share one frozen prediction")
            fields = ("scenario_cost_absolute_error", "occupied_absolute_error_gpu_seconds", "occupied_trajectory_l1_gpu_seconds")
            means = {}
            for field in fields:
                value = mean(mean(float(r[field]) for r in group if r["window"] == window)
                             for window in sorted({r["window"] for r in group}))
                if not math.isclose(value, float(summary[field]), abs_tol=1e-9):
                    raise ValueError(f"Equal-window MAE mismatch: {campaign}/{model}/{field}")
                means[field] = value
            fa_text = f"{fa}/{failed}" if failed else "---"
            # Match the reader-facing display names used by the paper table.
            # The CSV keeps the archived model IDs; this mapping is presentation only.
            study = "Arrivals" if campaign == "e5" else "14B"
            labels = {"C": "Lookup-69", "C30": "Lookup-30", "C_sg": "Lookup-SG",
                      "S14": "14B-Hist", "L14": "14B-Local"}
            label = labels[model]
            table.append(f"{study} & {label} & {correct}/{len(group)} & {fa_text} & {fr}/{passed} & {means[fields[0]]:.4f} & {means[fields[1]]:.2f} & {means[fields[2]]:.2f}" + chr(92)*2)
            checks.append({"campaign": campaign, "model": model, "correct": correct,
                           "cells": len(group), "false_accepts": fa, "false_rejects": fr})
    generated = ROOT.parents[1] / "generated/e5e6_rows.tex"
    if generated.read_text() != "\n".join(table) + "\n" + chr(92) + "bottomrule\n":
        raise ValueError("LaTeX table does not match reduced evidence")
    models = json.loads((ROOT / "e6/PREDICTOR_PARAMETERS.json").read_text())["models"]
    if {k:v for k,v in models["S14"].items() if k != "startup_seconds"} != {k:v for k,v in models["L14"].items() if k != "startup_seconds"}:
        raise ValueError("S14 and L14 differ beyond startup_seconds")
    print(json.dumps({"status": "PASS", "model_summaries": checks,
                      "single_field_contrast": True,
                      "scope": "Reduced comparison aggregation; not raw GPU execution verification"}, indent=2))


if __name__ == "__main__":
    main()
