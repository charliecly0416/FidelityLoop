"""Run the small CPU validation included with FidelityLoop.

This exercise checks conservation and policy-adapter behavior. It is intentionally
separate from the paper's A100 campaign; no GPU, network, or private trace is used.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from fidelityloop.framework import (
    FrameworkConfig, FrameworkSimulator, ConstantServiceEstimator,
    TargetPolicyAdapter, PPOActionAdapter,
)
from fidelityloop.legacy.maxopt_v2.workload import FIELDS


def rows():
    out=[]
    for i, kind in enumerate(("offline", "online", "offline", "online", "offline")):
        row=dict.fromkeys(FIELDS)
        row.update(request_id=f"smoke-{i}", source_sha256="synthetic", source_row_ordinal=i+1,
                   source_occurrence=0, raw_row_hash=f"row-{i}", raw_timestamp_s=i,
                   arrival_s=i, sim_tick=i, wall_monotonic_s=None,
                   job_type=kind, input_tokens=32, prompt_token_ids=[1]*32,
                   max_output_tokens=32, deadline_s=i+(8 if kind=="online" else 12),
                   split="validation", window_id="bounded_cpu_smoke", evidence={},
                   prompt_source_sha256="synthetic", prompt_offset=0)
        out.append(row)
    return out


def run():
    data=rows(); checks=[]
    for devices, slots in ((2,1),(2,2),(3,1)):
        cfg=FrameworkConfig.synthetic(window_seconds=12, drain_seconds=8,
                                      device_count=devices, local_slots=slots,
                                      initial_active_devices=devices,
                                      gpu_second_price=1.0, offline_miss_price=1.0)
        policy=TargetPolicyAdapter("allN" if devices > 2 else "all2", max_devices=devices)
        result=FrameworkSimulator(data, cfg, estimator=ConstantServiceEstimator(1.0, slots), policy=policy).run()
        summary=result["summary"]
        checks.append({"devices":devices,"slots":slots,"arrived":summary["arrived"],
                       "completed":summary["completed"],"censored":summary["censored"],
                       "request_conservation":summary["request_conservation"],
                       "state_conservation":summary["state_conservation"],
                       "cost_conservation":summary["cost_conservation"]})
    obs={"time":0,"queue_online":[],"queue_offline":[],
         "devices":[{"state":"active","jobs":[]},{"state":"off","jobs":[]}]}
    action=PPOActionAdapter(1, max_devices=2).decide(obs)
    assert action["target"] == 2
    assert all(c["request_conservation"] and c["state_conservation"] and c["cost_conservation"] for c in checks)
    return {"schema":"fidelityloop-cpu-validation-v1","status":"PASS",
            "checks":checks,"ppo_adapter_action":action,
            "scope":["CPU bounded facade","synthetic workload","no GPU or network"],
            "excluded":["heterogeneous-GPU accuracy","new PPO training","private trace redistribution"]}


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=Path("results/VALIDATION.json")); args=ap.parse_args()
    result=run(); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
