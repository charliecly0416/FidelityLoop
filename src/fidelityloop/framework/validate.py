"""Produce bounded CPU validation evidence for the framework facade."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from fidelityloop.legacy.maxopt_v2.workload import FIELDS
from fidelityloop.legacy.maxopt_v3.simulator import Simulator as HistoricalSimulator
try:
    from .config import FrameworkConfig
    from .estimators import ConstantServiceEstimator
    from .policies import TargetPolicyAdapter, GuardPolicyAdapter
    from .runtime import FrameworkSimulator
except ImportError:
    from fidelityloop.framework.config import FrameworkConfig
    from fidelityloop.framework.estimators import ConstantServiceEstimator
    from fidelityloop.framework.policies import TargetPolicyAdapter, GuardPolicyAdapter
    from fidelityloop.framework.runtime import FrameworkSimulator


def _rows():
    rows=[]
    for i,(arrival,kind) in enumerate(((0,"offline"),(1,"online"),(2,"offline"))):
        r=dict.fromkeys(FIELDS)
        r.update(request_id=f"framework-{i}", source_sha256="synthetic", source_row_ordinal=i+1,
                 source_occurrence=0, raw_row_hash="synthetic", raw_timestamp_s=arrival,
                 arrival_s=arrival, sim_tick=arrival, job_type=kind, input_tokens=128,
                 prompt_token_ids=[1]*128, max_output_tokens=128,
                 deadline_s=arrival+(300 if kind=="offline" else 60), split="validation",
                 window_id="framework_validation", evidence={}, prompt_source_sha256="synthetic",
                 prompt_offset=0)
        rows.append(r)
    return rows


def _model():
    lookup={f"grid_i{i}_o{o}_c{c}":o/100+i/10000 for i in (128,512,2048) for o in (32,128,256) for c in (1,4)}
    lookup["core_i256_o256_c4"]=2.5856
    return dict(kind="n4c_progress_lookup", lookup_wall_seconds=lookup,
                startup_seconds=5., shutdown_seconds=4., setup_seconds=5.)


def _contract(root):
    p=root/"artifact/bridge_execution/inputs/simulator_contract.json"
    c=json.loads(p.read_text(encoding="utf-8")); c["workload_revision"].update(window_seconds=20, drain_seconds=10)
    return c


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build(root, output):
    rows, contract = _rows(), _contract(root)
    model = _model()
    policy={"name":"all2"}
    cases = [
        ("C_all2", _model, {"name": "all2"}, {}, None, False, False),
        ("C_all1", _model, {"name": "all1"}, {}, None, False, False),
        ("C_hysteresis", _model, {"name": "hysteresis", "up_threshold": 8, "down_threshold": 0, "cooldown_seconds": 60, "min_devices": 0}, {}, None, False, False),
        ("C_guard_hysteresis", _model, {"name": "hysteresis", "up_threshold": 8, "down_threshold": 0, "cooldown_seconds": 60, "min_devices": 0}, {"startup_safe_seconds": 8.0, "startup_max_seconds": 15.0, "shutdown_seconds": 4.0}, _model, True, True),
    ]
    historical = None; config_evidence = []
    for name, case_model, case_policy, safety, guard_model, scale, reservation in cases:
        expected = HistoricalSimulator(rows, contract, case_model(), case_policy, safety=safety, guard_model=guard_model() if guard_model else None, scale=scale, reservation=reservation).run()
        got = FrameworkSimulator.historical(rows, contract, case_model(), TargetPolicyAdapter(case_policy), safety=safety, guard_model=guard_model() if guard_model else None, scale=scale, reservation=reservation).run()
        if expected != got: raise AssertionError("historical facade changed default output: " + name)
        config_evidence.append({"name": name, "exact": True, "requests_sha256": _sha(expected["requests"]), "events_sha256": _sha(expected["events"]), "summary_sha256": _sha(expected["summary"]), "event_count": len(expected["events"])})
        if historical is None: historical = expected
    exact = all(item["exact"] for item in config_evidence)
    extensions=[]
    for name, devices, slots in (("two_devices_two_slots",2,2),("three_devices_one_slot",3,1)):
        cfg=FrameworkConfig.synthetic(window_seconds=8, drain_seconds=8, device_count=devices,
                                      local_slots=slots, initial_active_devices=devices)
        result=FrameworkSimulator(rows, cfg, estimator=ConstantServiceEstimator(1.0, slots),
                                  policy=TargetPolicyAdapter("allN", max_devices=devices)).run()
        s=result["summary"]
        if not (s["request_conservation"] and s["state_conservation"] and s["cost_conservation"]):
            raise AssertionError(name+" conservation failed")
        extensions.append({"name":name,"device_count":devices,"local_slots":slots,
                           "requests":s["arrived"],"completed":s["completed"],"censored":s["censored"],
                           "request_conservation":s["request_conservation"],
                           "state_conservation":s["state_conservation"],"cost_conservation":s["cost_conservation"],
                           "events_sha256":_sha(result["events"]),"summary_sha256":_sha(s)})
    out={"schema":"maxopt-framework-validation-v1","status":"PASS",
         "historical_default":{"exact":exact,"configurations":config_evidence},"extensions":extensions,
         "scope":["CPU-only facade and synthetic bounded configurations","historical two-device default exact replay"],
         "excluded":["heterogeneous GPU accuracy","new policy training/tuning","arbitrary policy/device compatibility"]}
    output.mkdir(parents=True, exist_ok=True)
    (output/"VALIDATION.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    return out

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[3]); parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args(); out=args.output or args.root/"results/facade"
    print(json.dumps(build(args.root,out),ensure_ascii=False,indent=2))
