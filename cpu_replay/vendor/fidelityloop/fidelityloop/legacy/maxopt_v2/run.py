"""Supervised CPU execution of the V2 simulator, with portable result replay."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from .engine import Simulator, TargetPolicy, normalize_config
from .runtime import (RunContext, canonical_v2_source_files, durable_json, file_sha256,
                      no_symlinks, read_journal, canonical_bytes, sha256_bytes)
from .supervisor import Limits, supervise
from .workload import load_window


ROOT = Path(__file__).resolve().parents[2]


def collect_events(run_dir):
    """Recover complete prefixes, then require one contiguous global sequence."""
    events = []
    for path in (Path(run_dir) / "segments").glob("*/events.jsonl"):
        events.extend(read_journal(path, allow_truncated=not (path.parent / "worker_result.json").exists()))
    events.sort(key=lambda event: event["seq"])
    if [event["seq"] for event in events] != list(range(len(events))):
        raise ValueError("duplicate or missing event sequence")
    if len({event["event_id"] for event in events}) != len(events):
        raise ValueError("duplicate event_id")
    return events


def assert_equal_summary(actual, expected, path="summary"):
    """Integer/state fields are exact; floats use a fixed rounding tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path}: summary fields differ")
        for key in expected:
            assert_equal_summary(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, float):
        if type(actual) not in (int, float) or not math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError(f"{path}: numeric mismatch {actual!r} != {expected!r}")
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{path}: mismatch {actual!r} != {expected!r}")


def rebuild(run_dir):
    """No Simulator call: validate bundled inputs and recompute the full ledger."""
    from .ledger import recompute
    run_dir = no_symlinks(run_dir)
    planned = json.loads((run_dir / "planned_manifest.json").read_text())
    binding = json.loads((run_dir / "inputs/bindings.json").read_text())
    if set(binding) != set(planned["identity"]["input_sha256"]):
        raise ValueError("missing or extra bundled input binding")
    for name, record in binding.items():
        path = run_dir / "inputs" / record["name"]
        if Path(record["name"]).name != record["name"]:
            raise ValueError("unsafe input binding")
        if file_sha256(no_symlinks(path)) != planned["identity"]["input_sha256"][name]:
            raise ValueError("bundled input does not match planned input SHA")
    config = planned["config"]["engine"]
    if sha256_bytes(canonical_bytes(planned["config"])) != planned["identity"]["config_sha256"]:
        raise ValueError("planned config hash mismatch")
    events = collect_events(run_dir)
    summary = recompute(events, config)
    if not summary["finished"]:
        raise ValueError("partial prefix is not a complete evaluation result")
    saved = json.loads((run_dir / "summary.json").read_text())
    assert_equal_summary(summary, saved)
    assert_equal_summary(summary, json.loads((run_dir / "ledger.json").read_text()))
    for filename, kinds in [("completion_records.jsonl", {"local_complete", "sink_complete", "censor"}),
                            ("action_trace.jsonl", {"policy_decision", "startup", "ready", "drain", "drain_cancel", "shutdown"})]:
        if read_journal(run_dir / filename) != [event for event in events if event["kind"] in kinds]:
            raise ValueError("derived completion/action records disagree with authoritative events")
    result = json.loads((run_dir / "result_manifest.json").read_text())
    if result["technical_status"] != "completed":
        raise ValueError("worker did not complete")
    if result["planned_manifest_sha256"] != file_sha256(run_dir / "planned_manifest.json"):
        raise ValueError("result manifest identity mismatch")
    assert_equal_summary(summary, result["summary"])
    return summary


def _policy(args):
    if args.policy_config:
        return TargetPolicy(json.loads(no_symlinks(args.policy_config).read_text())).spec
    return TargetPolicy(json.loads(args.policy_json) if args.policy_json else "all1").spec


def _prepare(args):
    package = no_symlinks(args.package).resolve()
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if Path(args.window).name != args.window or args.window in {".", ".."}:
        raise ValueError("invalid window ID")
    record = manifest["files"][args.window + ".jsonl"]
    split = record["window"]["split"]
    if split not in {"train", "validation"} or args.split not in {None, split}:
        raise ValueError("N2/N3 execution permits only the selected train/validation split")
    raw_config = json.loads(no_symlinks(args.config).read_text())
    if raw_config.get("allow_locked_test", False):
        raise ValueError("locked-test authorization cannot be granted by a config flag")
    config = normalize_config(raw_config)
    if config["window_seconds"] != record["window"]["end"] - record["window"]["start"]:
        raise ValueError("config duration must equal the full declared workload window")
    policy = _policy(args)
    inputs = {"package_manifest": manifest_path, "window": package / (args.window + ".jsonl"),
              "engine_config_file": no_symlinks(args.config).resolve()}
    if args.policy_config:
        inputs["policy_config_file"] = no_symlinks(args.policy_config).resolve()
    # Optional constructed-fixture source records are explicitly hashed and
    # carried with the return package. Public input provenance is in manifest.
    for name, source in manifest.get("portable_source_files", {}).items():
        if Path(source["filename"]).name != source["filename"]:
            raise ValueError("unsafe portable input source")
        path = no_symlinks(package / source["filename"])
        if file_sha256(path) != source["sha256"]:
            raise ValueError("portable input source checksum mismatch")
        inputs["source_" + name] = path
    identity_config = {"engine": config, "policy": policy, "window_id": args.window,
                       "split": split, "mode": "independent_closed_loop"}
    metadata = {"split": split, "window": record["window"], "seed": raw_config.get("seed", 0),
                "policy": policy, "backend": "discrete_cpu_simulator",
                "real_backend": False, "gpu_executed": False,
                "service_parameters": "constructed_uncalibrated", "synthetic_api": True,
                "model": config["model"], "tokenizer": manifest.get("tokenizer"),
                "price_sha256": sha256_bytes(canonical_bytes(config["accounting"])),
                "scientific_contract": "diagnostic_only", "paper_claim_eligible": False,
                "checkpoint_every_steps": args.checkpoint_every,
                "fault_injection": bool(args.fault_after_step or args.stall_after_step),
                "trace_contract": {"events": "produced", "completion_records": "produced",
                                   "action_trace": "produced", "cost_ledger": "produced",
                                   "summary": "produced_on_complete"}}
    return package, inputs, identity_config, metadata


def worker(args):
    package, inputs, identity_config, metadata = _prepare(args)
    code_files = canonical_v2_source_files(ROOT)
    if args.resume:
        context = RunContext.open(args.resume, ROOT, identity_config, inputs, code_files)
        parent_checkpoint = context.latest_checkpoint()
        snapshot = context.load_checkpoint(parent_checkpoint)
        context.new_segment(parent_checkpoint)
    else:
        context = RunContext.create(ROOT, args.output_root, args.run_id, identity_config,
                                    inputs, code_files, metadata=metadata)
        (context.run_dir / "inputs").mkdir()
        bindings = {}
        for index, (name, path) in enumerate(inputs.items()):
            filename = f"{index:02d}_{Path(path).name}"
            data = Path(path).read_bytes()
            if sha256_bytes(data) != context.planned["identity"]["input_sha256"][name]:
                raise ValueError("input changed while bundling")
            with (context.run_dir / "inputs" / filename).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            bindings[name] = {"name": filename}
        durable_json(context.run_dir / "inputs/bindings.json", bindings, exclusive=True)
        context.new_segment()
        snapshot = None
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        rows = load_window(package, args.window, allowed_split=identity_config["split"])
        config = identity_config["engine"]
        simulator = (Simulator.restore(rows, config, snapshot, emit=context.append_event) if snapshot
                     else Simulator(rows, config, identity_config["policy"], emit=context.append_event))
        if snapshot is None:
            context.checkpoint(simulator.snapshot())
        while not simulator.finished:
            if stop or context.stop_requested():
                context.checkpoint(simulator.snapshot())
                context.finish("partial", "controlled_stop_at_complete_tick", summary=simulator.summary())
                return 0
            simulator.step()
            context.append_metric({"sim_tick": simulator.now, "pending": simulator.summary()["pending"]})
            context.heartbeat()
            # Crash before checkpoint to exercise replay of persisted events
            # newer than the latest complete application snapshot.
            if args.fault_after_step == simulator.now:
                raise RuntimeError("injected engine-worker exception before checkpoint")
            if args.stall_after_step == simulator.now:
                while True:
                    time.sleep(0.05)
            if simulator.now % args.checkpoint_every == 0 or simulator.finished:
                context.checkpoint(simulator.snapshot())
            if args.step_delay:
                time.sleep(args.step_delay)
        from .ledger import recompute
        events = collect_events(context.run_dir)
        summary = simulator.summary()
        independent = recompute(events, config)
        assert_equal_summary(independent, summary)
        durable_json(context.run_dir / "summary.json", summary)
        durable_json(context.run_dir / "ledger.json", independent)
        # Evaluations have explicit completion/action products as well as the
        # authoritative event journal, so a consumer need not infer omissions.
        for filename, kinds in [("completion_records.jsonl", {"local_complete", "sink_complete", "censor"}),
                                ("action_trace.jsonl", {"policy_decision", "startup", "ready", "drain", "drain_cancel", "shutdown"})]:
            data = b"".join(canonical_bytes(event) for event in events if event["kind"] in kinds)
            temporary = context.run_dir / (filename + ".tmp-" + uuid.uuid4().hex)
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, context.run_dir / filename)
        context.finish("completed", "engine_and_independent_ledger_match", summary=summary)
        return 0
    except Exception as error:
        if not (context.segment / "worker_result.json").exists():
            context.finish("failed", repr(error))
        else:
            durable_json(context.segment / "post_terminal_error.json", {"error": repr(error)}, exclusive=True)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--operation", choices=["run", "worker", "rebuild"], default="run")
    result.add_argument("--package")
    result.add_argument("--window")
    result.add_argument("--split", choices=["train", "validation"])
    policy = result.add_mutually_exclusive_group()
    policy.add_argument("--policy-json")
    policy.add_argument("--policy-config")
    result.add_argument("--config")
    result.add_argument("--output-root")
    result.add_argument("--run-id")
    result.add_argument("--resume")
    result.add_argument("--run-dir", help="input run directory for independent rebuild")
    result.add_argument("--limits-json")
    result.add_argument("--checkpoint-every", type=int, default=5)
    result.add_argument("--step-delay", type=float, default=0)
    result.add_argument("--fault-after-step", type=int)
    result.add_argument("--stall-after-step", type=int)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.operation == "rebuild":
        if not args.run_dir:
            argument_parser.error("--run-dir required for rebuild")
        print(json.dumps(rebuild(args.run_dir), sort_keys=True))
        return 0
    if not all([args.package, args.window, args.config, args.output_root, args.run_id]):
        argument_parser.error("--package/--window/--config/--output-root/--run-id required")
    if args.checkpoint_every < 1 or args.step_delay < 0:
        argument_parser.error("invalid checkpoint cadence or step delay")
    if args.operation == "worker":
        return worker(args)
    # Replace only the operation argument; all caller options are retained in
    # actual child argv and the supervisor's immutable plan.
    if "--operation" in argv:
        position = argv.index("--operation")
        del argv[position:position + 2]
    child = [sys.executable, "-m", "fidelityloop.legacy.maxopt_v2.run", "--operation", "worker", *argv]
    run_dir = Path(args.resume) if args.resume else Path(args.output_root) / args.run_id
    supervisor_dir = Path(args.output_root) / "supervisors" / (args.run_id + "-" + uuid.uuid4().hex)
    limits = Limits(**json.loads(no_symlinks(args.limits_json).read_text())) if args.limits_json else Limits()
    result = supervise(child, ROOT, run_dir, supervisor_dir, limits)
    print(json.dumps({"supervisor_dir": str(supervisor_dir), "run_dir": str(run_dir), **result}, sort_keys=True))
    return 0 if result["technical_status"] == "exited_zero" else 1


if __name__ == "__main__":
    raise SystemExit(main())
