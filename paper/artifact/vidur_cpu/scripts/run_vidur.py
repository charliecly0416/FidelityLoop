"""Run a locked CPU-only Vidur example and observe native request objects."""
import argparse
import atexit
import csv
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


COMMIT = "abae7f63aa857300f5cdc6f5e0d27860cd24721b"
ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["fixed32", "mixed32"], required=True)
    args = parser.parse_args()
    for key, value in {
        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }.items():
        os.environ[key] = value
    source = ROOT / "source" / f"vidur-{COMMIT}"
    out = ROOT / "runtime" / args.case
    out.mkdir(parents=True, exist_ok=True)
    # Upstream to_dict emits a Python set, which json.dump cannot serialize.
    # This changes only the exported representation, never the event handler.
    export_source = source / "vidur/events/global_schedule_event.py"
    original = export_source.read_text()
    before = '"replica_set": self._replica_set,'
    after = '"replica_set": sorted(self._replica_set),'
    if before in original:
        patched = original.replace(before, after)
        (ROOT / "runtime" / "native_export.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), patched.splitlines(keepends=True),
            fromfile="a/vidur/events/global_schedule_event.py", tofile="b/vidur/events/global_schedule_event.py")))
        export_source.write_text(patched)
    else:
        assert after in original, "Unexpected Vidur export source"
    os.chdir(source)
    sys.path.insert(0, str(source))
    from vidur.config import (
        ClusterConfig, MetricsConfig, ReplicaConfig, SarathiSchedulerConfig,
        SimulationConfig, TraceRequestGeneratorConfig,
        RandomForrestExecutionTimePredictorConfig,
    )
    from vidur.simulator import Simulator
    from vidur.utils.random import set_seeds

    workloads = [
        {"arrived_at": i / 10, "num_prefill_tokens": 512 if args.case == "mixed32" and i % 2 else 128,
         "num_decode_tokens": 128 if args.case == "mixed32" and i % 2 else 32}
        for i in range(32)
    ]
    with (out / "workload.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(workloads[0]))
        writer.writeheader()
        writer.writerows(workloads)
    metrics = MetricsConfig(
        output_dir=str(out / "native"), cache_dir=str(ROOT / "source" / "cache"),
        write_json_trace=True, enable_chrome_trace=True, store_plots=False,
        store_operation_metrics=True, keep_individual_batch_metrics=True,
        store_token_completion_metrics=True, store_utilization_metrics=True,
    )
    config = SimulationConfig(
        seed=42,
        cluster_config=ClusterConfig(
            num_replicas=1,
            replica_config=ReplicaConfig(model_name="meta-llama/Meta-Llama-3-8B", device="a100",
                                         tensor_parallel_size=1, num_pipeline_stages=1),
            replica_scheduler_config=SarathiSchedulerConfig(batch_size_cap=32, chunk_size=512),
        ),
        request_generator_config=TraceRequestGeneratorConfig(trace_file=str(out / "workload.csv"), max_tokens=1024),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            num_estimators=[100], max_depth=[8], min_samples_split=[2], k_fold_cv_splits=3,
            num_training_job_threads=2, prediction_max_prefill_chunk_size=512,
            prediction_max_batch_size=32, prediction_max_tokens_per_request=1024,
        ),
        metrics_config=metrics,
    )
    set_seeds(config.seed)
    started = time.monotonic()
    simulator = Simulator(config)
    requests = sorted((event._request for _, event in simulator._event_queue), key=lambda request: request.id)
    jsonl(out / "input_requests.jsonl", [
        {"id": request.id, "arrived_at": request.arrived_at,
         "num_prefill_tokens": request.num_prefill_tokens, "num_decode_tokens": request.num_decode_tokens}
        for request in requests
    ])
    simulation_started = time.monotonic()
    simulator.run()
    run_seconds = time.monotonic() - simulation_started
    records = [dict(request.to_dict(), completed=request.completed, prefill_completed_at=request.prefill_completed_at,
                    e2e_time=request.e2e_time) for request in requests]
    jsonl(out / "request_records.jsonl", records)
    assert len(records) == 32 and len({row["id"] for row in records}) == 32
    assert all(row["completed"] and row["num_processed_tokens"] == row["num_prefill_tokens"] + row["num_decode_tokens"] for row in records)
    assert all(row["arrived_at"] <= row["scheduled_at"] <= row["prefill_completed_at"] <= row["completed_at"] for row in records)
    assert not simulator._event_queue and simulator.scheduler.is_empty() and not simulator._terminate
    simulator._write_output()
    atexit.unregister(simulator._write_output)
    profiles = list((source / "data/profiling/compute/a100/meta-llama/Meta-Llama-3-8B").glob("*.csv"))
    metadata = {
        "official_commit": COMMIT, "case": args.case, "request_count": len(records),
        "completed_count": sum(row["completed"] for row in records),
        "event_queue_empty": not simulator._event_queue, "scheduler_empty": simulator.scheduler.is_empty(),
        "terminated_at_time_limit": simulator._terminate, "final_simulation_time_s": simulator._time,
        "training_and_simulation_wall_time_s": time.monotonic() - started,
        "simulation_wall_time_s": run_seconds, "native_output_dir": str(Path(metrics.output_dir).relative_to(ROOT)),
        "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "wandb", "plotly", "ddsketch", "fasteners"]},
        "profile_sha256": {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in profiles},
        "cpu_only": True, "official_source_patches": ["runtime/native_export.patch"],
        "torch_imported": "torch" in sys.modules,
        "workload_sha256": hashlib.sha256((out / "workload.csv").read_bytes()).hexdigest(),
        "request_record_sha256": hashlib.sha256((out / "request_records.jsonl").read_bytes()).hexdigest(),
    }
    dump(out / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
