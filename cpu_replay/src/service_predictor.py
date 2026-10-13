"""CPU-only fixed-admission service prediction with the native Vidur scheduler."""
import argparse
import atexit
import csv
import json
import math
import os
from pathlib import Path
import sys

from profile_qwen import MODEL, ROOT, audit_profile, sha256, write_json, validate_protocol

FROZEN_CONFIG = {
    "scheduler": "sarathi", "chunk_size": 16384, "batch_size_cap": 4,
    "block_size": 16, "num_estimators": [100], "max_depth": [8],
    "min_samples_split": [2], "k_fold_cv_splits": 3, "seed": 42,
    "num_training_job_threads": 2, "prediction_max_tokens_per_request": 2560,
    "prediction_max_prefill_chunk_size": 4096, "prediction_max_batch_size": 4,
    "compute_lookup_max_tokens": 8192, "kv_cache_prediction_granularity": 64,
    "skip_cpu_overhead_modeling": True, "attention_prefill_batching_overhead_fraction": 0.1,
    "attention_decode_batching_overhead_fraction": 0.1,
}


def validate_groups(document):
    """Reject outcomes and unknown fields before any native request is built."""
    if set(document) != {"groups"} or not isinstance(document["groups"], list) or not document["groups"]:
        raise ValueError("Expected nonempty {groups: [...]} only")
    seen_groups = set()
    for group in document["groups"]:
        if set(group) != {"group_id", "requests"}:
            raise ValueError("Group accepts group_id and requests only")
        if not isinstance(group["group_id"], str) or group["group_id"] in seen_groups:
            raise ValueError("Group IDs must be unique strings")
        seen_groups.add(group["group_id"])
        if not 1 <= len(group["requests"]) <= 4:
            raise ValueError("Each independent service group must contain 1–4 requests")
        seen_requests = set()
        for request in group["requests"]:
            if set(request) != {"request_id", "input_tokens", "output_tokens", "arrival_s"}:
                raise ValueError("Only request ID, token lengths and submit-relative arrival are permitted; outcomes forbidden")
            if not isinstance(request["request_id"], str) or request["request_id"] in seen_requests:
                raise ValueError("Request IDs must be unique strings within each group")
            seen_requests.add(request["request_id"])
            for key, high in (("input_tokens", 2048), ("output_tokens", 256)):
                value = request[key]
                low = 32 if key == "input_tokens" else 1
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"{key} outside frozen domain")
            arrival = request["arrival_s"]
            if type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < 0:
                raise ValueError("arrival_s must be a finite nonnegative number")
        if min(r["arrival_s"] for r in group["requests"]) != 0:
            raise ValueError("Arrival times must be relative to the first actual submit")
    return document["groups"]


def validate_admission_document(document, *, protocol_sha256, profile_manifest_sha256):
    """Accept the prepare-stage receipt, then isolate scheduling inputs."""
    fields = {"schema", "scope", "protocol_sha256", "measurements_sha256", "profile_manifest_sha256", "groups"}
    if set(document) != fields:
        raise ValueError("Expected the complete fixed-admission prepare document")
    expected = {
        "schema": "fidelityloop-fixed-admission-v1",
        "scope": "Observed admission only; no physical finish/deadline/latency fields",
        "protocol_sha256": protocol_sha256,
        "profile_manifest_sha256": profile_manifest_sha256,
    }
    for key, value in expected.items():
        if document[key] != value:
            raise ValueError(f"Admission document identity mismatch: {key}")
    measurement_hash = document["measurements_sha256"]
    if (not isinstance(measurement_hash, str) or len(measurement_hash) != 64
            or any(character not in "0123456789abcdef" for character in measurement_hash)):
        raise ValueError("Invalid admission measurements_sha256")
    # The measurement hash is provenance only; the predictor never opens outcomes.
    return validate_groups({"groups": document["groups"]})


def verify_profile(profile_dir, protocol_path):
    protocol = validate_protocol(protocol_path)
    expected_prediction = {"scheduler": "sarathi", "chunk_size": 16384, "batch_size_cap": 4,
                           "skip_cpu_overhead_modeling": True, "rf_trees": 100, "rf_max_depth": 8,
                           "rf_min_samples_split": 2, "cv": 3, "seed": 42, "threads": 2,
                           "compute_lookup_max_tokens": 8192, "prefill_lookup_max_tokens": 4096,
                           "kv_lookup_max_tokens": 2560, "retuning_on_validation": False,
                           "kv_cache_prediction_granularity": 64,
                           "attention_prefill_batching_overhead_fraction": 0.1,
                           "attention_decode_batching_overhead_fraction": 0.1}
    for key, value in expected_prediction.items():
        if protocol["prediction"].get(key) != value:
            raise ValueError(f"Unsupported prediction protocol value: {key}")
    manifest = json.loads((profile_dir / "profile_manifest.json").read_text())
    expected = {"schema": "fidelityloop-vidur-profile-v1", "status": "COMPLETE", "model": MODEL,
                "dtype": "float16", "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                "protocol_sha256": sha256(protocol_path)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Profile receipt mismatch: {key}")
    for name in ("mlp.csv", "attention.csv"):
        if manifest["files"][name]["sha256"] != sha256(profile_dir / name):
            raise ValueError(f"Profile hash mismatch: {name}")
    audit_profile(profile_dir)
    with (profile_dir / "mlp.csv").open(newline="") as stream:
        mlp = list(csv.DictReader(stream))
    with (profile_dir / "attention.csv").open(newline="") as stream:
        attention = list(csv.DictReader(stream))
    if min(int(r["num_tokens"]) for r in mlp) > 1 or max(int(r["num_tokens"]) for r in mlp) < 8192:
        raise ValueError("MLP training domain must cover 1–8192")
    if not any(int(r["prefill_chunk_size"]) == 8192 and int(r["kv_cache_size"]) == 0 for r in attention):
        raise ValueError("8192-token KV-save measured anchor missing")
    if not any(int(r["prefill_chunk_size"]) == 4096 and int(r["kv_cache_size"]) == 0 for r in attention):
        raise ValueError("4096-token aggregate-prefill measured anchor missing")
    for batch in range(1, 5):
        rows = [r for r in attention if int(r["prefill_chunk_size"]) == 0 and int(r["batch_size"]) == batch]
        if not rows or min(int(r["kv_cache_size"]) for r in rows) > 64 or max(int(r["kv_cache_size"]) for r in rows) < 2304:
            raise ValueError(f"Decode training domain missing for batch {batch}")
    return manifest


def bootstrap_cpu(source=None):
    for key, value in {"CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}.items():
        os.environ[key] = value
    sys.path.insert(0, str(source or ROOT / "vendor" / "vidur"))
    import qwen_registration  # noqa: F401


def run_native(groups, profile_dir, work_dir, *, model_name=MODEL, source=None):
    """Native requests/events/scheduler; subclass only separates compute lookup extent."""
    bootstrap_cpu(source)
    from vidur.config import (ClusterConfig, MetricsConfig, ReplicaConfig, SarathiSchedulerConfig,
                              SimulationConfig, TraceRequestGeneratorConfig, RandomForrestExecutionTimePredictorConfig)
    from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
    from vidur.execution_time_predictor.random_forrest_execution_time_predictor import RandomForrestExecutionTimePredictor
    from vidur.simulator import Simulator
    from vidur.entities import Request, Replica
    from vidur.events import BaseEvent
    from vidur.types import ExecutionTimePredictorType
    from vidur.utils.random import set_seeds

    class DomainBoundedPredictor(RandomForrestExecutionTimePredictor):
        def _predict_for_compute_models(self):
            original = self._max_tokens
            self._max_tokens = 8192
            try:
                return super()._predict_for_compute_models()
            finally:
                self._max_tokens = original

        def get_execution_time(self, batch, pipeline_stage):
            prefill = self._get_batch_prefill_attention_params(batch)
            decode_batch, decode_kv = self._get_batch_decode_attention_params(batch)
            aggregate_prefill = round(math.sqrt(sum(chunk * chunk for _, chunk in prefill)))
            aggregate_kv = sum(kv for kv, _ in prefill)
            if (batch._total_num_tokens_rounded > 8192 or sum(batch.num_tokens) > 8192
                    or aggregate_prefill > 4096 or aggregate_kv != 0
                    or decode_batch > 4 or decode_kv > 2304):
                raise ValueError("Runtime operator query escaped the frozen measured domain")
            lookup_queries.append({"compute_tokens": batch._total_num_tokens_rounded,
                                   "kv_save_tokens": sum(batch.num_tokens),
                                   "aggregate_prefill_tokens": aggregate_prefill,
                                   "aggregate_prefill_kv": aggregate_kv,
                                   "decode_batch": decode_batch, "decode_kv": decode_kv})
            return super().get_execution_time(batch, pipeline_stage)

    # Reuse only immutable trained models across independently initialized simulations.
    predictor_holder = []
    lookup_queries = []
    def predictor_factory(**kwargs):
        if not predictor_holder:
            predictor_holder.append(DomainBoundedPredictor(**kwargs))
        return predictor_holder[0]

    key = ExecutionTimePredictorType.RANDOM_FORREST
    original_class = ExecutionTimePredictorRegistry.get_class(key)
    ExecutionTimePredictorRegistry.unregister(key)
    ExecutionTimePredictorRegistry.register(key, predictor_factory)
    output = []
    try:
        for group_index, group in enumerate(groups):
            # Vidur entity IDs are process-global; each independent service group
            # must start from the same local namespace for replica_id=0.
            Request._id = Replica._id = -1
            BaseEvent._id = 0
            directory = Path(work_dir) / f"group_{group_index:03d}"
            directory.mkdir(parents=True, exist_ok=False)
            trace = directory / "requests.csv"
            with trace.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
                writer.writerows((r["arrival_s"], r["input_tokens"], r["output_tokens"]) for r in group["requests"])
            metrics = MetricsConfig(output_dir=str(directory / "native"), cache_dir=str(Path(work_dir) / "model_cache"),
                                    write_metrics=False, enable_chrome_trace=False, write_json_trace=False, store_plots=False)
            config = SimulationConfig(seed=42, cluster_config=ClusterConfig(num_replicas=1,
                replica_config=ReplicaConfig(model_name=model_name, device="a100", tensor_parallel_size=1, num_pipeline_stages=1),
                replica_scheduler_config=SarathiSchedulerConfig(batch_size_cap=4, chunk_size=16384, block_size=16)),
                request_generator_config=TraceRequestGeneratorConfig(trace_file=str(trace), max_tokens=2560),
                execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                    compute_input_file=str(Path(profile_dir).resolve() / "mlp.csv"),
                    attention_input_file=str(Path(profile_dir).resolve() / "attention.csv"), no_cache=True,
                    num_estimators=[100], max_depth=[8], min_samples_split=[2], k_fold_cv_splits=3,
                    num_training_job_threads=2, prediction_max_prefill_chunk_size=4096,
                    prediction_max_batch_size=4, prediction_max_tokens_per_request=2560,
                    kv_cache_prediction_granularity=64, skip_cpu_overhead_modeling=True,
                    attention_prefill_batching_overhead_fraction=0.1, attention_decode_batching_overhead_fraction=0.1),
                metrics_config=metrics)
            set_seeds(42)
            simulator = Simulator(config)
            atexit.unregister(simulator._write_output)
            requests = sorted((event._request for _, event in simulator._event_queue), key=lambda request: request.id)
            simulator.run()
            if simulator._event_queue or not simulator.scheduler.is_empty() or simulator._terminate:
                raise RuntimeError("Native simulation did not drain")
            records = []
            for spec, request in zip(group["requests"], requests):
                if (request.size != (spec["input_tokens"], spec["output_tokens"])
                        or not math.isclose(float(request.arrived_at), float(spec["arrival_s"]), rel_tol=0.0, abs_tol=1e-12)):
                    raise RuntimeError("Native generator changed request semantics")
                if not request.completed or request.num_processed_tokens != request.total_tokens:
                    raise RuntimeError("Native request incomplete")
                records.append({"request_id": spec["request_id"], "predicted_latency_s": request.e2e_time,
                                "predicted_completion_s": request.completed_at})
            output.append({"group_id": group["group_id"], "requests": records,
                           "predicted_makespan_s": max(r["predicted_completion_s"] for r in records)})
        write_json(Path(work_dir) / "lookup_domain_audit.json", {
            "native_batch_count": len(lookup_queries),
            "maximum_observed_query": {key: max(row[key] for row in lookup_queries) for key in lookup_queries[0]},
            "all_queries_within_frozen_lookup_domain": True,
            "model": model_name,
        })
    finally:
        ExecutionTimePredictorRegistry.unregister(key)
        ExecutionTimePredictorRegistry.register(key, original_class)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    groups = validate_admission_document(
        json.loads(args.requests.read_text()), protocol_sha256=sha256(args.protocol),
        profile_manifest_sha256=sha256(args.profile_dir / "profile_manifest.json"))
    manifest = verify_profile(args.profile_dir, args.protocol)
    work_dir = args.output.parent / (args.output.stem + "_native")
    work_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema": "fidelityloop-vidur-service-prediction-v1", "model": MODEL,
              "protocol_sha256": sha256(args.protocol), "requests_sha256": sha256(args.requests),
              "profile_manifest_sha256": sha256(args.profile_dir / "profile_manifest.json"),
              "frozen_predictor": FROZEN_CONFIG, "cpu_overhead_modeled": False,
              "lifecycle_modeled": False, "fixed_admission_only": True,
              "profile_files": manifest["files"],
              "groups": run_native(groups, args.profile_dir, work_dir)}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
