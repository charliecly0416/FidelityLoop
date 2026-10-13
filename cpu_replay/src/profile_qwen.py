"""Sequential, real-GPU Vidur profiling; no fabricated or zero-filled labels."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MLP_OPERATORS = (
    "attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj",
    "mlp_act", "input_layernorm", "post_attention_layernorm", "attn_rope", "add",
)
QWEN_SHAPE = {"n_embd": 3584, "n_kv_head": 4, "num_tensor_parallel_workers": 1}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def validate_protocol(path):
    protocol = json.loads(Path(path).read_text())
    expected = {"dtype": "float16", "tp": 1, "pp": 1, "mlp_max_tokens": 8192,
                "attention_grid_max_seq_len": 4096, "attention_extra_prefill_anchor": 8192,
                "attention_decode_max_kv": 2560,
                "attention_max_batch": 4, "block_size": 16, "memory_max_pipeline_parallel_size": 1,
                "smoke": {"mlp_tokens": 128, "prefill_tokens": 128, "decode_kv": 128, "batch_size": 1},
                "strict_profile_fields": True}
    if protocol.get("model") != MODEL:
        raise ValueError("Protocol model differs from frozen profiler")
    for key, value in expected.items():
        if protocol["profiling"].get(key) != value:
            raise ValueError(f"Unsupported profiling protocol value: {key}")
    return protocol


def flatten_result(result):
    row = {key: value for key, value in result.items() if key != "time_stats"}
    for operator, stats in result["time_stats"].items():
        for stat, value in stats.items():
            row[f"time_stats.{operator}.{stat}"] = value
    return row


def _boolean(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def audit_rows(rows, kind, *, qwen=True, full=False):
    """Check only applicable attention labels; opposite branch cells stay empty."""
    if not rows:
        raise ValueError(f"No {kind} measurements")
    counts = {"mlp": 0, "prefill": 0, "decode": 0}
    for index, row in enumerate(rows):
        expected = dict(QWEN_SHAPE) if qwen else {}
        if kind == "mlp":
            expected.update({"n_head": 28, "n_expanded_embd": 18944, "vocab_size": 152064} if qwen else {})
            operators = MLP_OPERATORS
            counts["mlp"] += 1
            tokens = int(row["num_tokens"])
            if not 1 <= tokens <= 8192 or not _boolean(row["use_gated_mlp"]):
                raise ValueError(f"Invalid MLP shape at row {index}")
        else:
            expected.update({"n_q_head": 28, "block_size": 16} if qwen else {})
            prefill = _boolean(row["is_prefill"])
            branch = "prefill" if prefill else "decode"
            counts[branch] += 1
            operators = ("attn_kv_cache_save", f"attn_{branch}")
            chunk, kv, batch = (int(row[key]) for key in ("prefill_chunk_size", "kv_cache_size", "batch_size"))
            if prefill:
                valid = chunk > 0 and kv >= 0 and batch == 1 and (chunk + kv <= 4096 or (chunk == 8192 and kv == 0))
            else:
                valid = chunk == 0 and 0 < kv <= 2560 and 1 <= batch <= 4
            if not valid:
                raise ValueError(f"Invalid attention shape at row {index}")
        for key, value in expected.items():
            if float(row[key]) != value:
                raise ValueError(f"Wrong {key} at {kind} row {index}")
        for operator in operators:
            key = f"time_stats.{operator}.median"
            try:
                value = float(row[key])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Missing measured label {key}, {kind} row {index}; zero-fill prohibited") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Nonpositive/nonfinite measured label {key}, row {index}")
    if full:
        needed = ["mlp"] if kind == "mlp" else ["prefill", "decode"]
        if any(counts[key] < 3 for key in needed):
            raise ValueError("Full profile requires at least three measured rows per RF training subset")
    return counts


def audit_profile(profile_dir, *, full=True, qwen=True):
    result = {}
    for kind in ("mlp", "attention"):
        with (Path(profile_dir) / f"{kind}.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        result[kind] = audit_rows(rows, kind, full=full, qwen=qwen)
    return result


def bootstrap_gpu():
    sys.path.insert(0, str(ROOT / "vendor" / "vidur"))
    import qwen_registration  # noqa: F401; isolated registry extension
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU using CUDA_VISIBLE_DEVICES")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen protocol requires an A100")
    if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
        raise RuntimeError("Frozen protocol requires an A100 80GB")
    torch.manual_seed(42)
    return torch


def run_worker(kind, stage, destination):
    torch = bootstrap_gpu()
    from vidur.profiling.common.model_config import ModelConfig
    # Sarathi's package initializer eagerly imports modules with from-imported timers.
    # Complete Vidur's intended instrumentation patch for those already bound names.
    import sarathi.metrics.cuda_timer
    old_timer = sarathi.metrics.cuda_timer.CudaTimer
    from vidur.profiling.common.cuda_timer import CudaTimer
    timer_rebound_modules = []
    for name, module in list(sys.modules.items()):
        if name.startswith("sarathi.") and getattr(module, "CudaTimer", None) is old_timer:
            module.CudaTimer = CudaTimer
            timer_rebound_modules.append(name)
    model = ModelConfig.from_model_name(MODEL)
    used_sources = {}
    for module_name in ("vidur.profiling.common.cuda_timer", "vidur.profiling.common.model_config",
                        "sarathi.model_executor.attention.flashinfer_attention_wrapper",
                        "sarathi.model_executor.attention.base_attention_wrapper"):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "__file__", None):
            used_sources[module_name] = {"path": module.__file__, "sha256": sha256(module.__file__)}
    packages = {}
    for name in ("torch", "flashinfer", "sarathi", "numpy", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata = {
        "kind": kind, "stage": stage, "model": MODEL, "dtype": "float16",
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "packages": packages, "python": sys.version,
        "timing_unit": "milliseconds", "seed": 42,
        "timing_method": "record_function" if kind == "mlp" else "kineto",
        "attention_backend": "pinned_sarathi_FLASHINFER_not_vllm_FLASH_ATTENTION",
        "timer_rebound_modules": timer_rebound_modules,
        "used_source_files": used_sources,
        "vidur_commit": "abae7f63aa857300f5cdc6f5e0d27860cd24721b",
        "sarathi_commit": "c94dbf438711a4670dbf7477175062e87748999d",
        "attention_page_table_rng": "upstream numpy.default_rng entropy (not seeded by torch seed)",
    }
    write_json(destination / f"{kind}_environment.json", metadata)
    if kind == "mlp":
        from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
        from vidur.profiling.utils import get_num_tokens_to_profile
        shapes = [128] if stage == "smoke" else sorted(set(get_num_tokens_to_profile(8192)), reverse=True)
        wrapper = MlpWrapper(model, 1, "record_function", 0, str(destination))
    else:
        from sarathi.config import ParallelConfig
        from sarathi.model_executor.attention import AttentionBackend
        from vidur.profiling.attention.attention_input import AttentionInput
        from vidur.profiling.attention.attention_wrapper import AttentionWrapper
        from vidur.profiling.utils import get_attention_input_combinations, get_max_num_blocks
        parallel = ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1)
        blocks = get_max_num_blocks(model, parallel, 16, torch.float16, max_pipeline_parallel_size=1)
        if stage == "smoke":
            shapes = [AttentionInput(128, 0, 1, True), AttentionInput(0, 128, 1, False)]
        else:
            shapes = [x for x in get_attention_input_combinations(4096, 1, 4, False, False)
                      if x.is_prefill or x.kv_cache_size <= 2560]
            shapes.append(AttentionInput(8192, 0, 1, True))
            shapes = list({(x.prefill_chunk_size, x.kv_cache_size, x.batch_size, x.is_prefill): x for x in shapes}.values())
        omitted = [vars(x) for x in shapes if not x.is_under_memory_limit(blocks * 16)]
        write_json(destination / "attention_grid.json", {"max_num_blocks": blocks, "max_pipeline_parallel_size": 1,
                   "requested_shapes": [vars(x) for x in shapes], "omitted_shapes": omitted})
        if omitted:
            raise RuntimeError("Frozen grid exceeds memory estimate; do not silently omit shapes")
        wrapper = AttentionWrapper(model, parallel, blocks, 8192, 16, AttentionBackend.FLASHINFER, torch.float16)
    with (destination / f"{kind}_raw.jsonl").open("x", encoding="utf-8") as raw:
        for index, shape in enumerate(shapes):
            measured = wrapper.profile(shape)
            raw.write(json.dumps(measured, default=str, allow_nan=False) + "\n")
            raw.flush()
            audit_rows([flatten_result(measured)], kind)
            print(f"{kind} {index + 1}/{len(shapes)}", flush=True)


def build_csv(destination, kind):
    rows = [flatten_result(json.loads(line)) for line in (destination / f"{kind}_raw.jsonl").read_text().splitlines()]
    audit_rows(rows, kind)
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (destination / f"{kind}.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--worker-kind", choices=["mlp", "attention"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_kind:
        run_worker(args.worker_kind, args.stage, args.output_dir)
        return
    protocol = validate_protocol(args.protocol)
    protocol_hash = sha256(args.protocol)
    if args.stage == "full":
        smoke_dir = args.output_dir / "smoke"
        smoke = json.loads((smoke_dir / "profile_manifest.json").read_text())
        if smoke["status"] != "SMOKE_COMPLETE" or smoke["protocol_sha256"] != protocol_hash:
            raise ValueError("Full profiling requires successful smoke under the same frozen protocol")
        for filename, receipt in smoke["files"].items():
            if sha256(smoke_dir / filename) != receipt["sha256"]:
                raise ValueError("Smoke profile hash mismatch")
    destination = args.output_dir.resolve() / args.stage
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        for kind in ("mlp", "attention"):
            command = [sys.executable, str(Path(__file__).resolve()), "--stage", args.stage,
                       "--output-dir", str(destination), "--worker-kind", kind]
            with (destination / f"{kind}.log").open("x", encoding="utf-8") as log:
                budget = 1800 if args.stage == "smoke" else protocol["limits"]["profile_stage_timeout_s"]
                remaining = budget - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Profiling stage total budget exhausted")
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
            build_csv(destination, kind)
        audit = audit_profile(destination, full=args.stage == "full")
        manifest = {"schema": "fidelityloop-vidur-profile-v1", "status": "COMPLETE" if args.stage == "full" else "SMOKE_COMPLETE",
                    "stage": args.stage, "model": MODEL, "dtype": "float16", "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1, "protocol_sha256": protocol_hash, "audit": audit,
                    "files": {name: {"sha256": sha256(destination / name)} for name in ("mlp.csv", "attention.csv")},
                    "elapsed_s": time.monotonic() - started, "gpu_samples_fabricated": False,
                    "profiler_script_sha256": sha256(__file__), "source_lock_sha256": sha256(ROOT / "SOURCE_LOCK.json") if (ROOT / "SOURCE_LOCK.json").exists() else None}
        write_json(destination / "profile_manifest.json", manifest)
    except Exception as exc:
        write_json(destination / "failure.json", {"status": "FAILED", "stage": args.stage,
                   "error": str(exc), "traceback": traceback.format_exc(), "protocol_sha256": protocol_hash,
                   "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
