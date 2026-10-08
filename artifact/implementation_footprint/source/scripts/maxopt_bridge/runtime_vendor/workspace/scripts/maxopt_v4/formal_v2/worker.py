"""Plan-configured TP1 vLLM worker for formal_v2.

The accepted V4 worker supplies request accounting and cleanup behavior. This
entry point deliberately owns engine construction so a Qwen float16 plan never
falls through to the older Llama bfloat16 default.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

# The backend starts this file by absolute path from a per-cell output folder.
# Do not depend on the controller's current directory or inherited PYTHONPATH.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.maxopt_v4.w2_gpu_all2 import llama_tp1_worker as accepted

LOCAL_SLOTS = 4
ALLOWED_DTYPES = {"float16", "bfloat16"}


class GenerationWriter:
    def __init__(self, path: str, generation: int) -> None:
        self.writer = accepted.EventWriter(path)
        self.generation, self.sequence = generation, 0

    def __call__(self, event: dict) -> None:
        self.writer(dict(event, generation=self.generation, worker_seq=self.sequence))
        self.sequence += 1

    def close(self) -> None:
        self.writer.close()


def validate_runtime(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("runtime must be a mapping")
    runtime = dict(value)
    # Plans bind the V3 runtime template verbatim.  Normalize it at this
    # boundary instead of asking each caller to maintain a second copy with
    # vLLM-internal field names.
    required = {
        "dtype", "context", "tp", "pp", "dp", "eager",
        "gpu_memory_utilization", "prefix_cache", "max_num_batched_tokens",
        "enable_chunked_prefill", "max_num_seqs", "local_slots",
    }
    if set(runtime) != required:
        raise ValueError("runtime keys differ from formal_v2 contract")
    if runtime["dtype"] not in ALLOWED_DTYPES:
        raise ValueError("runtime dtype must be float16 or bfloat16")
    if (runtime["tp"], runtime["pp"], runtime["dp"]) != (1, 1, 1):
        raise ValueError("formal_v2 requires independent TP1/PP1/DP1 workers")
    if runtime["context"] != 4096 or runtime["max_num_batched_tokens"] != 16384:
        raise ValueError("formal_v2 frozen context/batch limits changed")
    if runtime["max_num_seqs"] != 1024 or runtime["eager"] is not True:
        raise ValueError("formal_v2 frozen engine settings changed")
    if runtime["gpu_memory_utilization"] != 0.8 or runtime["prefix_cache"] is not False:
        raise ValueError("formal_v2 frozen cache settings changed")
    if runtime["enable_chunked_prefill"] is not True:
        raise ValueError("formal_v2 frozen prefill setting changed")
    if runtime["local_slots"] != LOCAL_SLOTS:
        raise ValueError("formal_v2 local-slot contract changed")
    return {
        "dtype": runtime["dtype"], "max_model_len": runtime["context"],
        "tensor_parallel_size": runtime["tp"], "pipeline_parallel_size": runtime["pp"],
        "data_parallel_size": runtime["dp"], "enforce_eager": runtime["eager"],
        "gpu_memory_utilization": runtime["gpu_memory_utilization"],
        "enable_prefix_caching": runtime["prefix_cache"],
        "max_num_batched_tokens": runtime["max_num_batched_tokens"],
        "enable_chunked_prefill": runtime["enable_chunked_prefill"],
        "max_num_seqs": runtime["max_num_seqs"],
        "engine_class": "vllm.v1.engine.async_llm.AsyncLLM",
    }


def create_real_engine(model: str, runtime: dict):
    import torch
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import RequestOutputKind

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must expose exactly one CUDA device")
    props = torch.cuda.get_device_properties(0)
    observed = {
        "model": str(Path(model).resolve()), "dtype": runtime["dtype"],
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1,
        "max_model_len": runtime["max_model_len"], "enforce_eager": runtime["enforce_eager"],
        "gpu_memory_utilization": runtime["gpu_memory_utilization"],
        "enable_prefix_caching": runtime["enable_prefix_caching"],
        "max_num_batched_tokens": runtime["max_num_batched_tokens"],
        "enable_chunked_prefill": runtime["enable_chunked_prefill"],
        "max_num_seqs": runtime["max_num_seqs"], "local_slots": LOCAL_SLOTS,
        "visible_device_count": torch.cuda.device_count(), "logical_device": 0,
        "device_name": props.name, "total_memory": props.total_memory,
        "device_capability": [props.major, props.minor], "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("vllm", "transformers", "tokenizers", "numpy")},
    }
    args = AsyncEngineArgs(
        model=observed["model"], tensor_parallel_size=1, pipeline_parallel_size=1,
        data_parallel_size=1, dtype=runtime["dtype"], max_model_len=runtime["max_model_len"],
        enforce_eager=True, gpu_memory_utilization=runtime["gpu_memory_utilization"],
        enable_prefix_caching=False, trust_remote_code=False,
        max_num_batched_tokens=runtime["max_num_batched_tokens"],
        enable_chunked_prefill=True, max_num_seqs=runtime["max_num_seqs"],
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    factory = lambda length: SamplingParams(**accepted.sampling_kwargs(length, RequestOutputKind.FINAL_ONLY))
    return engine, factory, observed


def effective_runtime(config, runtime: dict, engine: object) -> dict:
    engine_class = type(engine)
    return {
        "actual_engine_class": f"{engine_class.__module__}.{engine_class.__qualname__}",
        "effective_dtype": str(config.model_config.dtype),
        "effective_max_model_len": config.model_config.max_model_len,
        "effective_tensor_parallel_size": config.parallel_config.tensor_parallel_size,
        "effective_pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
        "effective_data_parallel_size": config.parallel_config.data_parallel_size,
        "effective_enforce_eager": config.model_config.enforce_eager,
        "effective_gpu_memory_utilization": config.cache_config.gpu_memory_utilization,
        "effective_enable_prefix_caching": config.cache_config.enable_prefix_caching,
        "effective_max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "effective_enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
        "effective_max_num_seqs": config.scheduler_config.max_num_seqs,
        "expected_dtype": "torch." + runtime["dtype"],
    }


def runtime_mismatches(observed: dict, runtime: dict) -> dict:
    expected = {
        "actual_engine_class": runtime["engine_class"],
        "effective_dtype": "torch." + runtime["dtype"],
        "effective_max_model_len": runtime["max_model_len"],
        "effective_tensor_parallel_size": 1, "effective_pipeline_parallel_size": 1,
        "effective_data_parallel_size": 1, "effective_enforce_eager": True,
        "effective_gpu_memory_utilization": 0.8, "effective_enable_prefix_caching": False,
        "effective_max_num_batched_tokens": 16384, "effective_enable_chunked_prefill": True,
        "effective_max_num_seqs": 1024,
    }
    return {key: {"expected": value, "actual": observed.get(key)}
            for key, value in expected.items() if observed.get(key) != value}


async def run_main(args, writer: GenerationWriter) -> int:
    runtime = validate_runtime(json.loads(args.runtime_json))
    worker = accepted.AsyncRequestWorker(None, None, writer, args.gpu_index)
    initial = {
        "model": str(Path(args.model).resolve()), "gpu_index": args.gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1,
        "runtime": runtime,
        "offline_environment": {key: os.environ.get(key) for key in
                                ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
    }
    worker.emit("init_start", runtime=initial,
                actual_engine_class=None, clock=time.get_clock_info("monotonic").implementation)
    exit_code = 0
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
            raise ValueError("CUDA_VISIBLE_DEVICES must equal the requested numeric physical index")
        if any(value != "1" for value in initial["offline_environment"].values()):
            raise ValueError("all Hugging Face offline environment flags must be 1")
        if not Path(args.model).is_dir():
            raise ValueError("model must be an existing local directory")
        worker.engine, worker.sampling_factory, observed = create_real_engine(args.model, runtime)
        observed.update(effective_runtime(await worker.engine.get_vllm_config(), runtime, worker.engine))
        mismatches = runtime_mismatches(observed, runtime)
        if mismatches:
            raise ValueError("effective engine configuration mismatch: " + repr(mismatches))
        worker.emit("init_done", runtime=observed, actual_engine_class=observed["actual_engine_class"])
        worker.emit("ready", runtime=observed, actual_engine_class=observed["actual_engine_class"],
                    readiness="constructed_only_controller_probe_required", **worker.snapshot())
        while not worker.closed:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                worker.emit("stdin_eof")
                break
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                worker.reject(None, f"invalid JSON command: {exc}")
                continue
            await worker.command(command)
    except Exception as exc:
        exit_code = 1
        worker.emit("fatal", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if worker.engine is not None:
            await worker.shutdown()
        else:
            worker.emit("closed", initialization_failed=True, cleanup_ok=True,
                        backend_process_exit_verified=False)
    return exit_code if worker.cleanup_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()
    if args.generation < 1:
        parser.error("generation must be positive")
    writer = GenerationWriter(args.events, args.generation)
    try:
        return asyncio.run(run_main(args, writer))
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
