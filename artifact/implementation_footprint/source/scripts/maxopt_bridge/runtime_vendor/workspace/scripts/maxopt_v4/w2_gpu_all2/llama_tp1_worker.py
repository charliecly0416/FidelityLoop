"""Migration/formal TP=1 worker with explicit 16384/true/1024 scheduler config.

Importing this module uses only the standard library. The real backend is loaded
only from main(), after the controller's device/offline environment is checked.
"""

import argparse
import asyncio
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


LOCAL_SLOTS = 4
MAX_MODEL_LEN = 4096
MAX_NUM_BATCHED_TOKENS = 16384
MAX_NUM_SEQS = 1024
ABORT_TIMEOUT_SECONDS = 10


class EventWriter:
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8", buffering=1)

    def __call__(self, event):
        self.file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


class AsyncRequestWorker:
    def __init__(self, engine, sampling_factory, event_writer, gpu_index):
        self.engine = engine
        self.sampling_factory = sampling_factory
        self.event_writer = event_writer
        self.gpu_index = gpu_index
        self.tasks = {}  # Retain every accepted task for final accounting.
        self.active = {}
        self.started = set()
        self.terminals = {}
        self.counts = Counter(submitted=0, finished=0, failed=0, cancelled=0, rejected=0)
        self.max_active = 0
        self.closing = False
        self.closed = False
        self.cleanup_ok = True

    def emit(self, kind, request_id=None, *, observed_ns=None, **fields):
        event = {
            "kind": kind,
            "request_id": request_id,
            "worker_monotonic_ns": time.monotonic_ns() if observed_ns is None else observed_ns,
            "pid": os.getpid(),
            "gpu_index": self.gpu_index,
            **fields,
        }
        self.event_writer(event)

    def snapshot(self):
        return {
            "active_request_ids": sorted(self.active),
            "active_count": len(self.active),
            "counts": dict(self.counts),
            "max_active": self.max_active,
            "terminal_count": len(self.terminals),
            "cleanup_ok": self.cleanup_ok,
        }

    def reject(self, request_id, reason):
        self.counts["rejected"] += 1
        self.emit("command_rejected", request_id, reason=reason)

    def terminal(self, kind, request_id, *, observed_ns=None, **fields):
        if request_id in self.terminals:
            return False
        self.terminals[request_id] = kind
        self.counts[kind] += 1
        self.emit(kind, request_id, observed_ns=observed_ns, **fields)
        return True

    async def abort(self, request_id, reason):
        try:
            await asyncio.wait_for(
                self.engine.abort(request_id), timeout=ABORT_TIMEOUT_SECONDS
            )
        except Exception as exc:
            self.cleanup_ok = False
            self.emit("abort_failed", request_id, reason=reason,
                      error_type=type(exc).__name__, error=str(exc),
                      backend_quiescence_confirmed=False)
            return False
        self.emit("abort_ack", request_id, reason=reason,
                  backend_quiescence_confirmed=False)
        return True

    async def run_request(self, request_id, prompt_token_ids, max_output_tokens):
        stream = None
        try:
            self.started.add(request_id)
            self.emit("submit_begin", request_id, prompt_token_ids=prompt_token_ids,
                      max_output_tokens=max_output_tokens,
                      actual_input_tokens=len(prompt_token_ids))
            params = self.sampling_factory(max_output_tokens)
            stream = self.engine.generate(
                prompt={"prompt_token_ids": list(prompt_token_ids)},
                sampling_params=params, request_id=request_id,
            )
            async for output in stream:
                if not output.finished:
                    continue
                finished_ns = time.monotonic_ns()
                if output.request_id != request_id:
                    raise ValueError("backend returned a different request_id")
                actual_prompt = list(output.prompt_token_ids or [])
                if actual_prompt != prompt_token_ids:
                    raise ValueError("backend prompt_token_ids do not match submitted tokens")
                if len(output.outputs) != 1:
                    raise ValueError("backend must return exactly one completion")
                completion = output.outputs[0]
                output_ids = list(completion.token_ids)
                if len(output_ids) != max_output_tokens:
                    raise ValueError(
                        f"output token count {len(output_ids)} != required {max_output_tokens}"
                    )
                if any(type(token) is not int or token < 0 for token in output_ids):
                    raise ValueError("backend output token IDs must be nonnegative integers")
                if completion.finish_reason is None:
                    raise ValueError("finished output lacks a finish_reason")
                self.terminal(
                    "finished", request_id, observed_ns=finished_ns,
                    observed_finished_monotonic_ns=finished_ns,
                    prompt_token_ids=actual_prompt, output_token_ids=output_ids,
                    actual_input_tokens=len(actual_prompt), actual_output_tokens=len(output_ids),
                    finish_reason=completion.finish_reason,
                    stop_reason=getattr(completion, "stop_reason", None),
                    backend_finished=True,
                )
                return
            raise RuntimeError("backend stream ended without a finished event")
        except asyncio.CancelledError:
            abort_returned = await self.abort(request_id, "cancelled")
            self.terminal("cancelled", request_id, abort_returned=abort_returned,
                          backend_quiescence_confirmed=False)
            raise
        except Exception as exc:
            abort_returned = await self.abort(request_id, "request_failed")
            self.terminal("failed", request_id, error_type=type(exc).__name__,
                          error=str(exc), abort_returned=abort_returned)
        finally:
            if stream is not None and hasattr(stream, "aclose"):
                try:
                    await stream.aclose()
                except Exception as exc:
                    self.cleanup_ok = False
                    self.emit("cleanup_failed", request_id, phase="stream_close",
                              error_type=type(exc).__name__, error=str(exc))

    def task_done(self, request_id, task):
        # A task cancelled before its first instruction never reaches its try block.
        if request_id not in self.terminals:
            if task.cancelled():
                self.terminal("cancelled", request_id,
                              cancellation_before_submit=request_id not in self.started,
                              backend_quiescence_confirmed=False)
            else:
                error = task.exception()
                self.terminal("failed", request_id,
                              error_type=type(error).__name__ if error else "MissingTerminal",
                              error=str(error) if error else "task exited without terminal evidence")
        elif not task.cancelled():
            # Retrieve exceptions even if logging/cleanup failed after a terminal.
            error = task.exception()
            if error is not None:
                self.cleanup_ok = False
                self.emit("cleanup_failed", request_id, phase="request_task",
                          error_type=type(error).__name__, error=str(error))
        self.active.pop(request_id, None)
        if not self.active:
            self.emit("idle", backend_quiescence_confirmed=False, **self.snapshot())

    def submit(self, command):
        request_id = command.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            self.reject(None, "request_id must be a nonempty string")
            return
        if self.closing:
            self.reject(request_id, "worker is closing")
            return
        if request_id in self.tasks:
            self.reject(request_id, "request_id was already submitted")
            return
        if len(self.active) >= LOCAL_SLOTS:
            self.reject(request_id, "all four local admission slots are occupied")
            return
        tokens = command.get("prompt_token_ids")
        length = command.get("max_output_tokens")
        if not isinstance(tokens, list) or not tokens or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            self.reject(request_id, "prompt_token_ids must be a nonempty list of nonnegative integers")
            return
        if type(length) is not int or length < 1 or len(tokens) + length > MAX_MODEL_LEN:
            self.reject(request_id, "invalid output length or total context exceeds 4096")
            return
        self.counts["submitted"] += 1
        task = asyncio.create_task(self.run_request(request_id, list(tokens), length))
        self.tasks[request_id] = task
        self.active[request_id] = task
        self.max_active = max(self.max_active, len(self.active))
        task.add_done_callback(lambda done, rid=request_id: self.task_done(rid, done))

    async def cancel(self, request_id):
        if not isinstance(request_id, str) or request_id not in self.tasks:
            self.reject(request_id if isinstance(request_id, str) else None,
                        "cancel request_id is unknown")
            return
        task = self.tasks[request_id]
        if task.done() or request_id in self.terminals:
            self.emit("cancel_ignored", request_id, reason="request is already terminal")
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def shutdown(self):
        if self.closed:
            return
        self.closing = True
        for task in tuple(self.active.values()):
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        # Let task_done callbacks publish terminal/idle evidence before closed.
        await asyncio.sleep(0)
        method_name = "shutdown" if callable(getattr(self.engine, "shutdown", None)) else "shutdown_background_loop"
        try:
            getattr(self.engine, method_name)()
        except Exception as exc:
            self.cleanup_ok = False
            self.emit("cleanup_failed", phase="engine_shutdown", method=method_name,
                      error_type=type(exc).__name__, error=str(exc))
        self.closed = True
        self.emit("closed", method=method_name, backend_process_exit_verified=False,
                  **self.snapshot())

    async def command(self, command):
        if not isinstance(command, dict):
            self.reject(None, "command must be a JSON object")
            return
        kind = command.get("kind", command.get("command"))
        if kind == "submit":
            self.submit(command)
        elif kind == "cancel":
            await self.cancel(command.get("request_id"))
        elif kind == "status":
            self.emit("status", **self.snapshot())
            if not self.active:
                self.emit("idle", backend_quiescence_confirmed=False, **self.snapshot())
        elif kind == "shutdown":
            await self.shutdown()
        else:
            request_id = command.get("request_id")
            self.reject(request_id if isinstance(request_id, str) else None,
                        "unknown command kind")


def sampling_kwargs(length, output_kind):
    return dict(n=1, temperature=0.0, top_p=1.0, seed=0,
                max_tokens=length, min_tokens=length, ignore_eos=True,
                stop=[], stop_token_ids=[], output_kind=output_kind)


def create_real_engine(model):
    # This function is invoked only inside run_main's running asyncio loop.
    import torch
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import RequestOutputKind

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must expose exactly one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    runtime = {
        "model": str(Path(model).resolve()),
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": False,
        "trust_remote_code": False,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "enable_chunked_prefill": True,
        "max_num_seqs": MAX_NUM_SEQS,
        "local_slots": LOCAL_SLOTS,
        "visible_device_count": torch.cuda.device_count(),
        "logical_device": 0,
        "device_name": properties.name,
        "total_memory": properties.total_memory,
        "device_capability": [properties.major, properties.minor],
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("vllm", "transformers", "tokenizers", "numpy")},
    }
    args = AsyncEngineArgs(
        model=runtime["model"], tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
        dtype="bfloat16", max_model_len=MAX_MODEL_LEN, enforce_eager=True,
        gpu_memory_utilization=0.8, enable_prefix_caching=False, trust_remote_code=False,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill=True, max_num_seqs=MAX_NUM_SEQS,
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    return engine, lambda length: SamplingParams(**sampling_kwargs(length, RequestOutputKind.FINAL_ONLY)), runtime


async def run_main(args, event_writer):
    worker = AsyncRequestWorker(None, None, event_writer, args.gpu_index)
    runtime = {
        "model": str(Path(args.model).resolve()),
        "gpu_index": args.gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_use_v1_requested": os.environ.get("VLLM_USE_V1"),
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": "bfloat16", "max_model_len": MAX_MODEL_LEN,
        "enforce_eager": True, "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": False, "local_slots": LOCAL_SLOTS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "enable_chunked_prefill": True, "max_num_seqs": MAX_NUM_SEQS,
        "offline_environment": {key: os.environ.get(key) for key in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
        )},
    }
    worker.emit("init_start", runtime=runtime,
                actual_engine_class=None, clock=time.get_clock_info("monotonic").implementation)
    exit_code = 0
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
            raise ValueError("CUDA_VISIBLE_DEVICES must equal the requested numeric physical index")
        if any(value != "1" for value in runtime["offline_environment"].values()):
            raise ValueError("all three Hugging Face offline environment flags must be 1")
        if not Path(args.model).is_dir():
            raise ValueError("model must be an existing local directory")
        worker.engine, worker.sampling_factory, observed_runtime = create_real_engine(args.model)
        runtime.update(observed_runtime)
        engine_class = type(worker.engine)
        actual_class = f"{engine_class.__module__}.{engine_class.__qualname__}"
        config = await worker.engine.get_vllm_config()
        runtime.update(
            actual_engine_class=actual_class,
            effective_dtype=str(config.model_config.dtype),
            effective_max_model_len=config.model_config.max_model_len,
            effective_tensor_parallel_size=config.parallel_config.tensor_parallel_size,
            effective_pipeline_parallel_size=config.parallel_config.pipeline_parallel_size,
            effective_data_parallel_size=config.parallel_config.data_parallel_size,
            effective_enforce_eager=config.model_config.enforce_eager,
            effective_gpu_memory_utilization=config.cache_config.gpu_memory_utilization,
            effective_enable_prefix_caching=config.cache_config.enable_prefix_caching,
            effective_max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
            effective_enable_chunked_prefill=config.scheduler_config.enable_chunked_prefill,
            effective_max_num_seqs=config.scheduler_config.max_num_seqs,
            vllm_use_v1_effective=os.environ.get("VLLM_USE_V1"),
        )
        expected = {
            "effective_dtype": "torch.bfloat16",
            "effective_max_model_len": MAX_MODEL_LEN,
            "effective_tensor_parallel_size": 1,
            "effective_pipeline_parallel_size": 1,
            "effective_data_parallel_size": 1,
            "effective_enforce_eager": True,
            "effective_gpu_memory_utilization": 0.8,
            "effective_enable_prefix_caching": False,
            "effective_max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "effective_enable_chunked_prefill": True,
            "effective_max_num_seqs": MAX_NUM_SEQS,
            "actual_engine_class": "vllm.v1.engine.async_llm.AsyncLLM",
        }
        mismatches = {key: {"expected": value, "actual": runtime[key]}
                      for key, value in expected.items() if runtime[key] != value}
        if mismatches:
            raise ValueError(f"effective engine configuration mismatch: {mismatches}")
        worker.emit("init_done", runtime=runtime, actual_engine_class=actual_class)
        worker.emit("ready", runtime=runtime, actual_engine_class=actual_class,
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
        worker.emit("fatal", error_type=type(exc).__name__, error=str(exc),
                    traceback=traceback.format_exc())
    finally:
        if worker.engine is not None:
            await worker.shutdown()
        else:
            worker.emit("closed", initialization_failed=True, cleanup_ok=True,
                        backend_process_exit_verified=False)
    return exit_code if worker.cleanup_ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()
    writer = EventWriter(args.events)
    try:
        return asyncio.run(run_main(args, writer))
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
