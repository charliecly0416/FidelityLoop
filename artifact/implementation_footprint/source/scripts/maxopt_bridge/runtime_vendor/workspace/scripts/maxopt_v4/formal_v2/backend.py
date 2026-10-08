"""Plan-configured backend using the audited ownership and cleanup primitive."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time

from scripts.maxopt_v4.w2_gpu_all2 import backend as legacy
from scripts.maxopt_v4.w2_gpu_all2 import evidence
from scripts.maxopt_v4.formal_v2.worker import validate_runtime

save = legacy.save
sha = legacy.sha


def expected_effective_runtime(runtime: dict) -> dict:
    return {
        "actual_engine_class": runtime["engine_class"],
        "effective_dtype": "torch." + runtime["dtype"],
        "effective_max_model_len": runtime["max_model_len"],
        "effective_tensor_parallel_size": 1, "effective_pipeline_parallel_size": 1,
        "effective_data_parallel_size": 1, "effective_enforce_eager": True,
        "effective_gpu_memory_utilization": 0.8, "effective_enable_prefix_caching": False,
        "effective_max_num_batched_tokens": 16384, "effective_enable_chunked_prefill": True,
        "effective_max_num_seqs": 1024,
    }


class AsyncWorkerBackend(legacy.AsyncWorkerBackend):
    """Retains the accepted generation ownership/cleanup implementation.

    Only startup is overridden: the child entry point and ready-time runtime
    contract are selected from the immutable formal_v2 plan instead of the
    Llama-only constants in the legacy backend.
    """

    async def _start(self, context):
        runtime = validate_runtime(self.plan["runtime"])
        gpu, generation, folder = context["gpu"], context["generation"], context["folder"]
        deadline = time.monotonic() + self.plan["limits"]["setup_seconds"]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONUNBUFFERED="1", VLLM_USE_V1="1")
        argv = [self.plan["python"], "-B", str((Path(__file__).parent / "worker.py").resolve()),
                "--gpu-index", str(gpu), "--generation", str(generation), "--model", self.plan["model"],
                # Bind the worker to the exact V3-template mapping in the
                # plan.  ``validate_runtime`` performs the one-way internal
                # normalization in both parent and child.
                "--runtime-json", json.dumps(self.plan["runtime"], sort_keys=True, separators=(",", ":")),
                "--events", str(folder / "events.jsonl")]
        context["stdout"], context["stderr"] = await asyncio.to_thread(
            legacy._prepare_start_files, folder)
        process = await self._create_process(argv, env, context["stdout"], context["stderr"])
        context["process"] = process
        try:
            # /proc traversal is synchronous.  Keep ownership capture off the
            # controller event loop so a generation launch cannot delay the
            # one-second observation clock.
            context["ownership"] = await asyncio.to_thread(
                evidence.capture_group_ownership, process.pid)
        except BaseException as exc:
            await self._terminate_unowned_process(context, exc, deadline)
            raise
        self.controller.actuator.spawned(gpu, generation, process.pid, context["ownership"], argv=argv,
                                         environment={key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
                                         "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "VLLM_USE_V1", "VLLM_NO_USAGE_STATS")})
        context["pump"] = self.schedule(self._pump(context))
        context["writer"] = self.schedule(self._writer(context))
        await self._until(context, lambda: any(event["kind"] == "ready" for event in context["events"]),
                          deadline, "engine ready")
        ready = next(event for event in context["events"] if event["kind"] == "ready")
        observed = ready["runtime"]
        differences = {key: (value, observed.get(key)) for key, value in expected_effective_runtime(runtime).items()
                       if observed.get(key) != value}
        if differences:
            raise ValueError("effective runtime differs from plan: " + repr(differences))
        probe_id = f"setup_g{gpu}_gen{generation}"
        probe = self.plan["setup_probe"]["prompt_token_ids"]
        await self._send(context, {"command": "submit", "request_id": probe_id,
                                   "prompt_token_ids": probe, "max_output_tokens": 32})
        await self._until(context, lambda: any(event["kind"] in ("finished", "failed", "cancelled")
                          and event["request_id"] == probe_id for event in context["events"]), deadline, "health probe")
        terminal = next(event for event in context["events"] if event["kind"] in ("finished", "failed", "cancelled")
                        and event["request_id"] == probe_id)
        if terminal["kind"] != "finished" or terminal.get("prompt_token_ids") != probe or len(terminal.get("output_token_ids", [])) != 32:
            raise ValueError("health probe evidence differs")
        marker = len(context["events"])
        await self._send(context, {"command": "status"})
        await self._until(context, lambda: any(event["kind"] == "status" and event.get("active_count") == 0
                          and event.get("active_request_ids") == [] for event in context["events"][marker:]), deadline,
                          "empty backend")
        empty = next(event for event in context["events"][marker:] if event["kind"] == "status" and event.get("active_count") == 0)
        snapshot = await asyncio.to_thread(evidence.snapshot_gpu, 5)
        members = await asyncio.to_thread(evidence.process_group_members, process.pid, True)
        expected_uuid = self.controller.gpu_uuid(gpu)
        apps = snapshot["compute_apps"]["rows"]
        if (snapshot["inventory"]["returncode"] != 0 or snapshot["compute_apps"]["returncode"] != 0
                or {row["index"]: row["uuid"] for row in snapshot["inventory"]["rows"]}.get(gpu) != expected_uuid
                or not any(row["gpu_uuid"] == expected_uuid for row in apps)
                or any(row["pid"] not in {member["pid"] for member in members}
                       for row in apps if row["gpu_uuid"] == expected_uuid)):
            raise ValueError("ready GPU process ownership/identity unproven")
        ready_evidence = await asyncio.to_thread(
            legacy._persist_ready_evidence, folder, snapshot, members, self.out)
        self.controller.actuator.ready(gpu, generation, health_probe_completed=True, queues_empty=True,
             health_probe_request_id=probe_id, probe_worker_seq=terminal["worker_seq"],
             empty_status_worker_seq=empty["worker_seq"], runtime=observed, evidence=ready_evidence)
