"""Bounded host evidence and owned-session cleanup; standard library only."""
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


PROC_ROOT = Path("/proc")
RUNTIME_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_WORKER_MULTIPROC_METHOD",
               "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _number(value):
    try:
        return int(value)
    except ValueError:
        return None


def _query_gpu(fields, query, timeout):
    command = ["nvidia-smi", "--query-{}={}".format(query, ",".join(fields)),
               "--format=csv,noheader,nounits"]
    result = {"command": command, "stdout": "", "stderr": "", "returncode": None,
              "timed_out": False, "rows": []}
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=timeout, check=False)
        result.update(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        result.update(stdout=text(exc.stdout), stderr=text(exc.stderr), timed_out=True)
    except OSError as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    if result["returncode"] == 0:
        for values in csv.reader(io.StringIO(result["stdout"]), skipinitialspace=True):
            if not values:
                continue
            if len(values) != len(fields):
                result.setdefault("parse_errors", []).append(values)
                continue
            raw = dict(zip(fields, (v.strip() for v in values)))
            if query == "gpu":
                row = {"index": _number(raw["index"]), "uuid": raw["uuid"], "name": raw["name"],
                       "driver_version": raw["driver_version"],
                       "memory_total_mib": _number(raw["memory.total"]),
                       "memory_used_mib": _number(raw["memory.used"])}
            else:
                row = {"pid": _number(raw["pid"]), "gpu_uuid": raw["gpu_uuid"],
                       "process_name": raw["process_name"],
                       "used_memory_mib": _number(raw["used_gpu_memory"])}
            result["rows"].append(row)
    return result


def snapshot_gpu(timeout=5):
    """Two read-only queries, each limited to timeout/2 (combined budget timeout)."""
    if not 0 < timeout <= 60:
        raise ValueError("GPU query timeout must be in (0,60] seconds")
    return {
        "captured_monotonic": time.monotonic(),
        "command_timeout_seconds": timeout / 2,
        "combined_command_timeout_seconds": timeout,
        "inventory": _query_gpu(("index", "uuid", "name", "driver_version", "memory.total", "memory.used"), "gpu", timeout / 2),
        "compute_apps": _query_gpu(("pid", "gpu_uuid", "process_name", "used_gpu_memory"), "compute-apps", timeout / 2),
    }


def _read_process(pid):
    """Parse /proc stat after the LAST ')' because comm may contain spaces/)."""
    try:
        text = (PROC_ROOT / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    head, tail = text.rsplit(")", 1)
    fields = tail.split()
    return {"pid": int(head.split("(", 1)[0].strip()), "state": fields[0],
            "pgrp": int(fields[2]), "session": int(fields[3]), "starttime": int(fields[19])}


def process_group_members(pgid, include_session=False):
    """Return group members including Z; optionally the whole owned session."""
    if not isinstance(pgid, int) or pgid <= 1:
        raise ValueError("invalid process group")
    rows = []
    for path in PROC_ROOT.iterdir():
        if path.name.isdecimal():
            row = _read_process(int(path.name))
            if row is not None and (row["pgrp"] == pgid or (include_session and row["session"] == pgid)):
                rows.append(row)
    return sorted(rows, key=lambda row: row["pid"])


def capture_group_ownership(pgid):
    """Call immediately after Popen(start_new_session=True), before waiting."""
    leader = _read_process(pgid)
    if leader is None or leader["pgrp"] != pgid or leader["session"] != pgid:
        raise ValueError("worker is not a live dedicated session leader")
    return {"pgid": pgid, "leader_starttime": leader["starttime"], "session": pgid,
            "captured_monotonic": time.monotonic()}


def _same_identity(a, b):
    return b is not None and all(a[k] == b[k] for k in ("pid", "pgrp", "session", "starttime"))


def _signal_member(row, sig):
    """Prefer pidfd; older hosts use an immediate checked, non-atomic signal."""
    result = {"pid": row["pid"], "starttime": row["starttime"], "signal": sig.name}
    fd = None
    try:
        if callable(getattr(os, "pidfd_open", None)) and callable(getattr(signal, "pidfd_send_signal", None)):
            fd = os.pidfd_open(row["pid"], 0)
            if not _same_identity(row, _read_process(row["pid"])):
                result["status"] = "identity_changed_not_signaled"
            else:
                signal.pidfd_send_signal(fd, sig)
                result["status"] = "sent_pidfd"
        else:
            if not _same_identity(row, _read_process(row["pid"])):
                result["status"] = "identity_changed_not_signaled"
            else:
                os.kill(row["pid"], sig)
                result["status"] = "sent_identity_checked_nonatomic"
                result["limitation"] = "pidfd unavailable; a small /proc-check to os.kill PID-reuse race remains"
    except ProcessLookupError:
        result["status"] = "already_exited"
    except OSError as exc:
        result.update(status="signal_error", error=str(exc))
    finally:
        if fd is not None:
            os.close(fd)
    return result


def cleanup_group(pgid, deadline_monotonic, ownership):
    """Bounded TERM then KILL for a previously captured dedicated worker group.

    Call even after the worker leader exits. Include children in other process
    groups within the captured session. Descendants that create a new session
    are outside this primitive and must be reported by the caller's
    GPU/process evidence; this function never guesses ownership of other groups.
    """
    if (ownership.get("pgid") != pgid or ownership.get("session") != pgid
            or not isinstance(ownership.get("leader_starttime"), int) or pgid <= 1
            or pgid in (os.getpgrp(), os.getpid())):
        raise ValueError("invalid/unsafe initial ownership token")
    before = process_group_members(pgid, include_session=True)
    result = {"pgid": pgid, "ownership": ownership, "members_before": before,
              "members_after": before, "signals": [], "complete": False,
              "deadline_exhausted": False, "ownership_rejected": False}
    term_until = min(deadline_monotonic, time.monotonic() + 1.0)
    sent = set()
    while time.monotonic() < deadline_monotonic:
        rows = process_group_members(pgid, include_session=True)
        live = [r for r in rows if r["state"] != "Z"]
        result["members_after"] = rows
        if not live:
            result["complete"] = True
            return result
        # If the group leader PID was reused, no member of that new session is ours.
        leader = _read_process(pgid)
        invalid_leader = leader is not None and leader["starttime"] != ownership["leader_starttime"]
        invalid_members = any(r["session"] != pgid or r["starttime"] < ownership["leader_starttime"] for r in live)
        if invalid_leader or invalid_members:
            result["ownership_rejected"] = True
            return result
        sig = signal.SIGTERM if time.monotonic() < term_until else signal.SIGKILL
        for row in live:
            if time.monotonic() >= deadline_monotonic:
                break
            key = (row["pid"], row["starttime"], sig)
            if key not in sent:
                result["signals"].append(_signal_member(row, sig))
                sent.add(key)
        remaining = deadline_monotonic - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.05, remaining))
    result["members_after"] = process_group_members(pgid, include_session=True)
    result["complete"] = not any(r["state"] != "Z" for r in result["members_after"])
    result["deadline_exhausted"] = not result["complete"]
    return result


def _file_binding(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    after = path.stat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) for k in keys):
        raise RuntimeError("file changed while hashing: " + str(path))
    return {"name": path.name, "path": str(path.resolve()), "bytes": after.st_size,
            "sha256": digest.hexdigest(), "stat_before_after_match": True,
            "stat": {k: getattr(after, k) for k in keys}}


def environment_model_snapshot(model_path, reference_path=None):
    """Hash the complete recursive local model closure and runtime environment."""
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError("model path is not a directory")
    selected = sorted((p for p in model_path.rglob("*") if p.is_file()),
                      key=lambda p: p.relative_to(model_path).as_posix())
    if not selected:
        raise ValueError("model directory has no files")
    files = []
    for path in selected:
        binding = _file_binding(path)
        binding["name"] = path.relative_to(model_path).as_posix()
        files.append(binding)
    closure_manifest = [{"path": f["name"], "bytes": f["bytes"], "sha256": f["sha256"]}
                        for f in files]
    closure_sha = hashlib.sha256(
        json.dumps(closure_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    packages = {}
    for name in ("torch", "vllm", "transformers", "tokenizers", "safetensors", "numpy", "ray", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    clock = time.get_clock_info("monotonic")
    result = {"schema": "maxopt-v4-w2-environment-model-v2",
              "python": {"executable": sys.executable, "version": sys.version},
              "platform": platform.platform(), "packages": packages,
              "runtime_env": {k: os.environ[k] for k in RUNTIME_ENV if k in os.environ},
              "monotonic_clock": {k: getattr(clock, k) for k in ("implementation", "monotonic", "adjustable", "resolution")},
              "model_path": str(model_path.resolve()), "model_files": files,
              "model_closure_manifest": closure_manifest,
              "model_closure_sha256": closure_sha,
              "model_bytes": sum(f["bytes"] for f in files), "full_weight_hashes": True,
              "captured_monotonic": time.monotonic()}
    if reference_path is not None:
        reference_path = Path(reference_path)
        result["reference"] = _file_binding(reference_path)
        try:
            reference = json.loads(reference_path.read_text())
            expected = {f["name"]: f["sha256"] for f in reference["model_files"]}
            actual = {f["name"]: f["sha256"] for f in files}
            result["reference"]["model_files_match"] = expected == actual
        except (ValueError, KeyError, TypeError):
            result["reference"]["model_files_match"] = None
            result["reference"]["comparison_reason"] = "reference lacks compatible model_files name/sha256 schema"
    return result
