"""Independent process supervisor; never creates a worker terminal record."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

from .runtime import (RunContext, append_json, durable_json, file_sha256,
                      no_symlinks, validate_output_root)


@dataclass(frozen=True)
class Limits:
    sample_seconds: float = 5.0
    rss_bytes: int = 4 * 1024**3
    min_disk_free_bytes: int = 10 * 1024**3
    min_available_ram_bytes: int = 2 * 1024**3
    heartbeat_timeout_seconds: float = 60.0
    startup_grace_seconds: float = 60.0
    warn_seconds: float = 10 * 3600.0
    timeout_seconds: float = 12 * 3600.0
    stop_grace_seconds: float = 30.0


def _owned_tree(process):
    try:
        root = psutil.Process(process.pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def _signal_tree(process, *, kill=False):
    # POSIX start_new_session gives this worker its own process group. Windows
    # uses psutil's owned process tree; no unrelated PID is ever targeted.
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        for child in reversed(_owned_tree(process)):
            try:
                child.kill() if kill else child.terminate()
            except psutil.NoSuchProcess:
                pass


def _cleanup_owned(process, grace_seconds):
    """Terminate and reap after a monitor failure, independent of disk writes."""
    descendants = _owned_tree(process)[1:]
    terminated = killed = False
    if process.poll() is None or descendants:
        _signal_tree(process)
        terminated = True
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_tree(process, kill=True)
            killed = True
            process.wait(timeout=max(5, grace_seconds))
    # A parent may exit before its descendants. Retain original psutil process
    # identities so cleanup does not accidentally target reused/unrelated PIDs.
    _, alive = psutil.wait_procs(descendants, timeout=grace_seconds)
    for child in alive:
        try:
            child.kill()
            killed = True
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=max(5, grace_seconds))
    process.wait()
    return terminated, killed


def _best_effort(action, errors, label):
    try:
        return action()
    except Exception as error:
        errors.append({"operation": label, "error": repr(error)})
        # If storage is exhausted this stderr message is the only remaining
        # observation; never claim the failed result/journal write succeeded.
        print(f"supervisor persistence failure ({label}): {error!r}", file=sys.stderr)
        return None


def supervise(argv, cwd, run_dir, supervisor_dir, limits=None, *, disk_free_probe=None):
    """Run and monitor one child; disk_free_probe is explicit fault injection.

    Cooperative workers watch run_dir/STOP_REQUEST.json. Nonresponsive workers
    receive terminate then kill after two stop-grace intervals. Heartbeat paths
    are discovered only inside this run's segment directory.
    """
    limits = limits or Limits()
    if limits.sample_seconds <= 0 or limits.stop_grace_seconds <= 0:
        raise ValueError("sampling and stop grace must be positive")
    cwd, run_dir = no_symlinks(cwd), no_symlinks(run_dir)
    validate_output_root(cwd, run_dir)
    validate_output_root(cwd, supervisor_dir)
    supervisor_dir = no_symlinks(supervisor_dir)
    supervisor_dir.mkdir(parents=True, exist_ok=False)
    planned = {"argv": list(argv), "cwd": str(cwd.resolve()), "run_dir": str(run_dir),
               "limits": asdict(limits), "created_unix": time.time(),
               "disk_free_fault_injected": disk_free_probe is not None,
               "gpu_metrics": "not_produced_cpu_only_supervisor"}
    durable_json(supervisor_dir / "supervisor_planned.json", planned, exclusive=True)
    journal = supervisor_dir / "resources.jsonl"
    started = time.monotonic()
    wall_started = time.time()
    trigger = None
    triggered_at = None
    terminated = killed = warned = False
    stop_written = False
    monitor_error = None
    persistence_errors = []
    observed_peak = 0
    previous_log_size = 0
    stdout_path = supervisor_dir / "stdout.log"
    stderr_path = supervisor_dir / "stderr.log"
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(list(argv), cwd=cwd, stdout=stdout, stderr=stderr, **options)
        except Exception as error:
            durable_json(supervisor_dir / "supervisor_result.json",
                         {"technical_status": "launch_failed", "reason": repr(error),
                          "exit_code": None, "worker_terminal_observed": False}, exclusive=True)
            raise
        try:
            append_json(journal, {"event": "launched", "pid": process.pid, "time": time.time()})
            previous_sample = started
            while process.poll() is None:
                now = time.monotonic()
                rss, cpu = 0, 0.0
                for owned in _owned_tree(process):
                    try:
                        rss += owned.memory_info().rss
                        cpu += sum(owned.cpu_times()[:2])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                observed_peak = max(observed_peak, rss)
                available_ram = psutil.virtual_memory().available
                disk_free = (disk_free_probe(supervisor_dir) if disk_free_probe
                             else shutil.disk_usage(supervisor_dir).free)
                heartbeats = list((run_dir / "segments").glob("*/heartbeat.json"))
                # A resumed run must emit a fresh heartbeat; previous segment times
                # do not satisfy startup liveness.
                current_heartbeats = []
                for path in heartbeats:
                    heartbeat = json.loads(no_symlinks(path).read_text())
                    if heartbeat["pid"] == process.pid:
                        current_heartbeats.append(heartbeat["time"])
                heartbeat_time = max(current_heartbeats, default=wall_started)
                heartbeat_age = time.time() - max(heartbeat_time, wall_started)
                log_size = stdout_path.stat().st_size + stderr_path.stat().st_size
                log_size += sum(p.stat().st_size for p in run_dir.rglob("*.jsonl"))
                append_json(journal, {"event": "sample", "time": time.time(),
                                      "elapsed_seconds": now - started, "rss_bytes": rss,
                                      "cpu_seconds": cpu, "available_ram_bytes": available_ram,
                                      "disk_free_bytes": disk_free, "heartbeat_age_seconds": heartbeat_age,
                                      "log_bytes": log_size,
                                      "log_bytes_per_second": (log_size - previous_log_size) /
                                      max(now - previous_sample, 1e-9)})
                previous_log_size, previous_sample = log_size, now
                if not warned and now - started >= limits.warn_seconds:
                    warned = True
                    append_json(journal, {"event": "time_budget_warning", "time": time.time()})
                if trigger is None:
                    if rss > limits.rss_bytes:
                        trigger = "rss_limit"
                    elif disk_free < limits.min_disk_free_bytes:
                        trigger = "disk_reserve"
                    elif available_ram < limits.min_available_ram_bytes:
                        trigger = "ram_reserve"
                    elif now - started >= limits.timeout_seconds:
                        trigger = "timeout"
                    elif (current_heartbeats and heartbeat_age > limits.heartbeat_timeout_seconds):
                        trigger = "heartbeat_timeout"
                    elif not current_heartbeats and now - started >= limits.startup_grace_seconds:
                        trigger = "startup_timeout"
                    if trigger:
                        triggered_at = now
                        append_json(journal, {"event": "stop_requested", "reason": trigger, "time": time.time()})
                if trigger:
                    # Do not create run_dir before the worker's exclusive create.
                    if run_dir.is_dir() and not stop_written:
                        durable_json(run_dir / "STOP_REQUEST.json",
                                     {"reason": trigger, "time": time.time(), "target_pid": process.pid})
                        stop_written = True
                    if not terminated and now - triggered_at >= limits.stop_grace_seconds:
                        _signal_tree(process)
                        terminated = True
                        append_json(journal, {"event": "terminate_sent", "time": time.time()})
                    if not killed and now - triggered_at >= 2 * limits.stop_grace_seconds:
                        _signal_tree(process, kill=True)
                        killed = True
                        append_json(journal, {"event": "kill_sent", "time": time.time()})
                time.sleep(limits.sample_seconds)
            exit_code = process.wait()
        except BaseException as error:
            monitor_error = {"type": type(error).__name__, "message": repr(error)}
        finally:
            if monitor_error is not None or process.poll() is None:
                stopped, forced = _cleanup_owned(process, limits.stop_grace_seconds)
                terminated = terminated or stopped
                killed = killed or forced
            exit_code = process.wait()
    worker_results = list((run_dir / "segments").glob("*/worker_result.json"))
    current_results = [p for p in worker_results if p.stat().st_mtime >= wall_started]
    recovery = {"observer": "independent_supervisor", "worker_pid": process.pid,
                "observed_state": "worker_terminal_present" if current_results else "partial",
                "worker_terminal_observed": bool(current_results),
                "latest_complete_checkpoint": None, "checkpoint_validation_error": None,
                "reason": "monitor_error" if monitor_error else (trigger or "process_exit"),
                "monitor_error": monitor_error, "exit_code": exit_code}
    planned_path = run_dir / "planned_manifest.json"
    if planned_path.exists() and (run_dir / "latest_checkpoint.json").exists():
        try:
            context = RunContext(run_dir, json.loads(planned_path.read_text()))
            checkpoint = context.latest_checkpoint()
            recovery["latest_complete_checkpoint"] = {
                "path": checkpoint.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(checkpoint)}
        except (ValueError, KeyError, OSError) as error:
            recovery["checkpoint_validation_error"] = repr(error)
    _best_effort(lambda: durable_json(supervisor_dir / "recovery_observation.json", recovery, exclusive=True),
                 persistence_errors, "recovery_observation")
    result = {"technical_status": "stopped" if trigger else ("exited_zero" if exit_code == 0 else "abnormal_exit"),
              "reason": trigger or ("process_exit" if exit_code == 0 else "worker_exception_or_signal"),
              "exit_code": exit_code, "pid": process.pid,
              "worker_terminal_observed": bool(current_results),
              "recovery_observation_sha256": _best_effort(
                  lambda: file_sha256(supervisor_dir / "recovery_observation.json"), persistence_errors,
                  "recovery_observation_checksum"),
              "worker_terminal_sha256": {str(p.relative_to(run_dir)): file_sha256(p) for p in current_results},
              "observed_peak_rss_bytes": observed_peak,
              "elapsed_seconds": time.monotonic() - started,
              "terminate_sent": terminated, "kill_sent": killed,
              "stdout_sha256": file_sha256(stdout_path), "stderr_sha256": file_sha256(stderr_path),
              "scientific_status": "not_assessed", "claim_status": "not_eligible",
              "monitor_error": monitor_error, "persistence_errors": persistence_errors}
    if monitor_error is not None:
        result["technical_status"] = "monitor_interrupted" if monitor_error["type"] == "KeyboardInterrupt" else "monitor_failed"
        result["reason"] = "monitor_error"
    _best_effort(lambda: append_json(journal, {"event": "process_exited", "exit_code": exit_code,
                                            "time": time.time(), "monitor_error": monitor_error}),
                 persistence_errors, "exit_journal")
    if persistence_errors and result["technical_status"] == "exited_zero":
        result["technical_status"] = "persistence_failed"
    result_path = supervisor_dir / "supervisor_result.json"
    saved = _best_effort(lambda: (durable_json(result_path, result, exclusive=True), True)[1],
                         persistence_errors, "supervisor_result")
    result["supervisor_result_saved"] = saved is True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--supervisor-dir", required=True)
    parser.add_argument("--limits-json")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    limits = Limits(**json.loads(Path(args.limits_json).read_text())) if args.limits_json else Limits()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        parser.error("worker argv required after --")
    result = supervise(argv, args.cwd, args.run_dir, args.supervisor_dir, limits)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["technical_status"] == "exited_zero" else 1


if __name__ == "__main__":
    raise SystemExit(main())
