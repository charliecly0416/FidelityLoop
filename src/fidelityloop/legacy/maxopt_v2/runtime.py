"""Small, durable run/segment protocol for V2 CPU diagnostics.

Checkpoints contain application-owned JSON state, not a PPO/RNG implementation.
Scientific acceptance is deliberately outside this module.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
import zipfile


SCHEMA = "maxopt-v2-runtime-1"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def no_symlinks(path):
    """Reject links before resolve(), including dangling links and ancestor links."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symbolic link prohibited: {part}")
    return path


def durable_json(path, value, *, exclusive=False):
    path = no_symlinks(path)
    data = canonical_bytes(value)
    if exclusive:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    # Directory durability is supported on POSIX; Windows has no equivalent fd.
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def append_json(path, value):
    path = no_symlinks(path)
    with path.open("ab") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def read_journal(path, *, allow_truncated=False):
    """Read durable records; only a trailing incomplete record may be ignored."""
    path = no_symlinks(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if allow_truncated and index == len(lines) - 1:
                break
            raise ValueError(f"incomplete journal: {path}")
        result.append(json.loads(line))
    return result


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def canonical_v2_source_files(repo):
    """Declared N1 -> runtime/engine source closure for the integrated V2 runner."""
    repo = Path(repo).resolve()
    files = sorted((repo / "scripts/maxopt_v2").glob("*.py"))
    generator = repo / "scripts/maxopt_n1_generate_workload.py"
    if generator.exists():
        files.append(generator)
    return files


def identity(repo, config, inputs, code_files):
    repo = no_symlinks(repo).resolve()
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("repo must be the actual git root")
    branch = _git(repo, "branch", "--show-current")
    if branch != "research-max-optimization-20260915":
        raise ValueError("V2 execution requires the isolated research branch")
    code_files = list(code_files)
    if not code_files:
        raise ValueError("code_files must declare the actual runtime dependencies")
    sources = {}
    for filename in [*code_files, Path(__file__), Path(__file__).with_name("__init__.py")]:
        path = no_symlinks(filename).resolve()
        sources[path.relative_to(repo).as_posix()] = file_sha256(path)
    input_hashes = {str(name): file_sha256(no_symlinks(path))
                    for name, path in inputs.items()}
    return {"repo": str(repo), "branch": branch,
            "head": _git(repo, "rev-parse", "HEAD"),
            "config_sha256": sha256_bytes(canonical_bytes(config)),
            "input_sha256": input_hashes, "code_sha256": sources,
            "source_set_content_sha256": sha256_bytes(canonical_bytes(sources))}


def validate_output_root(repo, output_root):
    repo = no_symlinks(repo).resolve()
    output = no_symlinks(output_root).resolve()
    relative = output.relative_to(repo / "artifacts")
    if not relative.parts or not relative.parts[0].startswith("max_optimization_v2_"):
        raise ValueError("output must be in the dedicated V2 artifacts root")
    return output


class RunContext:
    """Single worker owns each run; supervisor writes only its separate journal."""

    def __init__(self, run_dir, planned):
        self.run_dir = no_symlinks(run_dir)
        self.planned = planned
        self.segment = None
        self._event_ids = {}
        for path in sorted((self.run_dir / "segments").glob("*/events.jsonl")):
            for event in read_journal(path, allow_truncated=True):
                if "event_id" in event:
                    event_id = event["event_id"]
                    if event_id in self._event_ids:
                        raise ValueError(f"duplicate persisted event: {event_id}")
                    self._event_ids[event_id] = event

    @classmethod
    def create(cls, repo, output_root, run_id, config, inputs, code_files,
               metadata=None, argv=None):
        if Path.cwd().resolve() != Path(repo).resolve():
            raise ValueError("execution cwd must equal the isolated V2 git root")
        if not NAME.fullmatch(run_id):
            raise ValueError("invalid run_id")
        output = validate_output_root(repo, output_root)
        provenance = identity(repo, config, inputs, code_files)
        run_dir = output / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "segments").mkdir()
        environment = {"python_executable": sys.executable, "python_version": sys.version,
                       "psutil_version": version("psutil")}
        planned = {"schema": SCHEMA, "run_id": run_id, "state": "planned",
                   "created_unix": time.time(), "actual_cwd": str(Path.cwd().resolve()),
                   "argv": list(sys.argv if argv is None else argv),
                   "python": sys.version, "platform": platform.platform(),
                   "identity": provenance, "config": config,
                   "input_paths": {str(k): str(Path(v).resolve()) for k, v in inputs.items()},
                   "metadata": metadata or {},
                   "checkpoint_kind": "application_json_state_not_ppo",
                   "environment": environment,
                   "environment_sha256": sha256_bytes(canonical_bytes(environment))}
        durable_json(run_dir / "planned_manifest.json", planned, exclusive=True)
        append_json(run_dir / "status.jsonl", {"state": "planned", "time": time.time()})
        return cls(run_dir, planned)

    @classmethod
    def open(cls, run_dir, repo, config, inputs, code_files):
        if Path.cwd().resolve() != Path(repo).resolve():
            raise ValueError("execution cwd must equal the isolated V2 git root")
        run_dir = no_symlinks(run_dir)
        validate_output_root(repo, run_dir)
        planned = json.loads((run_dir / "planned_manifest.json").read_text())
        if planned.get("schema") != SCHEMA:
            raise ValueError("unsupported planned manifest schema")
        if planned["identity"] != identity(repo, config, inputs, code_files):
            raise ValueError("resume identity mismatch: code/config/input/branch/HEAD")
        if (run_dir / "result_manifest.json").exists():
            terminal = json.loads((run_dir / "result_manifest.json").read_text())
            if terminal["technical_status"] == "completed":
                raise ValueError("completed run cannot be resumed")
        return cls(run_dir, planned)

    def new_segment(self, parent_checkpoint=None):
        if self.segment is not None:
            raise ValueError("context already owns a segment")
        parent_sha = None
        if parent_checkpoint is not None:
            self.load_checkpoint(parent_checkpoint)
            parent_sha = file_sha256(parent_checkpoint)
        # UUID directories cannot overwrite a previous failed attempt.
        self.segment = self.run_dir / "segments" / uuid.uuid4().hex
        self.segment.mkdir()
        (self.segment / "checkpoints").mkdir()
        durable_json(self.segment / "segment_manifest.json",
                     {"schema": SCHEMA, "created_unix": time.time(),
                      "segment_id": self.segment.name, "pid": os.getpid(),
                      "parent_checkpoint_sha256": parent_sha,
                      "actual_cwd": str(Path.cwd().resolve()), "argv": sys.argv},
                     exclusive=True)
        append_json(self.run_dir / "status.jsonl",
                    {"state": "preflight_passed", "segment": self.segment.name,
                     "time": time.time()})
        append_json(self.run_dir / "status.jsonl",
                    {"state": "running", "segment": self.segment.name, "time": time.time()})
        self.heartbeat()
        return self.segment

    def _require_segment(self):
        if self.segment is None:
            raise ValueError("new_segment must be called first")
        if (self.segment / "worker_result.json").exists():
            raise ValueError("worker segment is already terminal")

    def heartbeat(self):
        self._require_segment()
        durable_json(self.segment / "heartbeat.json", {"time": time.time(), "pid": os.getpid()})

    def stop_requested(self):
        """Old supervisor requests must not stop a newly resumed process."""
        path = self.run_dir / "STOP_REQUEST.json"
        if not path.exists():
            return False
        request = json.loads(no_symlinks(path).read_text())
        return request.get("target_pid") == os.getpid()

    def append_event(self, event):
        self._require_segment()
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id is required for idempotent replay")
        if event_id in self._event_ids:
            if self._event_ids[event_id] != event:
                raise ValueError("conflicting event payload for existing event_id")
            return False
        append_json(self.segment / "events.jsonl", event)
        self._event_ids[event_id] = json.loads(canonical_bytes(event))
        return True

    def append_metric(self, metric):
        self._require_segment()
        append_json(self.segment / "metrics.jsonl", metric)

    def checkpoint(self, state):
        self._require_segment()
        payload = {"schema": SCHEMA, "state": state,
                   "identity": self.planned["identity"],
                   "planned_manifest_sha256": file_sha256(self.run_dir / "planned_manifest.json"),
                   "segment_id": self.segment.name, "created_unix": time.time()}
        envelope = {"payload": payload, "payload_sha256": sha256_bytes(canonical_bytes(payload))}
        checkpoint = self.segment / "checkpoints" / (uuid.uuid4().hex + ".json")
        durable_json(checkpoint, envelope)
        durable_json(self.run_dir / "latest_checkpoint.json",
                     {"path": checkpoint.relative_to(self.run_dir).as_posix(),
                      "sha256": file_sha256(checkpoint)})
        return checkpoint

    def latest_checkpoint(self):
        latest = json.loads((self.run_dir / "latest_checkpoint.json").read_text())
        member = _safe_member(latest["path"])
        path = no_symlinks(self.run_dir / member)
        if file_sha256(path) != latest["sha256"]:
            raise ValueError("latest checkpoint file checksum mismatch")
        self.load_checkpoint(path)
        return path

    def load_checkpoint(self, path):
        path = no_symlinks(path)
        path.relative_to(self.run_dir)
        envelope = json.loads(path.read_text())
        payload = envelope["payload"]
        if sha256_bytes(canonical_bytes(payload)) != envelope["payload_sha256"]:
            raise ValueError("checkpoint payload checksum mismatch")
        if payload["schema"] != SCHEMA or payload["identity"] != self.planned["identity"]:
            raise ValueError("checkpoint code/config/input identity mismatch")
        if payload["planned_manifest_sha256"] != file_sha256(self.run_dir / "planned_manifest.json"):
            raise ValueError("checkpoint planned manifest mismatch")
        return payload["state"]

    def finish(self, status, reason, *, scientific_status="diagnostic_only",
               claim_status="not_eligible", summary=None):
        self._require_segment()
        if status not in {"completed", "partial", "failed"}:
            raise ValueError("invalid technical terminal state")
        if scientific_status != "diagnostic_only" or claim_status != "not_eligible":
            raise ValueError("independent acceptance cannot be granted by worker flags")
        result = {"schema": SCHEMA, "run_id": self.planned["run_id"],
                  "segment_id": self.segment.name, "technical_status": status,
                  "reason": str(reason), "scientific_status": scientific_status,
                  "claim_status": claim_status, "summary": summary or {},
                  "planned_manifest_sha256": file_sha256(self.run_dir / "planned_manifest.json"),
                  "finished_unix": time.time()}
        durable_json(self.segment / "worker_result.json", result, exclusive=True)
        # The run pointer may progress partial -> completed; segment results never change.
        durable_json(self.run_dir / "result_manifest.json", result)
        append_json(self.run_dir / "status.jsonl",
                    {"state": status, "segment": self.segment.name,
                     "reason": str(reason), "time": time.time()})
        return result


def _safe_member(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or
            any(ord(char) < 32 for char in name)):
        raise ValueError("unsafe archive member")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError("unsafe archive member")
    # These names alias other paths/devices on Windows despite being valid
    # POSIX filenames. Packages are transferred between both environments.
    for part in path.parts:
        if part.endswith((".", " ")) or part.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }:
            raise ValueError("nonportable archive member")
    return path


def export_archive(run_dir, archive_path, *, max_uncompressed_bytes=256 * 1024 * 1024):
    run_dir = no_symlinks(run_dir).resolve()
    archive_path = no_symlinks(archive_path).resolve()
    if archive_path.is_relative_to(run_dir):
        raise ValueError("archive cannot be a member of itself")
    files = {}
    total_bytes = 0
    for path in sorted(run_dir.rglob("*")):
        no_symlinks(path)
        if path.is_file():
            name = path.relative_to(run_dir).as_posix()
            _safe_member(name)
            if name == "SHA256SUMS":
                raise ValueError("reserved checksum name")
            total_bytes += path.stat().st_size
            if total_bytes > max_uncompressed_bytes:
                raise ValueError("archive exceeds uncompressed size budget")
            files[name] = path.read_bytes()
    if "planned_manifest.json" not in files:
        raise ValueError("run has no planned manifest")
    checksums = "".join(f"{sha256_bytes(data)}  {name}\n" for name, data in files.items())
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("SHA256SUMS", checksums)
    return {"path": str(archive_path), "sha256": file_sha256(archive_path),
            "member_count": len(files), "uncompressed_bytes": sum(map(len, files.values()))}


def import_archive(archive_path, destination, *, max_uncompressed_bytes=256 * 1024 * 1024):
    """Validate every byte/member before creating a new extraction directory."""
    destination = no_symlinks(destination)
    if destination.exists():
        raise FileExistsError("import requires a new clean directory")
    with zipfile.ZipFile(no_symlinks(archive_path)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise ValueError("duplicate or case-colliding archive members")
        for entry in entries:
            _safe_member(entry.filename)
            mode = entry.external_attr >> 16
            if entry.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                raise ValueError("only regular archive files are allowed")
        lowered = {name.casefold() for name in names}
        for name in names:
            if any(parent.as_posix().casefold() in lowered
                   for parent in PurePosixPath(name).parents if parent.as_posix() != "."):
                raise ValueError("archive file/directory collision")
        if sum(entry.file_size for entry in entries) > max_uncompressed_bytes:
            raise ValueError("archive exceeds uncompressed size budget")
        if "SHA256SUMS" not in names:
            raise ValueError("archive has no checksum inventory")
        expected = {}
        for line in archive.read("SHA256SUMS").decode("utf-8").splitlines():
            digest, name = line.split("  ", 1)
            _safe_member(name)
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in expected or name == "SHA256SUMS":
                raise ValueError("invalid checksum inventory")
            expected[name] = digest
        if set(expected) != set(names) - {"SHA256SUMS"}:
            raise ValueError("extra or missing archive files")
        content = {name: archive.read(name) for name in expected}
        if any(sha256_bytes(data) != expected[name] for name, data in content.items()):
            raise ValueError("archive checksum mismatch")
        planned = json.loads(content["planned_manifest.json"])
        if planned.get("schema") != SCHEMA:
            raise ValueError("unsupported archive manifest schema")
        if "result_manifest.json" in content:
            result = json.loads(content["result_manifest.json"])
            if result.get("planned_manifest_sha256") != expected["planned_manifest.json"]:
                raise ValueError("result/planned manifest binding mismatch")
        destination.mkdir(parents=True, exist_ok=False)
        for name, data in content.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
        with (destination / "SHA256SUMS").open("xb") as handle:
            handle.write(archive.read("SHA256SUMS"))
    return destination


def rebuild_result(run_dir):
    """Rebuild a diagnostic row entirely from archived event/metric records."""
    run_dir = no_symlinks(run_dir)
    events, metrics, ids = [], [], set()
    for path in sorted((run_dir / "segments").glob("*/events.jsonl")):
        for event in read_journal(path, allow_truncated=not (path.parent / "worker_result.json").exists()):
            if event["event_id"] in ids:
                raise ValueError("duplicate event in archive")
            ids.add(event["event_id"])
            events.append(event)
    for path in sorted((run_dir / "segments").glob("*/metrics.jsonl")):
        metrics.extend(read_journal(path, allow_truncated=not (path.parent / "worker_result.json").exists()))
    return {"event_count": len(events), "metric_count": len(metrics),
            "toy_settled_value": sum(e.get("settled_value", 0) for e in events)}


def _toy_worker(args):
    """Real-process fault fixture. No scheduling or scientific claims."""
    repo = Path(args.repo).resolve()
    config = {"steps": args.steps, "fixture": "generic_counter"}
    inputs = {"input": Path(args.input).resolve()}
    sources = [Path(__file__).resolve()]
    if args.resume:
        context = RunContext.open(args.resume, repo, config, inputs, sources)
        checkpoint = context.latest_checkpoint()
        state = context.load_checkpoint(checkpoint)
        context.new_segment(checkpoint)
    else:
        context = RunContext.create(repo, args.output_root, args.run_id, config, inputs, sources)
        context.new_segment()
        state = {"next": 0, "settled": [], "sum": 0}
        context.checkpoint(state)
    stop = False

    def stop_handler(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        while state["next"] < args.steps:
            if stop or context.stop_requested():
                context.checkpoint(state)
                context.finish("partial", "controlled_stop")
                return 0
            index = state["next"]
            context.append_event({"event_id": f"settlement-{index}", "settled_value": index + 1})
            state["settled"].append(index)
            state["sum"] += index + 1
            state["next"] += 1
            context.checkpoint(state)
            context.append_metric({"step": state["next"], "sum": state["sum"]})
            context.heartbeat()
            if args.crash_at and state["next"] == args.crash_at:
                raise RuntimeError("injected worker exception")
            if args.stall_at and state["next"] == args.stall_at:
                while True:
                    time.sleep(0.05)
            if args.delay:
                time.sleep(args.delay)
        context.finish("completed", "counter_complete", summary=state)
        return 0
    except Exception as error:
        context.finish("failed", repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    toy = subparsers.add_parser("toy-worker")
    toy.add_argument("--repo", required=True)
    toy.add_argument("--output-root", required=True)
    toy.add_argument("--run-id", required=True)
    toy.add_argument("--input", required=True)
    toy.add_argument("--steps", type=int, default=10)
    toy.add_argument("--delay", type=float, default=0)
    toy.add_argument("--crash-at", type=int)
    toy.add_argument("--stall-at", type=int)
    toy.add_argument("--resume")
    rebuild = subparsers.add_parser("rebuild")
    rebuild.add_argument("run_dir")
    args = parser.parse_args()
    if args.command == "rebuild":
        print(json.dumps(rebuild_result(args.run_dir), sort_keys=True))
        return 0
    return _toy_worker(args)


if __name__ == "__main__":
    sys.exit(main())
