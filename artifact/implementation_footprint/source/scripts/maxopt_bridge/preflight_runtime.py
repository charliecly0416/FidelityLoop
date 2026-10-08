"""Load the unchanged E1 v3 runtime without an archive or GPU side effects."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parent / "runtime_vendor"
RUNTIME_TEMPLATE = VENDOR / "reference" / "runtime_template.json"


def verify_sources(vendor: Path = VENDOR) -> dict:
    manifest = json.loads((vendor / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"] + manifest.get("references", []):
        path = (vendor / row["path"]).resolve()
        if not path.is_relative_to(vendor.resolve()):
            raise ValueError("source path escapes vendor")
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("runtime source integrity failure: " + row["path"])
    return manifest


def _verify_loaded(row: dict) -> None:
    module = sys.modules.get(row["module"])
    if module is not None:
        path = getattr(module, "__file__", None)
        if not path or hashlib.sha256(Path(path).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("different runtime already loaded: " + row["module"])


def load_runtime():
    """Return original formal_v2.runtime; accept preloaded modules only if byte-identical."""
    manifest = verify_sources()
    for row in manifest["files"]:
        _verify_loaded(row)
    workspace = VENDOR / "workspace"
    parents = {"scripts"}
    for row in manifest["files"]:
        parts = row["module"].split(".")
        parents.update(".".join(parts[:i]) for i in range(1, len(parts)))
    for name in sorted(parents, key=lambda value: (value.count("."), value)):
        module = importlib.import_module(name)
        path = str(workspace.joinpath(*name.split(".")))
        if hasattr(module, "__path__") and path not in module.__path__:
            module.__path__ = [path, *module.__path__]
    for row in manifest["files"]:
        importlib.import_module(row["module"])
        _verify_loaded(row)
    return importlib.import_module("scripts.maxopt_v4.formal_v2.runtime")
