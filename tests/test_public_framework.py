import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_validation_script_passes(tmp_path):
    output = tmp_path / "validation.json"
    result = subprocess.run(
        [sys.executable, "experiments/run_framework_validation.py", "--output", str(output)],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(output.read_text())["status"] == "PASS"
    assert "fidelityloop-cpu-validation-v1" in result.stdout


def test_public_namespace_imports():
    sys.path.insert(0, str(ROOT / "src"))
    from fidelityloop.framework import FrameworkConfig, TargetPolicyAdapter

    assert FrameworkConfig.synthetic(device_count=3).device_count == 3
    assert TargetPolicyAdapter("allN", max_devices=3).decide(
        {"time": 0, "queue_online": [], "queue_offline": [], "local_running": 0,
         "devices": [{"state": "off", "jobs": []} for _ in range(3)]}
    )["target"] == 3


def test_registered_protocol_manifest_passes():
    from fidelityloop.bridge.protocol import check

    assert check()["status"] == "PASS_PROTOCOL_SELF_CHECK"


def _copy_protocol_release(destination):
    import shutil

    source = ROOT / "artifact" / "bridge_execution"
    shutil.copytree(source, destination / "artifact" / "bridge_execution")
    manifest = json.loads((source / "BASELINE_MANIFEST.json").read_text())
    for name in manifest["protected_files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)


def test_protocol_check_uses_the_requested_root(tmp_path):
    from fidelityloop.bridge.protocol import check

    _copy_protocol_release(tmp_path)
    assert check(tmp_path)["status"] == "PASS_PROTOCOL_SELF_CHECK"
    inputs = tmp_path / "artifact" / "bridge_execution" / "inputs" / "models.json"
    inputs.write_text(inputs.read_text() + "\n")
    import pytest

    with pytest.raises(ValueError, match="source identity mismatch"):
        check(tmp_path)


def test_protocol_check_rejects_changed_paper(tmp_path):
    import pytest
    from fidelityloop.bridge.protocol import check

    _copy_protocol_release(tmp_path)
    with (tmp_path / "paper" / "main.pdf").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="frozen paper identity changed"):
        check(tmp_path)
