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
