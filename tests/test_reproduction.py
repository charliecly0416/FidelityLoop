"""Exercise the public reproduction entry points without the historical checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_facade_cli_matches_historical_receipt(tmp_path):
    output = tmp_path / "facade"
    subprocess.run(
        [sys.executable, "-B", "-m", "fidelityloop.framework.validate",
         "--output", str(output)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True, capture_output=True, text=True,
    )
    expected = json.loads((ROOT / "evidence/HISTORICAL_FACADE_VALIDATION.json").read_text())
    assert json.loads((output / "VALIDATION.json").read_text()) == expected


def test_e1_artifact_runs_after_relocation(tmp_path):
    relocated = tmp_path / "e1"
    shutil.copytree(ROOT / "artifact/e1_repricing", relocated)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    verification = subprocess.run(
        [sys.executable, "-B", "verify.py"], cwd=relocated, env=env,
        check=True, capture_output=True, text=True,
    )
    result = json.loads(verification.stdout)
    assert result["status"] == "PASS_REDUCED_LEDGER_REPRICING"
    assert result["reference_comparison"]["numeric_values_checked"] == 169
    subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "-v", "test_analysis"],
        cwd=relocated, env=env, check=True, capture_output=True, text=True,
    )
