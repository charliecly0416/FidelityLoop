"""Paths, immutable outputs, and frozen source protection for CPU stages."""
from pathlib import Path
import subprocess

from fidelityloop.legacy.maxopt_v2.reselect import digest, read, lines, save, save_lines

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/max_optimization_v3_20260920'
OUT = ROOT / 'artifacts/max_optimization_v3_20260921'
OLD = ROOT / 'artifacts/max_optimization_v2_20260916'
DEV = OLD / 'n4b_cpu_reselection_20260919_r2'
PAYLOAD = OLD / 'n5_gpu_return_20260920_r1/payload'
RAW = OLD / 'n5_cpu_acceptance_20260920_r1/raw_recomputed.json'
CAMPAIGN = PAYLOAD / 'n5-frozen-heldout-campaign-20260919-r1'
PRED = PAYLOAD / 'n5_prediction_lock_20260919_r1'
TAG = 'maxopt-v2-paper-freeze-20260920'
H_ID = 'hysteresis_u8_d0_c60'
H = dict(name='hysteresis', up_threshold=8, down_threshold=0, cooldown_seconds=60, min_devices=0)


def verify_protection():
    subprocess.run(['git', 'diff', '--exit-code', TAG, '--'], cwd=ROOT, check=True, capture_output=True)
    manifest = read(OUT / 's0/protection.json')
    for name, expected in manifest['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    return len(manifest['files'])


def initialize():
    subprocess.run(['git', 'diff', '--exit-code', TAG, '--'], cwd=ROOT, check=True, capture_output=True)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    # Protect all tracked V1/V2 files plus untracked/ignored evidence actually reused.
    paths = {ROOT / name for name in tracked if name and (ROOT / name).is_file()}
    paths.update(p for base in (DEV, PRED, CAMPAIGN, PAYLOAD / 'scripts') for p in base.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths.update([RAW, DOC / 'V3_PREREGISTRATION.md', DOC / 'CLAUDE_V3_FINAL_REVIEW_20260920.md'])
    save(OUT / 's0/protection.json', dict(tag=TAG, files={str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}))
    save(OUT / 's0/registration.json', dict(schema='maxopt-v3-registration-v1',
         preregistration_sha256=digest(DOC / 'V3_PREREGISTRATION.md'),
         claude_sha256=digest(DOC / 'CLAUDE_V3_FINAL_REVIEW_20260920.md'),
         cpu_authorized=True, gpu_authorized=False, old_test_role='V3 development only',
         loss_weights=[.5, .25, .25], queue_definition='arrival timestamp to dispatch/cutoff; includes tick admission wait',
         model_grid=dict(startup=[.9, 1., 1.1], shutdown=[.8, 1., 1.2]),
         budgets=dict(s2_rollouts=500, s3_rollouts=1500, s4_rollouts=75, wall_hours=8),
         selection_salt='maxopt-v3-window-v1\n', thinning_salt='maxopt-v3-thin-v1'))
    print('S0 protection PASS:', verify_protection(), 'files')


if __name__ == '__main__':
    initialize()
