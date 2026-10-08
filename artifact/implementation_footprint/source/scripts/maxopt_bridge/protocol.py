"""Validate the registered F4 protocol and copied input identities.

This module performs no simulation, training, network access or GPU launch.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = Path('docs/maxopt_bridge_execution_20260930')
INPUT = Path('artifacts/maxopt_bridge_20260930/inputs')
F4_IDS = (
    'hysteresis_u8_d0_c60', 'hysteresis_u8_d0_c60__guard',
    'hysteresis_u8_d1_c60', 'hysteresis_u8_d1_c60__guard',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_contract(contract, models, candidates):
    if contract['schema'] != 'maxopt-bridge-contract-v1' or contract['route'] != 'F4':
        raise ValueError('unknown protocol or route; counts alone do not identify a matrix')
    arms = contract['f4_members']
    if tuple(a['id'] for a in arms) != F4_IDS:
        raise ValueError('F4 requires the four exact rule identities in registered order')
    source = {a['id']: a for a in candidates}
    for arm in arms:
        if {k: v for k, v in arm.items() if k != 'guard_model'} != source[arm['id']]:
            raise ValueError('rule configuration changed from historical source')
        if arm['guard_model'] != 'E':
            raise ValueError('guard model must remain identical across prediction arms')
    if set(models['C']) != set(models['C30']):
        raise ValueError('prediction-model fields differ')
    changes = {k for k in models['C'] if models['C'][k] != models['C30'][k]}
    if changes != {'startup_seconds'}:
        raise ValueError('only startup_seconds may differ between C and C30')
    if models['C']['startup_seconds'] != 69.219185665 or models['C30']['startup_seconds'] != 30.0:
        raise ValueError('historical endpoint contrast changed; requires a new contract')
    for model in ('C', 'C30'):
        if any(models[model][k] != models['E'][k] for k in ('kind', 'lookup_wall_seconds')):
            raise ValueError('guard service estimator is not equivalent to the historical candidates')
    if contract['timing'] != dict(window_seconds=1800, drain_seconds=300, decision_seconds=1, cooldown_seconds=60):
        raise ValueError('registered timing changed')
    matrix = contract['matrix']
    if (matrix['windows'], matrix['repeats_per_policy_window'], matrix['comparisons'],
            matrix['capacity_checks'], matrix['total_runs']) != (2, 3, 24, 2, 26):
        raise ValueError('F4 matrix is inconsistent')
    extra = matrix['conditional_with_raw_PPO']
    if (extra['comparisons'], extra['capacity_checks'], extra['total_runs']) != (36, 2, 38):
        raise ValueError('conditional F4 plus raw PPO matrix is inconsistent')
    if matrix['PPO_G_included'] or contract['PPO']['training_guard'] or contract['PPO']['pure_online_training']:
        raise ValueError('unregistered training or guard branch')
    if contract['PPO']['selection_order'] != ['equal_window_mean_total_cost', 'checkpoint_index', 'seed_id']:
        raise ValueError('checkpoint ordering differs from the accepted common contract')
    for key, expected in {
        'selection_evaluator': 'C30 under shared contract, guard off',
        'selection_gate': 'P1plus in every validation window',
        'any_group_without_eligible_checkpoint': 'both PPO physical arms NOT_RUN_TRAINING_GATE; F4 may continue',
    }.items():
        if contract['PPO'][key] != expected:
            raise ValueError('common evaluator, eligibility or symmetric exit changed: ' + key)
    if set(contract['PPO']['pilot_seeds']) & set(contract['PPO']['formal_seeds']):
        raise ValueError('pilot and formal seeds overlap')
    return True


def check(root=ROOT):
    root = Path(root).resolve()
    manifest = read(root / DOC / 'SOURCE_MANIFEST.json')
    for relative, item in manifest['files'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or sha256(path) != item['sha256']:
            raise ValueError('source identity mismatch: ' + relative)
    baseline = read(root / DOC / 'BASELINE_MANIFEST.json')
    for relative, expected in baseline['protected_files'].items():
        if sha256(root / relative) != expected:
            raise ValueError('frozen paper identity changed: ' + relative)
    contract = read(root / DOC / 'CONTRACT.json')
    validate_contract(contract, read(root / INPUT / 'models.json'), read(root / INPUT / 'candidates.json'))
    return dict(status='PASS_PROTOCOL_SELF_CHECK', contract_sha256=sha256(root / DOC / 'CONTRACT.json'),
                sources_checked=len(manifest['files']), protected_files_checked=len(baseline['protected_files']),
                scope='protocol and identities only; no execution or GPU acceptance')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(check(args.root), ensure_ascii=False, indent=2))
