"""Preregister finite baseline search before any validation outcome is read."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path

from .acquire import ROOT, digest
from .runtime import durable_json


def register(package, contract, output):
    package, contract, output = Path(package), Path(contract), Path(output)
    manifest = json.loads((package / 'manifest.json').read_text())
    configurations = [{'id': name, 'policy': {'name': name}} for name in ('all1', 'all2')]
    for pressure, cooldown in itertools.product((2, 4, 8), (0, 15, 30, 60)):
        configurations.append({'id': f'hpa_p{pressure}_c{cooldown}',
                               'policy': {'name': 'hpa', 'target_pressure': pressure,
                                          'cooldown_seconds': cooldown, 'min_devices': 0}})
    for up, down, cooldown in itertools.product((2, 4, 8), (0, 1), (15, 60)):
        configurations.append({'id': f'hysteresis_u{up}_d{down}_c{cooldown}',
                               'policy': {'name': 'hysteresis', 'up_threshold': up, 'down_threshold': down,
                                          'cooldown_seconds': cooldown, 'min_devices': 0}})
    configuration = json.loads(contract.read_text())
    configuration['model'] = {'identity': 'uncalibrated_equal_share_pilot_v1', 'prefill_tps': 1024.0,
                              'decode_tps': 128.0, 'local_slots': 4, 'startup_seconds': 30, 'max_devices': 2}
    windows = {name[:-6]: meta['sha256'] for name, meta in manifest['files'].items()
               if meta['window']['split'] == 'validation'}
    registry = {'schema': 'maxopt-n3-search-v1', 'state': 'preregistered',
                'package_path': str(package.resolve()), 'package_manifest_sha256': digest(package / 'manifest.json'),
                'contract_path': str(contract.resolve()), 'contract_sha256': digest(contract),
                'configurations': configurations, 'windows': windows, 'engine_config': configuration,
                'code_base_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'implementation_must_be_bound_per_run': True,
                'maximum_runs': len(windows) * len(configurations),
                'maximum_environment_steps': len(windows) * len(configurations) * 2100,
                'primary_fairness_unit': 'same_environment_steps_per_configuration_and_window',
                'model_budget': '12 configurations/family, plus all1/all2; repeat separately at N4b after calibration',
                'selection': 'feasible on every validation window first; then mean scenario cost; tie lexical config ID. If none feasible, report NO_FEASIBLE_POLICY; retain minimum sum constraint excess as diagnostic candidate only.',
                'test_performance_access': 'prohibited',
                'run_limits': {'max_seconds': 1800, 'max_rss_bytes': 4 * 1024**3, 'min_disk_free_bytes': 2 * 1024**3},
                'phase_gate': 'pilot only; N4b must bind accepted hardware calibration model and select again'}
    output.parent.mkdir(parents=True, exist_ok=True)
    durable_json(output, registry, exclusive=True)
    return {'path': str(output), 'sha256': digest(output), 'runs': registry['maximum_runs'],
            'environment_steps': registry['maximum_environment_steps']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(register(args.package, args.contract, args.output), indent=2))
