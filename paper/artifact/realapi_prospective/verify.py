"""Verify sealed portable evidence and recompute the prospective paper results."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def recompute(root):
    sys.path.insert(0, str(root))
    from toolkit.audit import evaluate, choose

    read = lambda p: json.loads(p.read_text())
    cells = []
    for path in sorted((root / 'records').glob('*.json')):
        item = read(path)
        record = item['record']
        cells.append((record, evaluate(record, physical=True)))
    design = read(root / 'design.json')
    candidates = design['candidate_policy_ids']
    windows = design['window_ids']
    expected = {(p, w, r) for p in candidates + ['all2'] for w in windows for r in (1, 2)}
    # Run IDs bind the repeat; the original verifier validates input and receipt binding.
    found = {(r['policy_id'], r['window_id'], int(r['run_id'].rsplit('__r', 1)[1].split('__',1)[0])) for r, _ in cells}
    if len(cells) != 36 or found != expected:
        raise ValueError('physical matrix differs from the registered 36 cells')
    costs = {}
    robust = {}
    for policy in candidates + ['all2']:
        group = [m for r, m in cells if r['policy_id'] == policy]
        costs[policy] = sum(m['system_scenario_cost'] for m in group) / len(group)
        robust[policy] = all(m['P1plus'] for m in group)
    viable = {p: costs[p] for p in candidates if robust[p]}
    best = min(viable.values())
    comparisons = {}
    for name in ('Local_real', 'Hist69_real', 'Fixed30_real', 'Local_timer5'):
        predictions = [read(p) for p in sorted((root / 'predictions' / name).glob('*.json'))]
        indexed = {(p['policy_id'], p['window_id']): evaluate(p, physical=False) for p in predictions}
        pairs = [(indexed[(r['policy_id'], r['window_id'])], m) for r, m in cells if r['policy_id'] in candidates]
        selection = choose(predictions, candidates, windows)
        comparisons[name] = dict(
            agreement=sum(p['P1plus'] == m['P1plus'] for p, m in pairs),
            false_accepts=sum(p['P1plus'] and not m['P1plus'] for p, m in pairs),
            false_rejects=sum(not p['P1plus'] and m['P1plus'] for p, m in pairs),
            cost_mae=sum(abs(p['system_scenario_cost'] - m['system_scenario_cost']) for p, m in pairs)/len(pairs),
            selected=selection['selected'],
            selected_physical_cost={p: costs[p] for p in selection['selected']},
            selected_premium_pct={p: 100*(costs[p]-best)/best if robust[p] else None for p in selection['selected']})
    return dict(status='PASS', physical_cells=len(cells), feasible_cells=sum(m['P1plus'] for _,m in cells),
        candidate_cells=32, consistently_feasible_candidates=viable, mean_system_costs=costs,
        successful_http_responses=sum(m['api_completed'] for _,m in cells), comparisons=comparisons)


def main():
    with zipfile.ZipFile(ROOT/'evidence.zip') as archive, tempfile.TemporaryDirectory(prefix='fidelityloop-evidence-') as temp:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('unsafe archive path')
        if archive.testzip():
            raise ValueError('archive CRC mismatch')
        archive.extractall(temp)
        subprocess.run([sys.executable, '-B', str(Path(temp)/'CPU_CHECK.py')], check=True)
        print(json.dumps(recompute(Path(temp)), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
