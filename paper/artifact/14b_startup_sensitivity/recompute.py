"""Retrospective CPU startup sweep for the 14B study; archived evidence remains read-only.

Only startup_seconds varies; every other L14 field, the guard model and guard safety stay fixed.
"""
from pathlib import Path
import os
import argparse, copy, json, sys
from concurrent.futures import ProcessPoolExecutor

_received = os.environ.get('FIDELITYLOOP_E6_RECEIVED')
if not _received:
    raise SystemExit(
        'Set FIDELITYLOOP_E6_RECEIVED to the frozen E6 package root '
        'before running this retrospective verifier.'
    )
RECEIVED = Path(_received).expanduser().resolve()
SOURCE, RAW = RECEIVED/'source_package', RECEIVED/'output'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
import e6_analysis as analysis
import e6_replay as replay
from e6_common import read

def setup():
    profile = read(RAW/'profile/MODEL_LOCK.json'); replay.validate_profile(profile)
    context, arms = replay.context_and_arms(profile)
    windows = read(SOURCE/'protocol/E6_DESIGN.json')['windows']
    inputs = {w: [json.loads(s) for s in (SOURCE/'inputs'/(w+'.jsonl')).read_text().splitlines()] for w in windows}
    return profile, context, arms, windows, inputs

def point(startup):
    profile, context, arms, windows, inputs = setup()
    params = copy.deepcopy(profile['models']['L14']); params['startup_seconds'] = float(startup)
    out = {}
    for w in windows:
        for policy, arm in arms.items():
            if policy == 'all2': continue
            v = replay.parent.predict_one(context, inputs[w], arm, params)
            out[policy+'|'+w] = dict(P1plus=v['request_metrics']['P1plus'],
                occupied=v['result']['summary']['gpu_occupied_seconds'], cost=v['costs']['total'])
    return startup, out

def physical_runs(windows):
    rows = []
    for cell in read(SOURCE/'E6_CELL_PLAN.json')['cells']:
        if cell['policy_id'] == 'all2': continue
        phys, _, _ = analysis.reduce_attempt(RAW/'campaign/cells'/cell['run_id']/'attempt_01')
        rows.append(dict(policy=cell['policy_id'], window=cell['window_id'], run_id=cell['run_id'],
            P1plus=phys['request_metrics']['P1plus'], occupied=phys['gpu_occupied_seconds'], cost=phys['costs']['total']))
    return rows

def self_check(profile, context, arms, windows, inputs):
    """Original S14/L14 predictions must reproduce exactly before any swept value is trusted."""
    originals = replay.verify_predictions(RAW/'predictions', RAW/'profile/MODEL_LOCK.json')
    idx = {(r['prediction_model'], r['policy_id'], r['window_id']): r for r in originals['cells']}
    import gzip
    n = 0
    for m in ('S14', 'L14'):
        for w in windows:
            for policy, arm in arms.items():
                v = replay.parent.predict_one(context, inputs[w], arm, profile['models'][m])
                exp = json.loads(gzip.decompress((RAW/'predictions'/idx[m, policy, w]['path']).read_bytes()))
                assert replay.parent.exact_science(v, exp), (m, policy, w); n += 1
    return n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--points', type=float, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.out.exists(): raise SystemExit('output must be new')
    profile, context, arms, windows, inputs = setup()
    reproduced = self_check(profile, context, arms, windows, inputs)
    phys = physical_runs(windows); assert len(phys) == 8
    with ProcessPoolExecutor(a.workers) as pool: preds = dict(pool.map(point, a.points))
    rows = []
    for s in a.points:
        p = preds[s]; fa = fr = 0; occ = cost = 0.
        for r in phys:
            q = p[r['policy']+'|'+r['window']]
            fa += q['P1plus'] and not r['P1plus']; fr += (not q['P1plus']) and r['P1plus']
            occ += abs(q['occupied']-r['occupied']); cost += abs(q['cost']-r['cost'])
        rows.append(dict(startup_seconds=s, agreement=8-fa-fr, false_accepts=fa, false_rejects=fr,
            occupied_mae=occ/8, cost_mae=cost/8, predicted=p))
        print(f"{s:8.3f}  {8-fa-fr}/8  FA={fa} FR={fr}  occMAE={occ/8:8.2f}  costMAE={cost/8:.4f}  "
              + ' '.join(f"{k}={'Y' if v['P1plus'] else 'n'}" for k, v in sorted(p.items())), flush=True)
    a.out.write_text(json.dumps(dict(status='PASS', retrospective=True, reproduced_original_predictions=reproduced,
        physical=phys, rows=rows), indent=2)+'\n')

if __name__ == '__main__':
    main()
