"""Post-hoc 2x2 service/startup diagnostic: 18 new CPU rollouts, no fitting."""
import argparse
import copy
from datetime import datetime, timezone
import itertools
from pathlib import Path
import statistics

from .calibrated_audit import audit
from .n5_acceptance import BASE, DEFAULT, FREL
from .n5_analysis import compare_prediction, reduce_group, route, table
from .reselect import ROOT, digest, execute_cell, lines, read, save

DOC = ROOT / 'docs/maxopt_n9_paper_20260920'
CELLS = ('original', 'service_only', 'startup_only', 'calibrated')
REGIMES = ('steady', 'recovery', 'burst_offline')


def factor_models(models):
    original, calibrated = models['original'], models['calibrated']
    assert original['setup_seconds'] == calibrated['setup_seconds']
    assert original['shutdown_seconds'] == calibrated['shutdown_seconds']
    result = {k: copy.deepcopy(v) for k, v in models.items()}
    result['service_only'] = copy.deepcopy(calibrated)
    result['service_only']['startup_seconds'] = original['startup_seconds']
    result['startup_only'] = copy.deepcopy(original)
    result['startup_only']['startup_seconds'] = calibrated['startup_seconds']
    return result


def contrasts(values):
    oo, co, oc, cc = [values[k] for k in CELLS]
    return dict(service_at_30=co-oo, service_at_69=cc-oc,
                startup_at_original_service=oc-oo, startup_at_calibrated_service=cc-co,
                interaction=cc-co-oc+oo)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = ROOT / FREL
    prediction = DEFAULT / 'payload/n5_prediction_lock_20260919_r1'
    raw_path = BASE / 'n5_cpu_acceptance_20260920_r1/raw_recomputed.json'
    lock_path = prediction / 'prediction_lock.json'
    assert digest(lock_path) == '79557ebbeda32ff04cbca4baeb95590feb1684a3118d897d9ea6e38759dcf942'
    locked_files = read(lock_path)['files']
    for name, expected in locked_files.items():
        assert digest(prediction / name) == expected, name
    sources = [Path(__file__), ROOT / 'scripts/maxopt_v2/calibrated.py',
               ROOT / 'scripts/maxopt_v2/calibrated_audit.py', ROOT / 'scripts/maxopt_v2/engine.py',
               ROOT / 'scripts/maxopt_v2/n5_analysis.py', DOC / 'R2_POSTHOC_DIAGNOSTIC_PLAN_zh.md',
               frozen / 'models.json', frozen / 'inputs/contract.json', frozen / 'n5_plan.json', raw_path]
    bindings = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    models = factor_models(read(frozen / 'models.json'))
    plan, contract, raw = read(frozen / 'n5_plan.json'), read(frozen / 'inputs/contract.json'), read(raw_path)
    save(output / 'design.json', dict(scope='post-hoc diagnostic; no test refit or policy selection',
         started_utc=datetime.now(timezone.utc).isoformat(), new_rollouts=18,
         models=models, bindings=bindings, prediction_lock_sha256=digest(lock_path)))
    results, comparisons, common_rows, audit_records, group_rows = {}, [], [], [], []
    for regime, spec in itertools.product(REGIMES, plan['policies']):
        window, policy = 'locked_test_' + regime + '_v2', spec['id']
        rows = lines(prediction / (window + '.jsonl'))
        predictions = {}
        for cell in CELLS:
            existing = cell in ('original', 'calibrated')
            if existing:
                folder = prediction / 'predictions' / cell / policy / window
            else:
                path = f'predictions/{cell}/{policy}/{window}'
                execute_cell(output, path, rows, contract, models[cell], spec['policy'], allow_locked_test=True)
                folder = output / path
            result, events = read(folder / 'result.json'), lines(folder / 'events.jsonl')
            check = audit(rows, events, result['summary'], contract, models[cell], spec['policy'], result['requests'])
            audit_records.append(dict(window=window, policy=policy, cell=cell, new=not existing, **check))
            predictions[cell] = result, events
            if policy in ('all1', 'all2') and cell in ('startup_only', 'calibrated'):
                peer = 'original' if cell == 'startup_only' else 'service_only'
                assert predictions[peer] == predictions[cell], 'fixed-policy startup invariance'
            for repeat in (1, 2, 3):
                rid = policy + '__' + window + '__r' + str(repeat)
                metrics, confusion = compare_prediction(raw[rid], result, events)
                comparisons.append(dict(run_id=rid, window=window, policy=policy, repeat=repeat,
                    cell=cell, origin='prelocked' if existing else 'posthoc',
                    **metrics, route_confusion=str(confusion)))
        for repeat in (1, 2, 3):
            rid = policy + '__' + window + '__r' + str(repeat)
            actual = raw[rid]['requests']
            common = [k for k, r in actual.items() if route(r['route']) == 'local' and r['status'] == 'completed'
                      and all(route(predictions[c][0]['requests'][k]['route']) == 'local'
                              and predictions[c][0]['requests'][k]['completed_at'] is not None for c in CELLS)]
            for cell in CELLS:
                pred = predictions[cell][0]['requests']
                errors = [abs((actual[k]['completed_at']-actual[k]['dispatched_at']) -
                              (pred[k]['completed_at']-pred[k]['dispatched_at'])) for k in common]
                common_rows.append(dict(run_id=rid, window=window, policy=policy, repeat=repeat, cell=cell,
                    common_local_count=len(common), all_arrivals=len(actual),
                    common_service_mae=statistics.mean(errors) if errors else None))
        metrics = ('gpu_occupied_seconds_absolute_error', 'gpu_active_seconds_absolute_error',
                   'shared_local_service_mae_seconds', 'shared_local_coverage_all_arrivals', 'target_disagreement_rate')
        for cell in CELLS:
            subset = [r for r in comparisons if (r['window'], r['policy'], r['cell']) == (window, policy, cell)]
            s = predictions[cell][0]['summary']
            group_rows.append(dict(window=window, policy=policy, cell=cell,
                origin='prelocked' if cell in ('original', 'calibrated') else 'posthoc',
                occupied_predicted=s['gpu_occupied_seconds'], active_predicted=s['gpu_active_seconds'],
                starts_predicted=s['startup_events'], **{k: statistics.mean(r[k] for r in subset) for k in metrics}))
        print('completed', window, policy, flush=True)
    effects = []
    for window, policy, metric in itertools.product(sorted({r['window'] for r in group_rows}),
            [s['id'] for s in plan['policies']], ('occupied_predicted', 'gpu_occupied_seconds_absolute_error',
                                               'active_predicted', 'gpu_active_seconds_absolute_error')):
        values = {r['cell']: r[metric] for r in group_rows if r['window'] == window and r['policy'] == policy}
        effects.append(dict(window=window, policy=policy, metric=metric, **contrasts(values)))
    assert len(comparisons) == 108 and sum(r['origin'] == 'posthoc' for r in comparisons) == 54
    assert sum(r['new'] for r in audit_records) == 18
    for name, rows in [('comparisons', comparisons), ('groups', group_rows), ('effects', effects), ('common_support', common_rows)]:
        table(output / (name + '.csv'), rows)
    save(output / 'event_audits.json', audit_records)
    # Recheck immutable inputs after execution, including every original prediction file.
    for name, expected in bindings.items():
        assert digest(ROOT / name) == expected
    for name, expected in locked_files.items():
        assert digest(prediction / name) == expected
    result = dict(status='PASS', new_cpu_rollouts=18, reused_prelocked_rollouts=18,
                  new_comparisons=54, total_comparisons=108, audited_cells=36,
                  fixed_policy_startup_invariance_checks=12, frozen_inputs_unchanged=True,
                  no_gpu=True, no_policy_reselection=True, no_refitting=True,
                  scope='post-hoc computational interventions; not physical causal identification')
    save(output / 'result.json', result)
    save(output / 'manifest.json', dict(files={str(p.relative_to(output)): digest(p)
         for p in sorted(output.rglob('*')) if p.is_file()}, inputs=bindings))
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
