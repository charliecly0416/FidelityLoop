"""Freeze accepted development inputs; bounded CPU search and N5 registration.

Usage: python -m fidelityloop.legacy.maxopt_v2.reselect {freeze,search,verify} ...
No GPU imports, subprocess inference, network calls, or held-out reads.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .calibrated import CalibratedSimulator, quantile
from .calibrated_audit import audit

ROOT = Path(__file__).resolve().parents[2]
DOC = Path('docs/max_optimization_v2_20260915')
REGIMES = ('steady', 'recovery', 'burst_offline')
CONTRACT_SHA = '1862df57e7109dc0242dccab0b7dc6531f319258605710a8ee4573ef2c97cfa3'
REGISTRATION_SHA = 'e36e05c8a696c8e69ca55e2c56fc3c51a2dcfd526bbaee6239a2d9da081b21a7'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def save_lines(path, objects):
    with Path(path).open('x', encoding='utf-8') as f:
        for obj in objects:
            f.write(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def configurations():
    result = [dict(id=n, policy=dict(name=n)) for n in ('all1', 'all2')]
    for p, c in itertools.product((2,4,8), (0,15,30,60)):
        result.append(dict(id=f'hpa_p{p}_c{c}', policy=dict(name='hpa', target_pressure=p, cooldown_seconds=c, min_devices=0)))
    for u, d, c in itertools.product((2,4,8), (0,1), (15,60)):
        result.append(dict(id=f'hysteresis_u{u}_d{d}_c{c}', policy=dict(name='hysteresis', up_threshold=u,
                            down_threshold=d, cooldown_seconds=c, min_devices=0)))
    return result


def rank(run_rows, specs):
    ranked = []
    for c in specs:
        cells = [r for r in run_rows if r['configuration'] == c['id']]
        if len(cells) != 3 or len({r['window'] for r in cells}) != 3:
            raise ValueError('incomplete/duplicate selection matrix')
        ranked.append(dict(id=c['id'], policy=c['policy'], feasible=all(r['summary']['feasible'] for r in cells),
                           mean_scenario_cost=statistics.mean(r['summary']['costs']['total'] for r in cells),
                           failed_windows=[r['window'] for r in cells if not r['summary']['feasible']]))
    ranked.sort(key=lambda c: (not c['feasible'], c['mean_scenario_cost'], c['id']))
    feasible = [r for r in ranked if r['feasible'] and r['policy']['name'] not in ('all1','all2')]
    return dict(selected_dynamic=feasible[0] if feasible else None,
                feasible_dynamic_count=len(feasible), ranking=ranked)


def freeze(accepted, output):
    accepted, output = Path(accepted), Path(output)
    revision = accepted/'stage2-development-revision-20260919-r1'
    contract_path = ROOT/DOC/'N1_FEASIBILITY_CONTRACT.json'
    if digest(contract_path) != CONTRACT_SHA or digest(revision/'registration.json') != REGISTRATION_SHA:
        raise ValueError('accepted contract/revision changed')
    manifest = read(revision/'manifest.json')
    files = {f['path']: f for f in manifest['files']}
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def copy(source, name, expected=None):
        source, destination = Path(source), output/name
        actual = digest(source)
        if expected is not None and actual != expected:
            raise ValueError('accepted input mismatch: '+str(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination) != actual:
            raise ValueError('copy mismatch')
        bindings[name] = dict(sha256=actual, bytes=destination.stat().st_size)

    copy(contract_path, 'inputs/contract.json', CONTRACT_SHA)
    copy(revision/'registration.json', 'inputs/revision_registration.json', REGISTRATION_SHA)
    copy(revision/'manifest.json', 'inputs/revision_manifest.json')
    copy(accepted/'artifacts/maxopt_stage2_formal_revision_state.json', 'inputs/revision_state.json')
    copy(accepted/'n4c-profile-return-20260918-r1/service_model.json', 'inputs/n4c_service_model.json')
    for split, regime in itertools.product(('train','validation'), REGIMES):
        name = f'C2/{split}_{regime}_v2.jsonl'
        copy(revision/name, 'inputs/'+name, files[name]['sha256'])
    # Keep exact RAW sources, not merely a fitted scalar or temporary /tmp path.
    startup_samples, shutdown_samples, setup_samples = [], [], []
    for regime, repeat in itertools.product(REGIMES, (1,2,3)):
        run_id = f'C2__train_{regime}_v2__main__r{repeat}'
        run = accepted/'stage2-development-campaign-20260919-r1'/run_id
        for name in ('controller_events.jsonl', 'request_events.jsonl'):
            copy(run/name, f'inputs/train_raw/{run_id}/{name}')
        launches, terminals, ready = {}, set(), {}
        for e in lines(run/'controller_events.jsonl'):
            if e['kind'] == 'worker_launch':
                launches[e['gpu']] = e['controller_monotonic_ns']
            if e['kind'] == 'worker_event':
                w, gpu = e['event'], e['gpu']
                if w['kind'] == 'finished' and w.get('request_id') == '__setup_gpu'+str(gpu):
                    terminals.add(gpu)
                if w['kind'] == 'status' and w.get('active_count') == 0 and gpu in terminals and gpu not in ready:
                    ready[gpu] = e['controller_monotonic_ns']
            if e['kind'] == 'local_ready_barrier':
                setup_samples.append(dict(run=run_id, seconds=e['setup_seconds']))
            if e['kind'] == 'cleanup_complete':
                shutdown_samples.append(dict(run=run_id, seconds=e['elapsed_seconds']))
        if set(ready) != {0,1}:
            raise ValueError('missing empty-ready evidence')
        startup_samples.extend(dict(run=run_id, gpu=g, seconds=(ready[g]-launches[g])/1e9) for g in (0,1))
    if (len(startup_samples), len(shutdown_samples), len(setup_samples)) != (18,9,9):
        raise ValueError('incomplete lifecycle sample set')
    lifecycle = {}
    for name, samples in (('startup',startup_samples), ('shutdown',shutdown_samples), ('setup',setup_samples)):
        values = [r['seconds'] for r in samples]
        lifecycle[name] = dict(samples=samples, median=statistics.median(values), p90=quantile(values,.9), maximum=max(values))
    save(output/'lifecycle_estimates.json', lifecycle)
    lookup = read(output/'inputs/n4c_service_model.json')['lookup_wall_seconds']
    common = dict(shutdown_seconds=lifecycle['shutdown']['median'], setup_seconds=lifecycle['setup']['median'])
    models = dict(calibrated=dict(kind='n4c_progress_lookup', lookup_wall_seconds=lookup,
                                 startup_seconds=lifecycle['startup']['median'], **common),
                  original=dict(kind='original_equal_share', startup_seconds=30.0, **common))
    save(output/'models.json', models)
    # Save the complete import closure and new-stage code for portable replay.
    source_names = ('__init__.py','acquire.py','workload.py','engine.py','calibrated.py',
                    'calibrated_audit.py','reselect.py','prepare_n5.py')
    for name in source_names:
        copy(ROOT/'scripts/maxopt_v2'/name, 'source/scripts/maxopt_v2/'+name)
    for name in ('N4B_CPU_RESELECTION_PLAN_20260919.md','STAGE2_FORMAL_RETURN_REVIEW_20260919.md'):
        copy(ROOT/DOC/name, 'source/'+str(DOC/name))
    for name in ('lifecycle_estimates.json','models.json'):
        bindings[name] = dict(sha256=digest(output/name), bytes=(output/name).stat().st_size)
    registry = dict(schema='maxopt-n4b-reselection-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    candidate='C2', capacity_revisions_used=1, capacity_revisions_limit=1,
                    initial_state='two_ready_empty_queues', decision_interval_seconds=1,
                    configurations=configurations(), windows=[f'validation_{r}_v2' for r in REGIMES],
                    models=list(models), runs_per_model=78, policy_ticks_per_model=163800,
                    train_diagnostic_runs_per_model=3,
                    selection='calibrated model; all three windows feasible; mean window scenario cost; lexical ID tie',
                    original_selection='reported separately; cannot replace calibrated winner',
                    locked_test_performance_opened=False, inputs_and_source=bindings,
                    uncertainty='mixed batch approximation; paired-ready startup and joint cleanup are lifecycle proxies; no dynamic hardware evidence yet')
    save(output/'registration.json', registry)
    return dict(status='FROZEN', directory=str(output), registration_sha256=digest(output/'registration.json'),
                startup_seconds=models['calibrated']['startup_seconds'], shutdown_seconds=common['shutdown_seconds'])


def check_bindings(output, current_code=True):
    output = Path(output)
    registry = read(output/'registration.json')
    for name, expected in registry['inputs_and_source'].items():
        if digest(output/name) != expected['sha256']:
            raise ValueError('frozen input/source changed: '+name)
        if current_code and name.startswith('source/scripts/') and digest(ROOT/name.removeprefix('source/')) != expected['sha256']:
            raise ValueError('running code differs from registration: '+name)
    if registry['configurations'] != configurations() or registry['runs_per_model'] != 78:
        raise ValueError('registered search space changed')
    return registry


def execute_cell(output, relative, rows, contract, model, policy, allow_locked_test=False):
    folder = output/relative
    folder.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    result = CalibratedSimulator(rows, contract, model, policy, allow_locked_test=allow_locked_test).run()
    verification = audit(rows, result['events'], result['summary'], contract, model, policy, result['requests'])
    save_lines(folder/'events.jsonl', result.pop('events'))
    save(folder/'result.json', result)
    save(folder/'verification.json', verification)
    return dict(path=relative, summary=result['summary'], verification=verification,
                seconds=time.perf_counter()-begin,
                files={n:digest(folder/n) for n in ('events.jsonl','result.json','verification.json')})


def train_comparison(output, predictions):
    report = []
    for model, regime, repeat in itertools.product(predictions, REGIMES, (1,2,3)):
        run = f'C2__train_{regime}_v2__main__r{repeat}'
        events = lines(output/f'inputs/train_raw/{run}/request_events.jsonl')
        dispatch = {e['request_id']:e for e in events if e['kind'] == 'dispatch'}
        terminals = {e['request_id']:e for e in events if e['kind'] == 'terminal' and e['status'] == 'completed' and not e['after_observation']}
        pred = predictions[model][regime]
        errors, relative_errors, route_equal = [], [], 0
        for rid, actual in terminals.items():
            p = pred['requests'][rid]
            route_equal += p['route'] == ('synthetic_api' if actual['route']=='synthetic_api' else int(actual['route'][-1]))
            if actual['route'].startswith('gpu') and isinstance(p['route'],int) and p['completed_at'] is not None:
                observed = actual['at_s']-dispatch[rid]['at_s']
                predicted = p['completed_at']-p['dispatched_at']
                errors.append(abs(observed-predicted))
                relative_errors.append(abs(observed-predicted)/observed)
        report.append(dict(model=model, run=run, shared_local_pairs=len(errors),
                           local_service_mae_seconds=statistics.mean(errors) if errors else None,
                           local_service_relative_error_median=quantile(relative_errors,.5),
                           local_service_relative_error_p90=quantile(relative_errors,.9),
                           route_agreement_rate=route_equal/len(terminals),
                           actual_api_accepted=sum(e['route']=='synthetic_api' for e in dispatch.values()),
                           predicted_api_accepted=pred['summary']['api_accepted'],
                           actual_completed=len(terminals), predicted_completed=pred['summary']['completed']))
    return dict(scope='train-only closed-loop development diagnostic; shared-local pairs exclude routing disagreements, which are reported separately',
                rows=report)


def search(output):
    output = Path(output)
    registry = check_bindings(output)
    contract, models = read(output/'inputs/contract.json'), read(output/'models.json')
    begin = time.perf_counter()
    cpu_begin = time.process_time()
    train_predictions = {}
    for name, model in models.items():
        train_predictions[name] = {}
        for regime in REGIMES:
            rows = lines(output/f'inputs/C2/train_{regime}_v2.jsonl')
            cell = execute_cell(output, f'train_diagnostics/{name}/{regime}', rows, contract, model, {'name':'all2'})
            train_predictions[name][regime] = read(output/cell['path']/'result.json')
    save(output/'train_comparison.json', train_comparison(output, train_predictions))
    data = {w: lines(output/f'inputs/C2/{w}.jsonl') for w in registry['windows']}
    cells = []
    for name, model in models.items():
        for spec, window in itertools.product(registry['configurations'], registry['windows']):
            cell = execute_cell(output, f'search/{name}/{spec["id"]}/{window}', data[window], contract, model, spec['policy'])
            cell.update(model=name, configuration=spec['id'], window=window)
            cells.append(cell)
        print(json.dumps(dict(model=name, completed=sum(c['model']==name for c in cells))), flush=True)
    selection = {name:rank([c for c in cells if c['model']==name], registry['configurations']) for name in models}
    save(output/'search_results.json', dict(registration_sha256=digest(output/'registration.json'),
                                           cells=cells, selections=selection, wall_seconds=time.perf_counter()-begin,
                                           cpu_seconds=time.process_time()-cpu_begin,
                                           cpu_policy_ticks=2*163800, search_run_count=len(cells), train_diagnostic_runs=6))
    with (output/'search_results.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model','configuration','window','feasible','online_arrivals','online_on_time','offline_on_time',
                         'gpu_seconds','api_accepted','startups','shutdowns','operating_cost','miss_penalty','total_cost'])
        for c in cells:
            s=c['summary']; p=s['populations']
            writer.writerow([c['model'],c['configuration'],c['window'],s['feasible'],p['online']['arrivals'],
                             p['online']['on_time'],p['offline']['on_time'],s['gpu_occupied_seconds'],s['api_accepted'],
                             s['startup_events'],s['shutdown_events'],s['costs']['operating'],s['costs']['offline_miss'],s['costs']['total']])
    policies = configurations()[:2]
    selected = selection['calibrated']['selected_dynamic']
    if selected:
        policies.append(dict(id=selected['id'], policy=selected['policy']))
    # Cyclic policy ordering balances early/middle/late slots across repeats.
    matrix = []
    for repeat in (1,2,3):
        for regime in REGIMES:
            shift = (repeat-1) % len(policies)
            for policy in policies[shift:]+policies[:shift]:
                matrix.append(dict(run_id=f'{policy["id"]}__locked_test_{regime}_v2__r{repeat}',
                                   policy=policy['id'], window=f'locked_test_{regime}_v2', repeat=repeat))
    save(output/'n5_plan.json', dict(schema='maxopt-n5-frozen-matrix-v1', selection_sha256=digest(output/'search_results.json'),
                                    registration_sha256=digest(output/'registration.json'), candidate='C2',
                                    policies=policies, matrix=matrix, runs=len(matrix), seconds_per_run=2100,
                                    observation_wall_hours=len(matrix)*2100/3600,
                                    reserved_two_gpu_hours=len(matrix)*2100/1800,
                                    locked_test_generated=False, predictions_locked=False,
                                    ready_to_execute=False, remaining_gate='bounded real lifecycle actuator verification, then one held-out derivation and pre-observation predictions',
                                    workload_revision_registration_sha256=REGISTRATION_SHA))
    return dict(status='CPU_SEARCH_COMPLETE', selections={n:s['selected_dynamic'] for n,s in selection.items()}, n5_runs=len(matrix))


def verify(output):
    output = Path(output)
    registry = check_bindings(output)
    results = read(output/'search_results.json')
    if results['registration_sha256'] != digest(output/'registration.json'):
        raise ValueError('search registration binding changed')
    contract, models = read(output/'inputs/contract.json'), read(output/'models.json')
    expected = set(itertools.product(models, (c['id'] for c in registry['configurations']), registry['windows']))
    seen, checks = set(), 0
    data = {w:lines(output/f'inputs/C2/{w}.jsonl') for w in registry['windows']}
    specs = {c['id']:c['policy'] for c in registry['configurations']}
    for cell in results['cells']:
        key = (cell['model'],cell['configuration'],cell['window'])
        if key not in expected or key in seen:
            raise ValueError('unexpected/duplicate cell')
        seen.add(key)
        folder = output/cell['path']
        for name, sha in cell['files'].items():
            if digest(folder/name) != sha:
                raise ValueError('run payload changed')
        recorded = read(folder/'result.json')
        if recorded['summary'] != cell['summary']:
            raise ValueError('summary differs from registered cell')
        v = audit(data[cell['window']], lines(folder/'events.jsonl'), recorded['summary'], contract, models[cell['model']],
                  specs[cell['configuration']], recorded['requests'])
        if v != cell['verification']:
            raise ValueError('independent ledger differs')
        checks += v['checks']
    if seen != expected:
        raise ValueError('incomplete matrix')
    for name in models:
        if rank([c for c in results['cells'] if c['model']==name],registry['configurations']) != results['selections'][name]:
            raise ValueError('selection not reproducible')
    plan = read(output/'n5_plan.json')
    selected = results['selections']['calibrated']['selected_dynamic']
    if plan['selection_sha256'] != digest(output/'search_results.json') or plan['runs'] != (27 if selected else 18):
        raise ValueError('N5 selection/matrix binding')
    return dict(status='PASS', runs=len(seen), independent_ledger_checks=checks,
                registration_sha256=digest(output/'registration.json'),
                results_sha256=digest(output/'search_results.json'),
                n5_plan_sha256=digest(output/'n5_plan.json'), locked_test_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','search','verify'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--accepted', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        if args.accepted is None:
            parser.error('freeze needs --accepted accepted Stage 2 extracted root')
        result = freeze(args.accepted, args.output)
    elif args.command == 'search':
        result = search(args.output)
    else:
        result = verify(args.output)
        save(args.output/'independent_verification.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
