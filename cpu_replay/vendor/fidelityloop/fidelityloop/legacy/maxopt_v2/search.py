"""Supervised N3 pilot: bounded durable journals, preregistered validation only."""
import argparse
import json
import os
import math
import hashlib
import signal
import sys
import time
from pathlib import Path

from .acquire import ROOT, digest
from .engine import Simulator, TargetPolicy, normalize_config
from .ledger import recompute
from .run import assert_equal_summary, collect_events, rebuild
from .runtime import RunContext, canonical_bytes, canonical_v2_source_files, durable_json
from .supervisor import Limits, supervise
from .workload import load_window, canonical


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in canonical_v2_source_files(ROOT)}


def validate_registry(registry, package):
    if registry.get('schema') != 'maxopt-n3-search-v1' or registry.get('state') != 'preregistered':
        raise ValueError('unregistered search')
    manifest = json.loads((Path(package) / 'manifest.json').read_text())
    windows = {n[:-6]: m['sha256'] for n, m in manifest['files'].items() if m['window']['split'] == 'validation'}
    if registry['windows'] != windows or digest(Path(package) / 'manifest.json') != registry['package_manifest_sha256']:
        raise ValueError('registered windows/package identity mismatch')
    if digest(registry['contract_path']) != registry['contract_sha256']:
        raise ValueError('contract identity mismatch')
    base = json.loads(Path(registry['contract_path']).read_text())
    if {k: v for k, v in registry['engine_config'].items() if k != 'model'} != base:
        raise ValueError('engine config differs from scientific contract')
    candidates = registry['configurations']
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate config ID')
    for family in ('hpa', 'hysteresis'):
        if not 1 <= sum(c['policy']['name'] == family for c in candidates) <= 12:
            raise ValueError('family search budget exceeded')
    steps = registry['engine_config']['window_seconds'] + registry['engine_config']['drain_seconds']
    if registry['maximum_runs'] != len(windows) * len(candidates) or registry['maximum_environment_steps'] != len(windows) * len(candidates) * steps:
        raise ValueError('registered resource budget mismatch')
    return steps


def worker(registry_path, package, output, config_id, window_id, *, profile=False):
    registry = json.loads(Path(registry_path).read_text())
    validate_registry(registry, package)
    if digest(Path(package) / 'manifest.json') != registry['package_manifest_sha256']:
        raise ValueError('registered input changed')
    if digest(registry['contract_path']) != registry['contract_sha256']:
        raise ValueError('registered contract changed')
    split = 'train' if profile else 'validation'
    if not profile and window_id not in registry['windows']:
        raise ValueError('window not preregistered')
    if profile and config_id != 'all1':
        raise ValueError('growth profile uses fixed all1')
    spec = next(c for c in registry['configurations'] if c['id'] == config_id)
    rows = load_window(package, window_id, allowed_split=split)
    config = normalize_config(registry['engine_config'])
    identity_config = {'engine': config, 'policy': spec['policy'], 'window_id': window_id,
                       'split': split, 'mode': 'independent_closed_loop',
                       'journal_durability': 'flush_fsync_before_100tick_checkpoint; failed_runs_not_resumed_by_this_entrypoint'}
    inputs = {'package_manifest': Path(package) / 'manifest.json',
              'window': Path(package) / (window_id + '.jsonl'), 'registry': Path(registry_path),
              'contract': Path(registry['contract_path'])}
    run_id = f'{config_id}__{window_id}'
    context = RunContext.create(ROOT, output, run_id, identity_config, inputs,
                                canonical_v2_source_files(ROOT),
                                metadata={'phase': 'N3', 'profile': profile, 'real_backend': False,
                                          'input_requests': len(rows), 'gpu_executed': False})
    (context.run_dir / 'inputs').mkdir()
    bindings = {}
    for i, (name, path) in enumerate(inputs.items()):
        filename = f'{i:02d}_{path.name}'
        data = path.read_bytes()
        target = context.run_dir / 'inputs' / filename
        with target.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if digest(target) != context.planned['identity']['input_sha256'][name]:
            raise ValueError('input changed while bundling')
        bindings[name] = {'name': filename}
    durable_json(context.run_dir / 'inputs/bindings.json', bindings, exclusive=True)
    context.new_segment()
    stop = False
    def stopped(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, stopped)
    signal.signal(signal.SIGTERM, stopped)
    started = time.monotonic()
    last_heartbeat = started
    try:
        with (context.segment / 'events.jsonl').open('xb') as journal:
            def emit(event):
                journal.write(canonical_bytes(event))
            simulator = Simulator(rows, config, spec['policy'], emit=emit)
            context.checkpoint(simulator.snapshot())
            while not simulator.finished:
                simulator.step()
                checkpoint_due = simulator.now % 100 == 0 or simulator.finished or stop
                if checkpoint_due:
                    journal.flush()
                    os.fsync(journal.fileno())
                    context.checkpoint(simulator.snapshot())
                    context.append_metric({'tick': simulator.now, 'requests': simulator.counts['arrived'],
                                           'wall_seconds': time.monotonic() - started,
                                           'event_bytes': journal.tell()})
                    stop = stop or context.stop_requested()
                if time.monotonic() - last_heartbeat >= 1:
                    context.heartbeat()
                    last_heartbeat = time.monotonic()
                if stop and not simulator.finished:
                    context.finish('partial', 'controlled_stop', summary=simulator.summary())
                    return 2
        events = collect_events(context.run_dir)
        summary = simulator.summary()
        independent = recompute(events, config)
        assert_equal_summary(independent, summary)
        for filename, value in [('summary.json', summary), ('ledger.json', independent)]:
            durable_json(context.run_dir / filename, value, exclusive=True)
        for filename, kinds in [('action_trace.jsonl', {'policy_decision', 'startup', 'ready', 'drain', 'shutdown', 'drain_cancel'}),
                                ('completion_records.jsonl', {'local_complete', 'sink_complete', 'censor'})]:
            with (context.run_dir / filename).open('xb') as stream:
                for event in events:
                    if event['kind'] in kinds:
                        stream.write(canonical_bytes(event))
                stream.flush()
                os.fsync(stream.fileno())
        context.finish('completed', 'independent_ledger_matches', summary=summary)
        assert_equal_summary(rebuild(context.run_dir), summary)
        return 0
    except Exception as error:
        if (context.segment / 'worker_result.json').exists():
            # Terminal records are immutable. Keep post-completion audit failure
            # separate, and preserve its original exception for the supervisor.
            durable_json(context.run_dir / 'audit_failure.json',
                         {'stage': 'post_completion_audit', 'error': repr(error)}, exclusive=True)
        else:
            context.finish('failed', repr(error))
        raise


def matrix(registry_path, package, output, profile=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    registry = json.loads(Path(registry_path).read_text())
    validate_registry(registry, package)
    registry_sha = digest(registry_path)
    frozen_sources = source_hashes()
    plan = {'schema': 'maxopt-n3-matrix-plan-v1', 'profile': profile,
            'registry_sha256': registry_sha, 'code_sha256': frozen_sources,
            'package_manifest_sha256': registry['package_manifest_sha256'],
            'contract_sha256': registry['contract_sha256']}
    durable_json(output / 'matrix_plan.json', plan, exclusive=True)
    configurations = ['all1'] if profile else [c['id'] for c in registry['configurations']]
    windows = [f'train_{regime}' for regime in ('steady_v2', 'burst_offline_v2', 'recovery_v2')] if profile else sorted(registry['windows'])
    reports = []
    for config_id in configurations:
        for window in windows:
            if digest(registry_path) != registry_sha:
                raise ValueError('registry changed during matrix')
            if source_hashes() != frozen_sources:
                raise ValueError('source changed during matrix')
            run_id = f'{config_id}__{window}'
            command = [sys.executable, '-m', 'fidelityloop.legacy.maxopt_v2.search', 'worker', '--registry', str(registry_path),
                       '--package', str(package), '--output', str(output), '--config-id', config_id, '--window', window]
            if profile:
                command.append('--profile')
            result = supervise(command, ROOT, output / run_id, output / 'supervisors' / run_id,
                               Limits(sample_seconds=.1, min_disk_free_bytes=2*1024**3, timeout_seconds=1800,
                                      heartbeat_timeout_seconds=60, startup_grace_seconds=60))
            if not result.get('supervisor_result_saved'):
                raise RuntimeError(f'cell {run_id}: supervisor result was not durably saved')
            # The returned wrapper also reports persistence success. Bind the
            # matrix to the actual immutable observation file used at selection.
            result = json.loads((output / 'supervisors' / run_id / 'supervisor_result.json').read_text())
            record = {'config_id': config_id, 'window': window, 'run_id': run_id, 'supervisor': result}
            if (output / run_id / 'summary.json').exists():
                record['summary'] = json.loads((output / run_id / 'summary.json').read_text())
            reports.append(record)
            with (output / 'run_registry.jsonl').open('ab') as stream:
                stream.write(canonical_bytes(record))
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps({'run_id': run_id, 'exit_code': result['exit_code'], 'status': result['technical_status']}), flush=True)
            if result['exit_code'] != 0:
                raise RuntimeError(f'cell {run_id} failed; preserved, repair before continuation')
    if source_hashes() != frozen_sources:
        raise ValueError('source changed during matrix')
    durable_json(output / 'matrix.json', {'registry_sha256': registry_sha, 'profile': profile,
                                         'matrix_plan_sha256': digest(output / 'matrix_plan.json'),
                                         'cells': reports, 'test_opened': False}, exclusive=True)


def rank_candidates(registry, cells):
    expected = {(c['id'], w) for c in registry['configurations'] for w in registry['windows']}
    actual = {(c['config_id'], c['window']) for c in cells}
    if actual != expected or len(cells) != len(expected):
        raise ValueError('incomplete search matrix')
    duration = registry['engine_config']['window_seconds'] + registry['engine_config']['drain_seconds']
    thresholds = registry['engine_config']['feasibility']
    online_limit = thresholds['max_online_violation_rate']
    offline_min = thresholds['min_offline_deadline_completion_rate']
    for cell in cells:
        s = cell['summary']
        if cell['supervisor']['exit_code'] != 0 or cell['supervisor']['technical_status'] != 'exited_zero' or s['finished'] is not True or s['elapsed_seconds'] != duration:
            raise ValueError('incomplete/failed cell cannot be selected')
        for key in ('total_cost', 'online_violation_rate', 'offline_deadline_completion_rate'):
            value = s[key]
            if value is None and key != 'total_cost':
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != 'total_cost' and value > 1):
                raise ValueError('invalid selection metric')
    rows = []
    for candidate in registry['configurations']:
        summaries = [c['summary'] for c in cells if c['config_id'] == candidate['id']]
        feasible = all((c['online_violation_rate'] or 0) <= online_limit and
                       (c['offline_deadline_completion_rate'] is None or c['offline_deadline_completion_rate'] >= offline_min) for c in summaries)
        excess = sum(max(0, (c['online_violation_rate'] or 0) - online_limit) +
                     (max(0, offline_min - c['offline_deadline_completion_rate']) if c['offline_deadline_completion_rate'] is not None else 0) for c in summaries)
        rows.append({'id': candidate['id'], 'policy': candidate['policy'], 'feasible_all_windows': feasible,
                     'constraint_excess': excess, 'mean_cost': sum(c['total_cost'] for c in summaries) / len(summaries)})
    choices = [r for r in rows if r['policy']['name'] in {'hpa', 'hysteresis'}]
    feasible = [r for r in choices if r['feasible_all_windows']]
    chosen = min(feasible, key=lambda r: (r['mean_cost'], r['id'])) if feasible else None
    diagnostic = min(choices, key=lambda r: (r['constraint_excess'], r['mean_cost'], r['id']))
    return {'stage': 'N3_pilot_not_N4b_final', 'status': 'FEASIBLE_PILOT' if chosen else 'NO_FEASIBLE_POLICY',
            'selected': chosen, 'diagnostic_candidate': diagnostic, 'all_configs': rows,
            'environment_steps': len(expected) * duration, 'locked_test_opened': False}


def select(matrix_path, registry_path):
    matrix_path = Path(matrix_path)
    matrix = json.loads(matrix_path.read_text())
    registry = json.loads(Path(registry_path).read_text())
    validate_registry(registry, registry['package_path'])
    if matrix['registry_sha256'] != digest(registry_path) or matrix['profile'] or matrix.get('test_opened') is not False:
        raise ValueError('matrix not bound to search registry')
    result = rank_candidates(registry, matrix['cells'])
    plan_path = matrix_path.parent / 'matrix_plan.json'
    if not plan_path.is_file() or matrix.get('matrix_plan_sha256') != digest(plan_path):
        raise ValueError('matrix source plan missing or mismatched')
    plan = json.loads(plan_path.read_text())
    if plan['registry_sha256'] != digest(registry_path) or plan['profile'] is not False or plan['package_manifest_sha256'] != registry['package_manifest_sha256'] or plan['contract_sha256'] != registry['contract_sha256']:
        raise ValueError('matrix source plan differs from registration')
    for cell in matrix['cells']:
        expected_id = f'{cell["config_id"]}__{cell["window"]}'
        if cell['run_id'] != expected_id or Path(expected_id).name != expected_id:
            raise ValueError('cell identity mismatch')
        run_dir = matrix_path.parent / expected_id
        if (run_dir / 'audit_failure.json').exists():
            raise ValueError('cell post-completion audit failed')
        if not (run_dir / 'planned_manifest.json').is_file():
            raise ValueError('cell run evidence is missing')
        summary = rebuild(run_dir)
        assert_equal_summary(summary, cell['summary'])
        observed = json.loads((matrix_path.parent / 'supervisors' / expected_id / 'supervisor_result.json').read_text())
        if observed != cell['supervisor']:
            raise ValueError('supervisor record mismatch')
        planned = json.loads((run_dir / 'planned_manifest.json').read_text())
        spec = next(c['policy'] for c in registry['configurations'] if c['id'] == cell['config_id'])
        expected_inputs = {'window': registry['windows'][cell['window']],
                           'package_manifest': registry['package_manifest_sha256'],
                           'contract': registry['contract_sha256'], 'registry': digest(registry_path)}
        if planned['config']['window_id'] != cell['window'] or planned['config']['split'] != 'validation' or planned['config']['policy'] != spec or planned['config']['engine'] != normalize_config(registry['engine_config']) or planned['identity']['input_sha256'] != expected_inputs:
            raise ValueError('run not bound to preregistered experiment')
        if planned['identity']['code_sha256'] != plan['code_sha256'] or planned['identity']['source_set_content_sha256'] != hashlib.sha256(canonical_bytes(plan['code_sha256'])).hexdigest():
            raise ValueError('mixed source versions in matrix')
        rows = load_window(registry['package_path'], cell['window'], allowed_split='validation')
        events = collect_events(run_dir)
        input_sha = hashlib.sha256(b''.join(canonical(row) + b'\n' for row in rows)).hexdigest()
        if events[0]['input_sha256'] != input_sha or events[0]['policy'] != TargetPolicy(spec).spec or events[0]['mode'] != 'independent_closed_loop':
            raise ValueError('journal not bound to registered input/policy/mode')
        keys = ('request_id', 'job_type', 'arrival_s', 'deadline_s', 'input_tokens', 'max_output_tokens')
        if [{k: e[k] for k in keys} for e in events if e['kind'] == 'arrival'] != [{k: row[k] for k in keys} for row in rows]:
            raise ValueError('journal arrivals differ from registered workload')
    return {**result, 'matrix_sha256': digest(matrix_path), 'registry_sha256': digest(registry_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['worker', 'matrix', 'select'])
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-id')
    parser.add_argument('--window')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.operation == 'worker':
        raise SystemExit(worker(args.registry, args.package, args.output, args.config_id, args.window, profile=args.profile))
    if args.operation == 'matrix':
        matrix(args.registry, args.package, args.output, args.profile)
    else:
        result = select(args.output / 'matrix.json', args.registry)
        durable_json(args.output / 'selection.json', result, exclusive=True)
        print(json.dumps(result, indent=2))
