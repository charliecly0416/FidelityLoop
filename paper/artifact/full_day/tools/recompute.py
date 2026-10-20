"""Offline audit of anonymous relative-time projections; Python standard library."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 86700
NS = 10**9
PRICES = dict(gpu=.001, startup=.002, shutdown=.001, api_input=1e-6,
              api_output=3e-6, offline_miss=.01)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def quantile(values, q):
    require(values and 0 <= q <= 1, 'invalid quantile')
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def score(expected, completed, grace=0):
    require(math.isfinite(grace) and grace >= 0, 'invalid scoring grace')
    counts = {kind: dict(arrivals=0, on_time=0, late=0, unfinished=0)
              for kind in ('online', 'offline')}
    require(set(completed) <= set(expected), 'unknown completion ID')
    for rid, row in expected.items():
        p = counts[row['job_type']]
        p['arrivals'] += 1
        at = completed.get(rid)
        require(at is None or math.isfinite(at) and at >= row['arrival_s'], 'invalid completion time')
        if at is None or at > HORIZON:
            outcome = 'unfinished'
        else:
            outcome = 'on_time' if at <= row['deadline_s'] + (grace if row['job_type'] == 'offline' else 0) else 'late'
        p[outcome] += 1
    for p in counts.values():
        p['on_time_rate'] = p['on_time'] / p['arrivals'] if p['arrivals'] else None
    online = counts['online']
    offline = counts['offline']
    return dict(populations=counts,
                online_only=online['arrivals'] > 0 and online['on_time'] == online['arrivals'],
                P1plus=all(p['arrivals'] > 0 and p['on_time'] == p['arrivals'] for p in counts.values()),
                offline_successful_fraction=offline['on_time_rate'])


def occupied(lifecycle):
    launches, releases, ready = {}, {}, {}
    starts = stops = 0
    last_seq = -1
    for event in lifecycle:
        require(event['seq'] > last_seq, 'lifecycle sequence not increasing')
        last_seq = event['seq']
        key = (event['gpu'], event['generation'])
        require(event['gpu'] in (0, 1) and event['generation'] >= 1, 'invalid generation identity')
        at = event['at_ns']
        require(type(at) is int, 'noninteger relative clock')
        kind = event['kind']
        if kind == 'launch_issued':
            require(key not in launches, 'duplicate launch')
            launches[key] = at
            starts += event['phase'] == 'window' and 0 <= at < HORIZON * NS
        elif kind == 'verified_release':
            require(key in launches and key not in releases and at >= launches[key], 'invalid release')
            require(event['release_verified'] is True, 'release lacks verification')
            releases[key] = at
        elif kind in ('ready', 'ready_during_stopping'):
            require(key in launches and at >= launches[key] and key not in ready, 'invalid readiness')
            ready[key] = at
        elif kind == 'shutdown_issued':
            require(key in launches and at >= launches[key], 'shutdown without launch')
            stops += event['phase'] == 'window' and 0 <= at < HORIZON * NS
        else:
            raise ValueError('unknown lifecycle kind')
    require(set(launches) == set(releases), 'missing verified release after cleanup')
    by_gpu = {}
    for gpu in (0, 1):
        intervals = sorted((start, releases[key]) for key, start in launches.items() if key[0] == gpu)
        require(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])),
                'overlapping generations on a GPU')
        by_gpu[str(gpu)] = sum(max(0, min(HORIZON * NS, end) - max(0, start))
                               for start, end in intervals) / NS
    total = sum(by_gpu.values())
    require(0 <= total <= 2 * HORIZON + 1e-9, 'occupied time exceeds capacity')
    startups = [(ready[k] - start) / NS for k, start in launches.items()
                if 0 <= start < HORIZON * NS and k in ready]
    return dict(gpu_seconds=total, per_gpu_seconds=by_gpu, startup_events=starts, shutdown_events=stops,
                startup_sample_count=len(startups), startup_p95=quantile(startups, .95) if startups else None)


def selections(predictions):
    output = {}
    for model in ('Share', 'Lookup69', 'Lookup30', 'Local'):
        candidates = []
        for policy in ('H', 'Hguard'):
            group = [p for p in predictions if p['model'] == model and p['policy'] == policy]
            require(len(group) == 2 and {p['day'] for p in group} == {'D1', 'D2'}, 'selection pool incomplete')
            if all(p['P1plus'] for p in group):
                candidates.append((sum(p['frozen_scenario_cost'] for p in group) / 2, policy))
        best = min((cost for cost, _ in candidates), default=None)
        output[model] = [policy for cost, policy in candidates if abs(cost - best) <= 1e-9]
    return output


def verify_inventory(root):
    manifest = read(root / 'FILES_SHA256.json')['files']
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'symlink in artifact')
        if path.is_file() and path != root / 'FILES_SHA256.json':
            rel = path.relative_to(root).as_posix()
            actual.add(rel)
            require(rel in manifest, 'unlisted member: ' + rel)
            require(path.stat().st_size == manifest[rel]['bytes'] and
                    hashlib.sha256(path.read_bytes()).hexdigest() == manifest[rel]['sha256'],
                    'content differs: ' + rel)
    require(actual == set(manifest), 'missing member')


def privacy_check(root):
    # Positive allowlist during export is primary; this catches common accidental leaks.
    forbidden = ('/home/', '/lustre/', '/Users/', 'GPU-',
                 'model_path', 'origin_monotonic_ns', 'controller_monotonic_ns',
                 'worker_monotonic_ns', 'hostname', 'ownership', '"pid"', '"utc"')
    scanned = 0
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix == '.gz':
            stream = gzip.open(path, 'rt')
        else:
            stream = path.open()
        with stream:
            for line in stream:
                # This scanner necessarily contains its own forbidden strings.
                if path.name == 'recompute.py':
                    continue
                require(not any(term.lower() in line.lower() for term in forbidden),
                        'private identifier pattern in ' + path.relative_to(root).as_posix())
        scanned += 1
    return scanned


def close(actual, expected, path='value'):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'mapping keys differ: ' + path)
        for key in expected:
            close(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'list size differs: ' + path)
        for index, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, path + '.' + str(index))
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-7),
                'number differs: ' + path)
    else:
        require(actual == expected, 'value differs: ' + path)


def recompute(root=ROOT, check_expected=True):
    root = Path(root)
    verify_inventory(root)
    scanned = privacy_check(root)
    protocol = read(root / 'protocol/PROTOCOL.json')
    inputs = {}
    for day in ('D1', 'D2'):
        rows = list(records(root / f'data/inputs/{day}.jsonl.gz'))
        require(len({r['request_id'] for r in rows}) == len(rows), 'duplicate expected ID')
        require(all(r['deadline_s'] == r['arrival_s'] + (60 if r['job_type'] == 'online' else 300)
                    and 0 <= r['arrival_s'] < 86400 for r in rows), 'input deadline/arrival differs')
        inputs[day] = {r['request_id']: r for r in rows}
    runs, completed_by_run, grace_rows = [], {}, []
    for info in protocol['runs']:
        rid = info['run_id']
        folder = root / 'data/runs' / rid
        expected = inputs[info['day']]
        terminals, dispatches, api_ids = {}, {}, set()
        for event in records(folder / 'request_events.jsonl.gz'):
            request_id = event['request_id']
            require(request_id in expected, 'unknown request')
            at = event['at_ns'] / NS
            require(at >= expected[request_id]['arrival_s'], 'event before arrival')
            if event['kind'] == 'terminal':
                require(request_id not in terminals, 'duplicate terminal')
                if event['status'] == 'completed':
                    require(request_id in dispatches and at >= dispatches[request_id], 'completion before dispatch')
                terminals[request_id] = at if event['status'] == 'completed' else None
            elif event['kind'] == 'dispatch':
                require(request_id not in dispatches, 'duplicate dispatch')
                dispatches[request_id] = at
                if event['route'] == 'synthetic_api':
                    api_ids.add(request_id)
            else:
                require(event['kind'] == 'release', 'unknown request event')
        require(set(terminals) == set(expected), 'terminal population incomplete')
        physical = score(expected, terminals)
        completed_by_run[rid] = terminals
        for grace in protocol['deadline_score_graces']:
            p = score(expected, terminals, grace)['populations']['offline']
            grace_rows.append(dict(run_id=rid, grace_seconds=grace, **p))
        lifecycle = occupied(list(records(folder / 'lifecycle.jsonl.gz')))
        ticks, max_lateness, max_actuation, gaps, abandoned = [], 0., 0., 0, 0
        for event in records(folder / 'control.jsonl.gz'):
            if event['kind'] == 'policy_tick':
                ticks.append(event['tick'])
                require(event['target'] in (0, 1, 2), 'invalid target')
                require(event['scheduled_ns'] == event['tick'] * NS, 'schedule moved')
                require(event['actuation_ns'] >= event['scheduled_ns'], 'actuation backdated')
                require(sum(event['observation'][key] for key in ('active', 'starting', 'draining', 'off')) == 2,
                        'device count differs')
                max_lateness = max(max_lateness, event['lateness_seconds'])
                max_actuation = max(max_actuation, (event['actuation_ns'] - event['scheduled_ns']) / NS)
            elif event['kind'] == 'policy_gap':
                gaps += 1
            elif event['kind'] == 'decision_abandoned':
                abandoned += 1
            else:
                raise ValueError('unknown control record')
        require(ticks == list(range(HORIZON)) and not gaps and not abandoned, 'control coverage differs')
        api = list(records(folder / 'api_accept.jsonl.gz'))
        require({e['request_id'] for e in api} == api_ids and len(api) == len(api_ids), 'API acceptance/dispatch differs')
        require(all(expected[e['request_id']]['job_type'] == 'online' and
                    (e['input_tokens'], e['output_budget']) ==
                    (expected[e['request_id']]['input_tokens'], expected[e['request_id']]['max_output_tokens'])
                    for e in api), 'API token ledger differs')
        offline = physical['populations']['offline']
        costs = dict(gpu=lifecycle['gpu_seconds'] * PRICES['gpu'],
                     startup=lifecycle['startup_events'] * PRICES['startup'],
                     shutdown=lifecycle['shutdown_events'] * PRICES['shutdown'],
                     synthetic_api=sum(e['input_tokens'] * PRICES['api_input'] +
                                       e['output_budget'] * PRICES['api_output'] for e in api),
                     offline_miss=(offline['late'] + offline['unfinished']) * PRICES['offline_miss'])
        operating = sum(costs[k] for k in ('gpu', 'startup', 'shutdown', 'synthetic_api'))
        violations = [dict(request_id=key, arrival_s=row['arrival_s'], deadline_s=row['deadline_s'],
                           completed_at=terminals[key], dispatched_at=dispatches.get(key),
                           outcome='unfinished' if terminals[key] is None or terminals[key] > HORIZON else 'late')
                      for key, row in expected.items() if terminals[key] is None or
                      terminals[key] > min(row['deadline_s'], HORIZON)]
        runs.append(dict(**info, **physical, **lifecycle, costs=costs, operating_cost=operating,
                         scenario_cost=operating + costs['offline_miss'], api_accepted=len(api),
                         policy_ticks=len(ticks), control_gaps=gaps, abandoned_decisions=abandoned,
                         max_tick_lateness=max_lateness, max_actuation_lateness=max_actuation,
                         violations=violations))
    predictions = []
    for item in protocol['predictions']:
        rows = list(records(root / item['file']))
        require(len(rows) == len(inputs[item['day']]) and len({r['request_id'] for r in rows}) == len(rows),
                'prediction population differs')
        predicted = score(inputs[item['day']], {r['request_id']: r['completed_at'] for r in rows})
        predictions.append(dict(day=item['day'], model=item['model'], policy=item['policy'],
                                P1plus=predicted['P1plus'], populations=predicted['populations'],
                                frozen_scenario_cost=item['frozen_scenario_cost']))
    by_id = {r['run_id']: r for r in runs}
    primary = [r for r in runs if r['repeat'] == 1 and r['policy'] != 'all2']
    dynamic = [r for r in runs if r['policy'] != 'all2']
    metric_projection = dict(scope='fixed-trajectory metric projection, not removing offline arrivals or rerunning policies',
        primary_dynamic_cases=len(primary), primary_online_only_pass=sum(r['online_only'] for r in primary),
        primary_mixed_pass=sum(r['P1plus'] for r in primary),
        primary_online_only_pass_but_mixed_fail=sum(r['online_only'] and not r['P1plus'] for r in primary),
        all_dynamic_cases=len(dynamic), all_dynamic_online_only_pass=sum(r['online_only'] for r in dynamic),
        all_dynamic_mixed_pass=sum(r['P1plus'] for r in dynamic),
        all_runs_online_only_pass=sum(r['online_only'] for r in runs), all_runs_mixed_pass=sum(r['P1plus'] for r in runs))
    confusion = {}
    for model in ('Share', 'Lookup69', 'Lookup30', 'Local'):
        counter = Counter()
        for case in primary:
            pred = next(p for p in predictions if (p['day'], p['policy'], p['model']) ==
                        (case['day'], case['policy'], model))
            counter['false_accept' if pred['P1plus'] and not case['P1plus'] else
                    'false_reject' if not pred['P1plus'] and case['P1plus'] else 'agreement'] += 1
        confusion[model] = dict(agreement=counter['agreement'], false_accept=counter['false_accept'],
                                false_reject=counter['false_reject'], cases=4)
    quantile_rows = []
    for item in protocol['startup_sensitivity']:
        rows = list(records(root / item['file']))
        require(len(rows) == len(inputs[item['day']]) and len({r['request_id'] for r in rows}) == len(rows),
                'startup sensitivity population differs')
        metrics = score(inputs[item['day']], {r['request_id']: r['completed_at'] for r in rows})
        quantile_rows.append(dict(day=item['day'], policy=item['policy'], quantile=item['quantile'],
                                  startup_seconds=item['startup_seconds'], predicted_P1plus=metrics['P1plus'],
                                  physical_P1plus=by_id[f"{item['day']}_{item['policy']}_r1"]['P1plus'],
                                  frozen_scenario_cost=item['frozen_scenario_cost']))
    sweep = []
    for q in (.5, .75, .9, .95, 1.):
        group = [r for r in quantile_rows if r['quantile'] == q]
        candidates = []
        for policy in ('H', 'Hguard'):
            pair = [r for r in group if r['policy'] == policy]
            require(len(pair) == 2, 'startup sweep incomplete')
            if all(r['predicted_P1plus'] for r in pair):
                candidates.append((sum(r['frozen_scenario_cost'] for r in pair) / 2, policy))
        best = min((c for c, _ in candidates), default=None)
        sweep.append(dict(quantile=q, choices=[p for c, p in candidates if abs(c - best) <= 1e-9],
                          agreement=sum(r['predicted_P1plus'] == r['physical_P1plus'] for r in group)))
    saving = 100 * (1 - by_id['D2_Hguard_r1']['scenario_cost'] / by_id['D2_all2_r1']['scenario_cost'])
    result = dict(runs=runs, predictions=predictions, selection=selections(predictions), confusion=confusion,
                  metric_projection=metric_projection, deadline_score_sensitivity=grace_rows,
                  startup_statistic_sensitivity=quantile_rows, startup_statistic_selection=sweep,
                  d2_conditional_saving_percent=saving)
    if check_expected:
        close(result, read(root / 'expected/RECOMPUTED_RESULTS.json'))
    return dict(status='PASS', scope='anonymous audit projection; no simulator or GPU execution',
                privacy_scanned_files=scanned, results=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='optional NEW JSON output file outside this frozen directory')
    args = parser.parse_args()
    value = recompute()
    if args.output:
        target = Path(args.output).resolve()
        require(not target.is_relative_to(ROOT.resolve()), 'write output outside frozen artifact')
        with target.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k: value[k] for k in ('status', 'scope', 'privacy_scanned_files')}, indent=2))
    print(json.dumps(dict(physical_runs=len(value['results']['runs']),
                         frozen_predictions=len(value['results']['predictions']),
                         selection=value['results']['selection'],
                         metric_projection=value['results']['metric_projection']), indent=2))
