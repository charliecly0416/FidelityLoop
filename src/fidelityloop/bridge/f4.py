"""Bounded F4 historical replay; never generates final test requests or trains."""
import argparse
import copy
import csv
import gzip
import io
import json
import math
from pathlib import Path
import statistics
import time

from fidelityloop.bridge import protocol
from fidelityloop.bridge.metrics import cost_breakdown, evaluate_requests
from fidelityloop.legacy.maxopt_v3.audit import audit as audit_events
from fidelityloop.legacy.maxopt_v3.search import choose
from fidelityloop.legacy.maxopt_v3.simulator import Simulator


def validate_arm(arm):
    """Fail closed on ID/spec mismatches, including half-enabled guards."""
    if not isinstance(arm, dict) or arm.get('id') not in protocol.F4_IDS:
        raise ValueError('unregistered F4 policy identity')
    index = protocol.F4_IDS.index(arm['id'])
    guarded = bool(index % 2)
    expected = dict(id=protocol.F4_IDS[index], policy=dict(
        name='hysteresis', up_threshold=8, down_threshold=index // 2,
        cooldown_seconds=60, min_devices=0), scale=guarded,
        reservation=guarded, guard_model='E')
    if arm != expected:
        raise ValueError('F4 identity and exact policy/guard specification disagree')
    return copy.deepcopy(arm)


def load_context(root=protocol.ROOT):
    root = Path(root).resolve()
    receipt = protocol.check(root)
    review = protocol.read(root / protocol.DOC / 'S01_REVIEW.json')
    if (review['status'] != 'PASS_S01_ONLY' or
            receipt['contract_sha256'] != review['reviewed_contract_sha256']):
        raise ValueError('F4 requires the independently accepted S01 contract SHA')
    source = root / protocol.INPUT
    contract = protocol.read(root / protocol.DOC / 'CONTRACT.json')
    arms = {a['id']: validate_arm(a) for a in contract['f4_members']}
    if tuple(arms) != protocol.F4_IDS:
        raise ValueError('incomplete or reordered F4 registry')
    return dict(root=root, receipt=receipt, contract=contract, arms=arms,
                models=protocol.read(source / 'models.json'),
                safety=protocol.read(source / 'guard_safety.json'),
                simulator_contract=protocol.read(source / 'simulator_contract.json'),
                candidates=protocol.read(source / 'candidates.json'))


def simulator(context, rows, policy_id, prediction_model):
    if prediction_model not in ('C', 'C30'):
        raise ValueError('F4 replay permits only C and C30 prediction environments')
    if policy_id not in context['arms']:
        raise ValueError('unregistered F4 policy identity')
    arm = validate_arm(context['arms'][policy_id])
    return Simulator(rows, context['simulator_contract'], context['models'][prediction_model],
                     arm['policy'], safety=context['safety'], guard_model=context['models']['E'],
                     scale=arm['scale'], reservation=arm['reservation'])


def replay_cell(context, rows, policy_id, prediction_model):
    sim = simulator(context, rows, policy_id, prediction_model)
    result = sim.run()
    arm = context['arms'][policy_id]
    verified = audit_events(rows, result['events'], result['summary'], context['simulator_contract'],
                            context['models'][prediction_model], arm['policy'], result['requests'],
                            safety=context['safety'], guard_model=context['models']['E'],
                            scale=arm['scale'], reservation=arm['reservation'])
    metrics = evaluate_requests(rows, result['requests'], horizon=sim.end)
    costs = cost_breakdown(rows, metrics, occupied_seconds=sim.occupied, starts=sim.starts,
                           shutdowns=sim.shutdowns, api_ids=sim.api_ids,
                           prices=context['simulator_contract']['accounting'])
    for name, value in result['summary']['costs'].items():
        if not math.isclose(value, costs[name], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError('registered cost evaluator disagrees: ' + name)
    for kind, population in result['summary']['populations'].items():
        if any(population[key] != metrics['populations'][kind][key]
               for key in ('arrivals', 'on_time', 'not_on_time', 'on_time_rate')):
            raise ValueError('registered request evaluator disagrees: ' + kind)
    return dict(schema='maxopt-bridge-f4-replay-cell-v1', prediction_model=prediction_model,
                policy_id=policy_id, window_id=rows[0]['window_id'],
                result=result, event_audit=verified, request_metrics=metrics, costs=costs)


def historical_selection(records, candidates, rankings):
    """Rebuild old rankings from all 50 candidates, without executing old CLIs."""
    ids = {c['id'] for c in candidates}
    if len(candidates) != 50 or len(ids) != 50:
        raise ValueError('historical selection requires the complete 50-candidate pool')
    output = {}
    for model in ('C', 'C30', 'E', 'O', 'V'):
        rows = [r for r in records if r['model'] == model]
        keys = {(r['configuration'], r['window']) for r in rows}
        windows = {r['window'] for r in rows}
        if (len(rows) != 300 or len(keys) != 300 or len(windows) != 6 or
                keys != {(candidate, window) for candidate in ids for window in windows}):
            raise ValueError('incomplete historical search matrix for ' + model)
        rebuilt = choose(rows, candidates)
        if rebuilt != rankings[model]:
            raise ValueError('historical full-pool ranking differs for ' + model)
        winner = rebuilt['selected']
        tied = [r['id'] for r in rebuilt['ranking'] if r['feasible'] == winner['feasible']
                and r['mean_cost'] == winner['mean_cost']
                and r['worst_offline_slack'] == winner['worst_offline_slack']]
        output[model] = dict(selected=winner, equal_cost_and_slack_ids=tied,
                             candidate_count=50, development_cells=300,
                             ranking_rebuilt_exactly=True)
    return output


def compare_historical(cell, historical):
    """Require historical summaries to agree; retain separate new P1plus results."""
    current = copy.deepcopy(cell['result']['summary'])
    current.pop('outside_window')
    expected = historical['summary']
    differences = []

    def compare(a, b, path='summary'):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences.append(path + ': keys')
                return
            for key in a:
                compare(a[key], b[key], path + '.' + key)
        elif type(a) in (int, float) and type(b) in (int, float):
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8):
                differences.append(path)
        elif a != b:
            differences.append(path)

    compare(current, expected)
    return dict(matches=not differences, differences=differences,
                comparison='all historical summary fields; abs_tol=1e-8, rel_tol=1e-10')


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _compressed(value):
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
        zipped.write(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    return stream.getvalue()


def build(root=protocol.ROOT):
    started = time.monotonic()
    context = load_context(root)
    root = context['root']
    doc, inputs = root / protocol.DOC, root / protocol.INPUT
    output = root / 'artifacts/maxopt_bridge_20260930/f4'
    if output.exists():
        raise ValueError('F4 output already exists; preserve completed evidence')
    records = protocol.read(inputs / 'historical/search_results.json')
    rankings = protocol.read(inputs / 'historical/rankings.json')
    selection = historical_selection(records, context['candidates'], rankings)
    by_key = {(r['model'], r['configuration'], r['window']): r for r in records}
    windows = sorted((inputs / 'development').glob('*.jsonl'))
    if len(windows) != 6:
        raise ValueError('exactly six registered old development files are required')
    baseline = protocol.read(doc / 'BASELINE_MANIFEST.json')
    input_bindings = {str(p.relative_to(root)): protocol.sha256(p)
                      for p in [doc / 'CONTRACT.json', doc / 'SOURCE_MANIFEST.json',
                                *windows, *[inputs / n for n in ('models.json', 'candidates.json',
                                'guard_safety.json', 'simulator_contract.json')],
                                *[inputs / 'historical' / n for n in
                                  ('search_results.json', 'rankings.json', 'selection.json')]]}
    output.mkdir(parents=True)
    manifest = dict(schema='maxopt-bridge-f4-policy-manifest-v1', status='HISTORICAL_REPLAY_ONLY',
                    contract_sha256=context['receipt']['contract_sha256'],
                    members=list(context['arms'].values()), initial_target=2,
                    initial_devices='two active, empty queues', decision_seconds=1,
                    observation_schema='maxopt-v3-observation-v1', input_sha256=input_bindings,
                    full_pool_selection=selection, final_test_inputs_generated=False,
                    deployment_status='NOT_RUN_NEEDS_GPU_DEVELOPMENT_PREFLIGHT')
    _write_json(doc / 'A_POLICY_MANIFEST.json', manifest)
    limit = context['contract']['budget']
    cap_seconds = limit['development_replay_wall_hours_cap'] * 3600
    cap_bytes = limit['development_compressed_output_GiB_cap'] * 1024 ** 3
    table, artifact_index, compressed_bytes = [], [], 0
    for model in ('C', 'C30'):
        for policy_id in protocol.F4_IDS:
            for path in windows:
                if time.monotonic() - started >= cap_seconds:
                    raise RuntimeError('F4 one-hour development replay budget exhausted')
                rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
                cell = replay_cell(context, rows, policy_id, model)
                historical = by_key[(model, policy_id, path.stem)]
                comparison = compare_historical(cell, historical)
                cell['historical_comparison'] = comparison
                cell['input_sha256'] = protocol.sha256(path)
                blob = _compressed(cell)
                if compressed_bytes + len(blob) > cap_bytes:
                    raise RuntimeError('F4 compressed output budget exhausted')
                dest = output / model / policy_id / (path.stem + '.json.gz')
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as stream:
                    stream.write(blob)
                compressed_bytes += len(blob)
                artifact_index.append(dict(path=dest.relative_to(root).as_posix(),
                                           sha256=protocol.sha256(dest), bytes=len(blob)))
                metrics, costs = cell['request_metrics'], cell['costs']
                rank = next(i + 1 for i, r in enumerate(rankings[model]['ranking']) if r['id'] == policy_id)
                row = dict(prediction_model=model, policy_id=policy_id, window=path.stem,
                           split=rows[0]['split'], full_pool_rank=rank,
                           full_pool_selected=selection[model]['selected']['id'],
                           is_full_pool_selected=policy_id == selection[model]['selected']['id'],
                           tied_with_selected_before_id=policy_id in selection[model]['equal_cost_and_slack_ids'],
                           historical_summary_match=comparison['matches'],
                           legacy_summary_feasible=cell['result']['summary']['feasible'],
                           historical_P1=metrics['historical_P1'], P1plus=metrics['P1plus'],
                           online_arrivals=metrics['populations']['online']['arrivals'],
                           online_on_time=metrics['populations']['online']['on_time'],
                           offline_arrivals=metrics['populations']['offline']['arrivals'],
                           offline_on_time=metrics['populations']['offline']['on_time'],
                           unfinished=metrics['unfinished'], late=metrics['late'],
                           **{k + '_cost': costs[k] for k in ('gpu', 'startup', 'shutdown',
                              'synthetic_api', 'operating', 'offline_miss', 'total')},
                           historical_total_cost=historical['summary']['costs']['total'],
                           gpu_occupied_seconds=cell['result']['summary']['gpu_occupied_seconds'],
                           startup_events=cell['result']['summary']['startup_events'],
                           shutdown_events=cell['result']['summary']['shutdown_events'],
                           guard_ticks=sum(bool(e.get('guard', {}).get('risky_ids')) for e in cell['result']['events']),
                           event_audit=cell['event_audit']['status'],
                           evidence_path=dest.relative_to(root).as_posix(), input_sha256=cell['input_sha256'])
                table.append(row)
                print(json.dumps(dict(completed=len(table), total=48, model=model,
                                      policy_id=policy_id, window=path.stem,
                                      historical_match=comparison['matches'], P1plus=metrics['P1plus'])), flush=True)
    if len(table) != 48 or any(not r['historical_summary_match'] for r in table):
        raise ValueError('historical F4 replay is incomplete or differs; retain raw evidence for review')
    csv_path = doc / 'A_DECISION_LINK_TABLE.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    for relative, digest in input_bindings.items():
        if protocol.sha256(root / relative) != digest:
            raise ValueError('input changed during replay: ' + relative)
    for relative, digest in baseline['protected_files'].items():
        if protocol.sha256(root / relative) != digest:
            raise ValueError('protected source changed: ' + relative)
    receipt = dict(schema='maxopt-bridge-f4-replay-receipt-v1', cells=len(table),
                   elapsed_seconds=time.monotonic() - started, compressed_bytes=compressed_bytes,
                   historical_summaries_match=True, event_audits_passed=48,
                   P1plus_cells=sum(r['P1plus'] for r in table),
                   historical_P1_cells=sum(r['historical_P1'] for r in table),
                   artifact_index=artifact_index, input_sha256=input_bindings,
                   table_sha256=protocol.sha256(csv_path), sources_unchanged=True,
                   protected_files_verified=len(baseline['protected_files']),
                   scope='48 old development cells, not final prediction or physical results')
    _write_json(output / 'REPLAY_RECEIPT.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=protocol.ROOT)
    print(json.dumps(build(parser.parse_args().root), ensure_ascii=False, indent=2))
