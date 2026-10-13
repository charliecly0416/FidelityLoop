"""Inspect screening errors from paired records: no GPU, weights or network."""
import argparse
import json
from collections import defaultdict
from statistics import mean
from load_trace import ROOT, load_trace, read_index, measure, check_reference


def pair_measures(physical, prediction):
    if (physical['campaign'], physical['policy'], physical['window']) != (
            prediction['campaign'], prediction['policy'], prediction['window']):
        raise ValueError('paired experiment coordinates differ')
    if physical['kind'] != 'physical' or prediction['kind'] != 'prediction':
        raise ValueError('pair kinds differ')
    if physical['prices'] != prediction['prices'] or physical['horizon_s'] != prediction['horizon_s']:
        raise ValueError('paired accounting contracts differ')
    fields = ('request_id', 'job_type', 'arrival_s', 'deadline_s', 'input_tokens', 'max_output_tokens')
    rows = lambda r: sorted(tuple(x[k] for k in fields) for x in r['requests'])
    if rows(physical) != rows(prediction):
        raise ValueError('paired request schedules differ')
    a, b = measure(physical), measure(prediction)
    check_reference(physical, a)
    check_reference(prediction, b)
    return a, b


def inspect(campaign_name='E6', root=ROOT):
    index = read_index(root)
    campaign = next((c for c in index['campaigns'] if c['campaign'] == campaign_name), None)
    if campaign is None:
        raise ValueError('unknown campaign')
    entries = {x['record_id']: x for k in ('physical', 'predictions') for x in campaign[k]}
    loaded = {}
    def record(rid):
        if rid not in loaded:
            loaded[rid] = load_trace(entries[rid], root)
        return loaded[rid]
    groups, mismatches = defaultdict(list), []
    seen = set()
    for pair in campaign['pairs']:
        key = pair['physical_id'], pair['prediction_id']
        if key in seen:
            raise ValueError('duplicate pair')
        seen.add(key)
        physical, prediction = map(record, key)
        # Match the paper's original per-campaign comparison populations.
        if campaign_name in ('E3', 'E4', 'E5', 'E6') and physical['policy'] == 'all2':
            continue
        a, b = pair_measures(physical, prediction)
        model = prediction['replay_model']
        groups[model].append((physical['window'], a, b, key[1]))
        if a['P1plus'] != b['P1plus']:
            missed = [r['request_id'] for r in physical['requests']
                      if r['status'] != 'completed' or r['completed_at_s'] is None
                      or r['completed_at_s'] > min(r['deadline_s'], physical['horizon_s'])]
            mismatches.append(dict(physical_id=key[0], prediction_id=key[1],
                replay_model=model, error='false_accept' if b['P1plus'] else 'false_reject',
                physical_offline=a['populations']['offline'],
                predicted_offline=b['populations']['offline'],
                physical_missed_request_ids=missed))
    summaries = []
    for model, pairs in sorted(groups.items()):
        feasible = sum(a['P1plus'] for _, a, _, _ in pairs)
        infeasible = len(pairs) - feasible
        fa = sum(not a['P1plus'] and b['P1plus'] for _, a, b, _ in pairs)
        fr = sum(a['P1plus'] and not b['P1plus'] for _, a, b, _ in pairs)
        window_errors = defaultdict(list)
        for w, a, b, _ in pairs:
            window_errors[w].append((abs(a['gpu_occupied_seconds']-b['gpu_occupied_seconds']),
                                     abs(a['cost_total']-b['cost_total'])))
        summaries.append(dict(replay_model=model, compared_runs=len(pairs),
            unique_predictions=len({p for _, _, _, p in pairs}),
            agreement=len(pairs)-fa-fr, false_accepts=fa, false_rejects=fr,
            physically_feasible_runs=feasible, physically_infeasible_runs=infeasible,
            false_accept_rate=fa/infeasible if infeasible else None,
            false_reject_rate=fr/feasible if feasible else None,
            occupied_mae_gpu_seconds=mean(mean(x[0] for x in values) for values in window_errors.values()),
            cost_mae_scenario_USD=mean(mean(x[1] for x in values) for values in window_errors.values())))
    return dict(campaign=campaign_name,
        scope='Original frozen predictions paired with physical runs. Equal-window errors; repeated runs share predictions. Null rates have empty denominators. This does not rerun inference or evaluate a new policy.',
        summaries=summaries, mismatches=mismatches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default='E6')
    parser.add_argument('--root', type=__import__('pathlib').Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(inspect(args.campaign, args.root), indent=2))
