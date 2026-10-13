"""Study metrics with explicit denominators; no policy selection or execution."""
import math


def evaluate_requests(rows, records, *, horizon=2100):
    ids = [r['request_id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(records):
        raise ValueError('request identities must be unique and exactly match records')
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('invalid observation horizon')
    populations = {k: dict(arrivals=0, completed=0, on_time=0, late=0, unfinished=0)
                   for k in ('online', 'offline')}
    post_horizon = 0
    for row in rows:
        kind = row['job_type']
        if kind not in populations:
            raise ValueError('unknown job type')
        arrival, deadline = row['arrival_s'], row['deadline_s']
        if not all(math.isfinite(x) for x in (arrival, deadline)) or not 0 <= arrival <= deadline or arrival >= horizon:
            raise ValueError('invalid arrival or deadline')
        record = records[row['request_id']]
        if 'completed_at' not in record:
            raise ValueError('missing terminal field; missing log is not an unfinished result')
        completed = record['completed_at']
        if completed is not None and (not math.isfinite(completed) or completed < arrival):
            raise ValueError('invalid completion timestamp')
        p = populations[kind]
        p['arrivals'] += 1
        if completed is None or completed > horizon:
            p['unfinished'] += 1
            post_horizon += completed is not None
        else:
            p['completed'] += 1
            p['on_time' if completed <= deadline else 'late'] += 1
    for p in populations.values():
        p['not_on_time'] = p['late'] + p['unfinished']
        p['on_time_rate'] = p['on_time'] / p['arrivals'] if p['arrivals'] else None
    online, offline = (populations[k] for k in ('online', 'offline'))
    eligible = online['arrivals'] > 0 and offline['arrivals'] == 120
    unfinished = sum(p['unfinished'] for p in populations.values())
    return dict(populations=populations, unfinished=unfinished,
                completed=sum(p['completed'] for p in populations.values()),
                late=sum(p['late'] for p in populations.values()),
                P1plus=eligible and all(p['not_on_time'] == 0 for p in populations.values()),
                historical_P1=eligible and unfinished == 0 and offline['on_time'] == 120
                and online['on_time'] * 100 >= 99 * online['arrivals'],
                post_horizon_completions=post_horizon, horizon_seconds=horizon)


def cost_breakdown(rows, request_metrics, *, occupied_seconds, starts, shutdowns, api_ids, prices):
    by_id = {r['request_id']: r for r in rows}
    api_ids = list(api_ids)
    if len(set(api_ids)) != len(api_ids) or not set(api_ids) <= set(by_id):
        raise ValueError('invalid API acceptance identities')
    if any(by_id[rid]['job_type'] != 'online' for rid in api_ids):
        raise ValueError('offline requests cannot use the registered synthetic API')
    if not math.isfinite(occupied_seconds) or occupied_seconds < 0:
        raise ValueError('invalid occupied GPU seconds')
    if any(type(n) is not int or n < 0 for n in (starts, shutdowns)):
        raise ValueError('event counts must be nonnegative integers')
    keys = ('gpu_second', 'startup_event', 'shutdown_event', 'api_input_token',
            'api_output_token', 'offline_deadline_miss')
    if any(not math.isfinite(prices[k]) or prices[k] < 0 for k in keys):
        raise ValueError('invalid prices')
    components = dict(gpu=occupied_seconds * prices['gpu_second'],
                      startup=starts * prices['startup_event'],
                      shutdown=shutdowns * prices['shutdown_event'],
                      synthetic_api=sum(by_id[i]['input_tokens'] * prices['api_input_token']
                                        + by_id[i]['max_output_tokens'] * prices['api_output_token']
                                        for i in api_ids))
    operating = sum(components.values())
    penalty = request_metrics['populations']['offline']['not_on_time'] * prices['offline_deadline_miss']
    return dict(**components, operating=operating, offline_miss=penalty, total=operating + penalty,
                units='scenario_USD_not_real_bill', scope='registered_observation_window')
