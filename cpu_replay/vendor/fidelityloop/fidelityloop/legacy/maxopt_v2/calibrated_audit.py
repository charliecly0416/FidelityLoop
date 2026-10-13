"""Event-only conservation/admission/lifecycle/cost check; no service simulator."""
import math
from .engine import TargetPolicy


def audit(rows, events, summary, contract, model, policy=None, requests=None):
    by_id = {r['request_id']: r for r in rows}
    released, waiting, running, completed, dispatched = set(), [], {}, {}, {}
    states, due = ['active', 'active'], [None, None]
    starts = stops = ticks = 0
    occupied = active = busy = queue_area = last = 0.0
    end = sum(contract['workload_revision'][k] for k in ('window_seconds', 'drain_seconds'))
    api_ids = []
    checks = 0
    controller = TargetPolicy(policy) if policy is not None else None
    if controller is not None:
        controller.state['target'] = 2

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise ValueError(message)

    def local_available():
        counts = {g: sum(v == g for v in running.values()) for g in (0, 1) if states[g] == 'active'}
        return sorted((g for g in counts if counts[g] < 4), key=lambda g: (counts[g], g))

    for seq, e in enumerate(events):
        now = e['time']
        require(e['seq'] == seq and math.isfinite(now) and last <= now <= end, 'event sequence/clock')
        dt = now-last
        occupied += dt * sum(s != 'off' for s in states)
        active += dt * states.count('active')
        busy += dt * sum(g in running.values() for g in (0, 1))
        queue_area += dt * len(waiting)
        last = now
        kind = e['kind']
        if kind == 'initial':
            require(seq == 0 and now == 0 and e['states'] == states and e['queues_empty'], 'initial state')
        elif kind == 'release':
            rid = e['request_id']
            require(rid in by_id and rid not in released and now == by_id[rid]['arrival_s'], 'release')
            released.add(rid)
            waiting.append(rid)
        elif kind == 'decision':
            require(now == ticks and type(e['target']) is int and 0 <= e['target'] <= 2, 'policy tick/target')
            expected = dict(time=ticks,
                            queue_online=sum(by_id[r]['job_type'] == 'online' for r in waiting),
                            queue_offline=sum(by_id[r]['job_type'] == 'offline' for r in waiting),
                            local_running=sum(isinstance(g, int) for g in running.values()),
                            active=states.count('active'), starting=states.count('starting'),
                            draining=states.count('draining')+states.count('stopping'), off=states.count('off'))
            require(e['observation'] == expected, 'noncausal/incorrect policy observation')
            if controller is not None:
                require(e['target'] == controller.decide(expected), 'frozen policy decision')
            ticks += 1
        elif kind == 'state':
            gpu, old, new = e['gpu'], e['previous'], e['state']
            require(old == states[gpu], 'state continuity')
            require((old, new) in {('active','draining'), ('draining','active'), ('draining','stopping'),
                                    ('stopping','off'), ('off','starting'), ('starting','active')}, 'state transition')
            if new in ('stopping', 'off', 'starting'):
                require(gpu not in running.values(), 'release/start with pending local request')
            if new in ('starting', 'stopping'):
                duration = model['startup_seconds' if new == 'starting' else 'shutdown_seconds']
                require(math.isclose(e['due'], now+duration, abs_tol=1e-8), 'transition duration')
                starts += new == 'starting'
                stops += new == 'stopping'
            elif old in ('starting', 'stopping'):
                require(math.isclose(now, due[gpu], abs_tol=1e-8), 'early ready/release')
            states[gpu], due[gpu] = new, e['due']
        elif kind == 'dispatch':
            rid, route = e['request_id'], e['route']
            require(rid in waiting and rid not in dispatched and now < end and float(now).is_integer(), 'dispatch identity/time')
            row = by_id[rid]
            same = [r for r in waiting if by_id[r]['job_type'] == row['job_type']]
            require(same[0] == rid, 'FCFS admission')
            available = local_available()
            if route == 'synthetic_api':
                require(row['job_type'] == 'online' and now-row['arrival_s'] >= 10 and not available,
                        'API eligibility/local priority')
                require(list(running.values()).count('synthetic_api') < 8, 'API capacity')
                api_ids.append(rid)
            else:
                require(route in (0,1) and available and route == available[0], 'local slot/tie-break')
                if row['job_type'] == 'offline':
                    require(not any(by_id[r]['job_type'] == 'online' for r in waiting), 'online first')
            waiting.remove(rid)
            dispatched[rid] = dict(time=now, route=route)
            running[rid] = route
        elif kind == 'complete':
            rid = e['request_id']
            require(rid in running and rid not in completed and running[rid] == e['route'], 'completion conservation')
            if running[rid] == 'synthetic_api':
                require(math.isclose(now-dispatched[rid]['time'], 5, abs_tol=1e-8), 'API elapsed service')
            completed[rid] = now
            del running[rid]
        elif kind == 'end':
            require(now == end and seq == len(events)-1, 'window end')
        else:
            raise ValueError('unknown event')
    require(events[0]['kind'] == 'initial' and events[-1]['kind'] == 'end', 'event boundaries')
    require(released == set(by_id) and ticks == end, 'all-arrivals denominator/tick budget')
    require(set(waiting) | set(running) | set(completed) == released, 'request conservation')
    if requests is not None:
        projection = {rid:dict(dispatched_at=dispatched[rid]['time'] if rid in dispatched else None,
                               route=dispatched[rid]['route'] if rid in dispatched else None,
                               completed_at=completed.get(rid)) for rid in by_id}
        require(requests == projection, 'request prediction/event projection')
    populations = {}
    for kind in ('online', 'offline'):
        group = [r for r in rows if r['job_type'] == kind]
        timely = sum(r['request_id'] in completed and completed[r['request_id']] <= (
            r['arrival_s']+contract['feasibility']['online_e2e_sla_seconds'] if kind == 'online' else r['deadline_s']) for r in group)
        populations[kind] = dict(arrivals=len(group), on_time=timely, not_on_time=len(group)-timely,
                                 on_time_rate=timely/len(group) if group else None)
    require(summary['populations'] == populations, 'SLO counts')
    require(summary['completed'] == len(completed) and summary['censored'] == len(rows)-len(completed), 'terminal summary')
    p = contract['accounting']
    costs = dict(gpu=occupied*p['gpu_second'], startup=starts*p['startup_event'], shutdown=stops*p['shutdown_event'],
                 synthetic_api=sum(by_id[r]['input_tokens']*p['api_input_token']+by_id[r]['max_output_tokens']*p['api_output_token'] for r in api_ids),
                 offline_miss=populations['offline']['not_on_time']*p['offline_deadline_miss'])
    costs['operating'] = sum(costs[k] for k in ('gpu','startup','shutdown','synthetic_api'))
    costs['total'] = costs['operating']+costs['offline_miss']
    for k, v in costs.items():
        require(math.isclose(summary['costs'][k], v, abs_tol=1e-8), 'ledger cost '+k)
    for k, v in dict(gpu_occupied_seconds=occupied, gpu_active_seconds=active, service_busy_seconds=busy,
                     queue_seconds=queue_area, startup_events=starts, shutdown_events=stops, api_accepted=len(api_ids)).items():
        require(math.isclose(summary[k], v, abs_tol=1e-7), 'ledger summary '+k)
    f = contract['feasibility']
    feasible = (all(populations[k]['arrivals'] for k in populations)
                and populations['online']['not_on_time'] <= f['max_online_violation_rate']*populations['online']['arrivals']
                and populations['offline']['on_time'] >= f['min_offline_deadline_completion_rate']*populations['offline']['arrivals'])
    require(summary['feasible'] == bool(feasible), 'feasibility classification')
    return dict(status='PASS', checks=checks, request_count=len(rows), event_count=len(events),
                gpu_occupied_seconds=occupied, costs=costs, feasible=bool(feasible))
