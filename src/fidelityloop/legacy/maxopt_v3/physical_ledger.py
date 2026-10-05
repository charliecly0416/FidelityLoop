"""N5 request, dynamic lifecycle, and causal policy accounting; no GPU imports."""
from collections import defaultdict
import math

from . import runtime_adapter  # establish isolated accepted-runtime imports
from scripts.maxopt_stage2_formal.ledger import reduce_run as request_ledger
from .policy import DeadlinePolicy, SCHEMA, public_job

STATES = {'off', 'starting', 'active', 'draining', 'stopping'}
TRANSITIONS = {('off', 'starting'), ('starting', 'active'), ('active', 'draining'),
               ('draining', 'active'), ('draining', 'stopping'), ('stopping', 'off')}
TERMINALS = {'finished', 'failed', 'cancelled'}


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def overlap(start, stop, left, right):
    return max(0, min(stop, right) - max(start, left)) / 1e9


def reduce_requests(expected, events, observed_seconds, *, run_errors=()):
    result = request_ledger(expected, events, observed_seconds, run_errors=run_errors)
    result['schema'] = 'maxopt-n5-request-ledger-v1'
    for kind in ('online', 'offline'):
        latencies = [r['latency_s'] for r in result['requests'] if r['job_type'] == kind and r['latency_s'] is not None]
        result[kind]['latency_seconds'] = dict(completed_samples=len(latencies),
                                              **{label: percentile(latencies, value)
                                                 for label, value in [('p50', .5), ('p95', .95), ('p99', .99)]})
    assignments = {row['request_id']: row for row in events if row['kind'] == 'dispatch'}
    for row in result['requests']:
        assignment = assignments.get(row['request_id'], {})
        row.update({key: assignment.get(key) for key in ('gpu', 'generation', 'pid')})
        row['queue_seconds'] = (row['submission_s'] - row['arrival_s']) if row['submission_s'] is not None else None
        row['local_service_seconds'] = (row['completion_s'] - row['submission_s']
                                        if row['route'] in ('gpu0', 'gpu1') and row['completion_s'] is not None else None)
    result['ttft'] = dict(measured=False, reason='no per-token first-token observation')
    result['scope'] = 'request projection; enclosing N5 verifier binds RAW, generations, policy, and actual release'
    return result


def reduce_lifecycle(expected, controller, requests, *, origin_ns, cutoff_ns, policy=None, require_ticks=True, guard=None):
    """Replay receipts before each decision; charge occupation until verified release.

    The supplied policy is a fresh instance of the frozen CPU TargetPolicy. Its
    common initial target is 2; no real future completion enters decide().
    """
    errors = []
    def check(value, name, detail=None):
        if not value:
            errors.append(dict(reason=name, detail=detail))
        return bool(value)
    by_id = {row['request_id']: row for row in expected}
    projections = defaultdict(list)
    for event in requests:
        projections[event['raw_controller_seq']].append(event)
    devices = {gpu: dict(state='off', generation=0, pid=None, occupied=False, jobs=set()) for gpu in (0, 1)}
    generations, running, released = {}, {}, set()
    queues = {'online': [], 'offline': []}
    api = set()
    accounting = {phase: dict(gpu_occupied_seconds=0.0, gpu_active_seconds=0.0, service_busy_seconds=0.0,
                              queue_seconds=0.0, startup_events=0, shutdown_events=0) for phase in ('setup', 'window', 'cleanup')}
    first_ns = controller[0]['controller_monotonic_ns'] if controller else origin_ns
    last_ns = controller[-1]['controller_monotonic_ns'] if controller else cutoff_ns
    windows = dict(setup=(first_ns, origin_ns), window=(origin_ns, cutoff_ns), cleanup=(cutoff_ns, last_ns))
    ticks, admission_ticks, target = [], [], 2
    pending_actuation = []
    since = {0: 0, 1: 0}
    dispatch_times = {}
    decision = None
    def causal_observation(tick):
        return dict(schema=SCHEMA,time=tick,
                    **{'queue_'+kind:[public_job(by_id[r]) for r in queues[kind]] for kind in ('online','offline')},
                    devices=[dict(state=devices[g]['state'],since=(since[g]-origin_ns)/1e9,
                                  jobs=[public_job(by_id[r],(t-origin_ns)/1e9) for r,t in dispatch_times.items() if r in devices[g]['jobs']])
                             for g in (0,1)])
    if policy is not None:
        policy.state['target'] = 2
    previous = first_ns
    observed_first_service = set()
    forced_cleanup = set()
    for index, event in enumerate(controller):
        now = event['controller_monotonic_ns']
        check(type(now) is int and now >= previous and event.get('seq') == index, 'controller_sequence_clock', index)
        for phase, (left, right) in windows.items():
            dt = overlap(previous, now, left, right)
            book = accounting[phase]
            book['gpu_occupied_seconds'] += dt * sum(d['occupied'] for d in devices.values())
            book['gpu_active_seconds'] += dt * sum(d['state'] == 'active' for d in devices.values())
            book['service_busy_seconds'] += dt * sum(bool(d['jobs']) for d in devices.values())
            book['queue_seconds'] += dt * sum(len(q) for q in queues.values())
        previous = now
        phase = 'setup' if now < origin_ns else 'window' if now < cutoff_ns else 'cleanup'
        kind = event['kind']
        gpu, generation = event.get('gpu'), event.get('generation')
        device = devices.get(gpu)
        key = (gpu, generation)
        if kind == 'observation_start':
            check(event.get('queues_empty') is True and all(d['state'] == 'active' and not d['jobs']
                  and d['generation'] == 1 for d in devices.values()), 'common_two_ready_empty_initial_state')
        elif kind == 'launch_issued':
            check(now < cutoff_ns, 'no_new_launch_during_terminal_cleanup', key)
            check(gpu in devices and type(generation) is int and generation == device['generation'] + 1
                  and not device['occupied'] and device['state'] in ('off', 'starting'), 'nonoverlapping_generation_launch', key)
            device.update(generation=generation, occupied=True, pid=None)
            generations[key] = dict(gpu=gpu, generation=generation, launch_ns=now, pid=None,
                                    ready_ns=None, shutdown_ns=None, release_ns=None)
            accounting[phase]['startup_events'] += 1
        elif kind == 'worker_spawned':
            check(key in generations and device['occupied'] and generation == device['generation']
                  and generations[key]['pid'] is None and type(event.get('pid')) is int, 'generation_spawn_identity', key)
            if key in generations:
                generations[key]['pid'] = event['pid']
                device['pid'] = event['pid']
        elif kind == 'lifecycle_state':
            old, new = event['previous'], event['state']
            terminal_cleanup = key in forced_cleanup and now >= cutoff_ns and event.get('phase') == 'cleanup'
            legal = (old, new) in TRANSITIONS or (terminal_cleanup and new == 'stopping' and old in ('starting', 'active'))
            check(gpu in devices and old == device['state'] and legal, 'legal_lifecycle_transition', (gpu, old, new))
            if require_ticks and origin_ns <= now < cutoff_ns:
                transition = (gpu, old, new)
                if pending_actuation:
                    check(transition == pending_actuation.pop(0), 'frozen_target_actuation_order', transition)
                else:
                    check((old, new) in {('starting', 'active'), ('stopping', 'off'), ('draining', 'stopping')},
                          'no_extra_target_action_between_ticks', transition)
            if new in ('starting', 'stopping', 'off') and not (terminal_cleanup and new == 'stopping'):
                check(not device['jobs'], 'transition_with_running_requests', (gpu, old, new))
            if old == 'starting' and new == 'active':
                check(key in generations and generations[key]['ready_ns'] is not None, 'active_requires_health_ready', key)
            if old == 'stopping' and new == 'off':
                check(key in generations and generations[key]['release_ns'] is not None, 'off_requires_verified_release', key)
            if old == 'draining' and new == 'active':
                check(key in generations and generations[key]['shutdown_ns'] is None, 'cannot_cancel_issued_shutdown', key)
            device['state'] = new
            since[gpu] = now
        elif kind == 'ready':
            check(key in generations and generations[key]['ready_ns'] is None
                  and device['state'] == 'starting' and event.get('pid') == device['pid'], 'unique_ready_for_starting_generation', key)
            if key in generations:
                generations[key]['ready_ns'] = now
        elif kind == 'shutdown_issued':
            forced = (event.get('forced') is True and now >= cutoff_ns and event.get('phase') == 'cleanup'
                      and bool(event.get('reason')))
            if forced:
                forced_cleanup.add(key)
            check(key in generations and generations[key]['shutdown_ns'] is None
                  and ((device['state'] in ('draining', 'stopping') and not device['jobs']) or forced), 'shutdown_after_drain_once', key)
            if key in generations:
                generations[key]['shutdown_ns'] = now
            accounting[phase]['shutdown_events'] += 1
        elif kind == 'verified_release':
            check(key in generations and generations[key]['shutdown_ns'] is not None
                  and generations[key]['release_ns'] is None and device['state'] == 'stopping'
                  and (not device['jobs'] or key in forced_cleanup) and event.get('pid') == device['pid'], 'verified_release_once_after_shutdown', key)
            if key in generations:
                generations[key]['release_ns'] = now
            device['occupied'] = False
            if key in forced_cleanup:
                for rid in list(device['jobs']):
                    running.pop(rid, None)
                device['jobs'].clear()
        elif kind == 'worker_receipt':
            worker = event['worker_event']
            rid = worker.get('request_id')
            if worker['kind'] in TERMINALS and rid in running:
                assigned = running.pop(rid)
                check(assigned == key, 'terminal_generation_route', rid)
                devices[assigned[0]]['jobs'].discard(rid)
        elif kind == 'api_completed':
            rid = event['request_id']
            check(rid in api, 'unique_api_completion', rid)
            api.discard(rid)
        elif kind == 'first_service':
            check(key not in observed_first_service and key in generations and generations[key]['ready_ns'] is not None,
                  'first_service_identity', key)
            observed_first_service.add(key)
        elif kind == 'policy_tick':
            tick = event['tick']
            check(type(tick) is int and tick == len(ticks)
                  and origin_ns + tick * 10**9 <= now < min(cutoff_ns, origin_ns + (tick + 1) * 10**9),
                  'one_causal_policy_tick_per_second', tick)
            expected_observation = dict(time=tick, queue_online=len(queues['online']), queue_offline=len(queues['offline']),
                                        local_running=len(running), active=sum(d['state'] == 'active' for d in devices.values()),
                                        starting=sum(d['state'] == 'starting' for d in devices.values()),
                                        draining=sum(d['state'] in ('draining', 'stopping') for d in devices.values()),
                                        off=sum(d['state'] == 'off' for d in devices.values()))
            if guard is not None:
                expected_observation = causal_observation(tick)
            check(event['observation'] == expected_observation, 'policy_observation_from_current_RAW_only', tick)
            check(all(row['request_id'] in released for row in expected if row['arrival_s'] <= tick), 'arrivals_processed_before_policy', tick)
            if guard is not None:
                decision=guard.decide(expected_observation)
                check(event.get('guard')==decision and event['target']==decision['target'], 'V3_guard_state_replay', tick)
                check(event.get('policy_state_before')==decision['policy_state_before'] and event.get('policy_state_after')==decision['policy_state_after'], 'V3_policy_state_projection', tick)
            elif policy is not None:
                check(event.get('policy_state_before') == policy.state, 'policy_state_before', tick)
                computed = policy.decide(expected_observation)
                check(event.get('target') == computed and event.get('policy_state_after') == policy.state, 'frozen_TargetPolicy_replay', tick)
            target = event['target']
            check(type(target) is int and 0 <= target <= 2, 'target_range', tick)
            check(not pending_actuation, 'previous_tick_actuation_completed', tick)
            planned_states = {g: d['state'] for g, d in devices.items()}
            def change(g, state):
                pending_actuation.append((g, planned_states[g], state))
                planned_states[g] = state
            def capacity():
                return sum(state in ('active', 'starting') for state in planned_states.values())
            for g in (1, 0):
                if capacity() > target and planned_states[g] == 'active':
                    change(g, 'draining')
                    if not devices[g]['jobs']:
                        change(g, 'stopping')
            for g in (0, 1):
                if capacity() < target and planned_states[g] == 'draining':
                    change(g, 'active')
            for g in (0, 1):
                if capacity() < target and planned_states[g] == 'off':
                    change(g, 'starting')
            ticks.append(tick)
        elif kind == 'admission_complete':
            tick = event['tick']
            check(ticks and tick == ticks[-1] and tick not in admission_ticks, 'admission_tick_identity', tick)
            check(not pending_actuation, 'target_actions_before_admission', tick)
            slots = any(d['state'] == 'active' and len(d['jobs']) < 4 for d in devices.values())
            check(not (slots and (queues['online'] or queues['offline'])), 'eligible_local_not_silently_held', tick)
            check(not (queues['online'] and len(api) < 8 and (now-origin_ns)/1e9 - by_id[queues['online'][0]]['arrival_s'] >= 10),
                  'eligible_api_not_silently_held', tick)
            admission_ticks.append(tick)
        for projection in projections.get(index, []):
            rid = projection['request_id']
            if rid not in by_id:
                check(False, 'unplanned_projection', rid)
                continue
            row = by_id[rid]
            if projection['kind'] == 'release':
                check(rid not in released and now >= origin_ns + row['arrival_s'] * 10**9, 'release_identity_clock', rid)
                released.add(rid)
                queues[row['job_type']].append(rid)
            elif projection['kind'] == 'dispatch':
                queue = queues[row['job_type']]
                check(queue and queue[0] == rid and now < cutoff_ns, 'FCFS_dispatch_before_cutoff', rid)
                if guard is not None and projection['route']!='synthetic_api':
                    check(row['job_type']==guard.dispatch_kind(causal_observation(ticks[-1]),decision['reserve']), 'V3_reservation_admission', rid)
                if rid in queue:
                    queue.remove(rid)
                available = sorted((g for g, d in devices.items() if d['state'] == 'active' and len(d['jobs']) < 4),
                                   key=lambda g: (len(devices[g]['jobs']), g))
                route = projection['route']
                if route == 'synthetic_api':
                    check(row['job_type'] == 'online' and not available and len(api) < 8
                          and projection['at_s'] - row['arrival_s'] >= 10, 'API_after_local_and_wait_with_capacity', rid)
                    api.add(rid)
                else:
                    g = projection['gpu']
                    check(available and available[0] == g and route == 'gpu' + str(g), 'local_active_least_inflight_gpu_tie', rid)
                    check(projection['generation'] == devices[g]['generation'] and projection['pid'] == devices[g]['pid'],
                          'dispatch_current_generation', rid)
                    if row['job_type'] == 'offline' and guard is None:
                        check(not queues['online'], 'online_before_offline', rid)
                    dispatch_times[rid]=now
                    running[rid] = (g, projection['generation'])
                    devices[g]['jobs'].add(rid)
            elif projection['kind'] == 'terminal' and projection['status'] == 'rejected':
                check(False, 'unplanned_admission_rejection', rid)
    check(released == set(by_id), 'all_arrivals_released')
    if require_ticks:
        check(ticks == list(range(2100)) and admission_ticks == ticks, 'full_2100_policy_admission_ticks')
    check(all(not d['occupied'] and d['state'] == 'off' and not d['jobs'] for d in devices.values()), 'all_generations_released_after_cleanup')
    for key, item in generations.items():
        required = ('pid', 'shutdown_ns', 'release_ns') if key in forced_cleanup else ('pid', 'ready_ns', 'shutdown_ns', 'release_ns')
        check(all(item[name] is not None for name in required),
              'complete_generation_lifecycle', key)
        item['never_ready'] = item['ready_ns'] is None
        item['forced_terminal_cleanup'] = key in forced_cleanup
    for phase, book in accounting.items():
        book['gpu_cost'] = book['gpu_occupied_seconds'] * .001
        book['startup_cost'] = book['startup_events'] * .002
        book['shutdown_cost'] = book['shutdown_events'] * .001
        book['holding_and_transition_cost'] = book['gpu_cost'] + book['startup_cost'] + book['shutdown_cost']
    return dict(schema='maxopt-n5-dynamic-lifecycle-ledger-v1', status='TECHNICAL_INVALID' if errors else 'PASS',
                technical_errors=errors, phases=accounting, generations=list(generations.values()),
                policy_ticks=len(ticks), admission_ticks=len(admission_ticks), all_arrivals_denominator=len(by_id),
                endpoint='launch_issued through verified_release; all non-off states charged',
                unit='scenario_USD', actual_energy_or_bill_claim=False)
