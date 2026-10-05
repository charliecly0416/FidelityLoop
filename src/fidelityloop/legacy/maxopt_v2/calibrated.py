"""CPU-only closed-loop predictor for the accepted C2/ready contract.

One-second decisions/admission; continuous service completions. Homogeneous
N4c batch medians approximate mixed service, not token-level vLLM execution.
The original N3 simulator and its evidence are intentionally untouched.
"""
import math
from collections import deque

from .engine import TargetPolicy
from .workload import FIELDS


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def service_seconds(model, row, concurrency):
    """Time to process one unit of progress at current local concurrency."""
    inp, out = row['input_tokens'], row['max_output_tokens']
    if not 1 <= concurrency <= 4:
        raise ValueError('local concurrency outside accepted 1..4')
    if model['kind'] == 'original_equal_share':
        return concurrency * (inp / 1024.0 + out / 128.0)
    if model['kind'] != 'n4c_progress_lookup' or not 128 <= inp <= 2048 or out not in (32, 128, 256):
        raise ValueError('service model extrapolation prohibited')
    lookup = model['lookup_wall_seconds']

    def at(c):
        if c == 4 and inp == 256 and out == 256:
            return lookup['core_i256_o256_c4']
        left, right = (128, 512) if inp <= 512 else (512, 2048)
        a = lookup[f'grid_i{left}_o{out}_c{c}']
        b = lookup[f'grid_i{right}_o{out}_c{c}']
        return a + (b - a) * (inp - left) / (right - left)

    return at(1) + (at(4) - at(1)) * (concurrency - 1) / 3


def validate_rows(rows, window, allow_locked_test):
    ids, membership = set(), set()
    previous = -1
    for r in rows:
        if set(r) != FIELDS:
            raise ValueError('unknown/missing workload fields')
        if r['split'] not in ('train', 'validation') and not (allow_locked_test and r['split'] == 'locked_test'):
            raise ValueError('locked-test access requires explicit prediction preparation')
        membership.add((r['split'], r['window_id']))
        if len(membership) > 1 or r['request_id'] in ids:
            raise ValueError('mixed window or duplicate request')
        ids.add(r['request_id'])
        if (type(r['arrival_s']) is not int or not previous <= r['arrival_s'] < window
                or r['sim_tick'] != r['arrival_s'] or r['wall_monotonic_s'] is not None
                or r['job_type'] not in ('online', 'offline')
                or type(r['deadline_s']) is not int or r['deadline_s'] < r['arrival_s']
                or len(r['prompt_token_ids']) != r['input_tokens']
                or type(r['max_output_tokens']) is not int or r['max_output_tokens'] <= 0):
            raise ValueError('invalid workload row')
        previous = r['arrival_s']


def metrics(rows, records, occupied, busy, active, queue_area, starts, shutdowns, api_ids, contract):
    prices, limits = contract['accounting'], contract['feasibility']
    populations = {}
    for kind in ('online', 'offline'):
        subset = [r for r in rows if r['job_type'] == kind]
        on_time = sum(records[r['request_id']]['completed_at'] is not None
                      and records[r['request_id']]['completed_at'] <= (
                          r['arrival_s'] + limits['online_e2e_sla_seconds'] if kind == 'online' else r['deadline_s'])
                      for r in subset)
        populations[kind] = dict(arrivals=len(subset), on_time=on_time,
                                 not_on_time=len(subset)-on_time,
                                 on_time_rate=on_time/len(subset) if subset else None)
    by_id = {r['request_id']: r for r in rows}
    api_cost = sum(by_id[r]['input_tokens'] * prices['api_input_token']
                   + by_id[r]['max_output_tokens'] * prices['api_output_token'] for r in api_ids)
    operating = occupied * prices['gpu_second'] + starts * prices['startup_event'] + shutdowns * prices['shutdown_event'] + api_cost
    penalty = populations['offline']['not_on_time'] * prices['offline_deadline_miss']
    local_times = [r['completed_at']-r['dispatched_at'] for r in records.values()
                   if r['completed_at'] is not None and isinstance(r['route'], int)]
    latencies = [records[r['request_id']]['completed_at']-r['arrival_s'] for r in rows
                 if records[r['request_id']]['completed_at'] is not None]
    return dict(populations=populations,
                feasible=(populations['online']['arrivals'] > 0 and populations['offline']['arrivals'] > 0
                          and populations['online']['not_on_time'] <= limits['max_online_violation_rate'] * populations['online']['arrivals']
                          and populations['offline']['on_time'] >= limits['min_offline_deadline_completion_rate'] * populations['offline']['arrivals']),
                completed=sum(r['completed_at'] is not None for r in records.values()),
                censored=sum(r['completed_at'] is None for r in records.values()),
                api_accepted=len(api_ids), gpu_occupied_seconds=occupied, gpu_active_seconds=active,
                service_busy_seconds=busy, queue_seconds=queue_area,
                startup_events=starts, shutdown_events=shutdowns,
                costs=dict(gpu=occupied*prices['gpu_second'], startup=starts*prices['startup_event'],
                           shutdown=shutdowns*prices['shutdown_event'], synthetic_api=api_cost,
                           offline_miss=penalty, operating=operating, total=operating+penalty),
                local_service_mean_seconds=sum(local_times)/len(local_times) if local_times else None,
                e2e_p50_seconds=quantile(latencies, .5), e2e_p90_seconds=quantile(latencies, .9),
                e2e_max_seconds=max(latencies) if latencies else None)


class CalibratedSimulator:
    def __init__(self, rows, contract, model, policy, *, allow_locked_test=False):
        self.window = contract['workload_revision']['window_seconds']
        self.end = self.window + contract['workload_revision']['drain_seconds']
        validate_rows(rows, self.window, allow_locked_test)
        for row in rows:
            service_seconds(model, row, 1)  # fail on unsupported shape before running
        for field in ('startup_seconds', 'shutdown_seconds', 'setup_seconds'):
            if not math.isfinite(model[field]) or model[field] <= 0:
                raise ValueError('invalid lifecycle duration')
        if contract['synthetic_api'] != dict(capacity=8, latency_seconds=5,
                route_online_after_wait_seconds=10, offline_route=False, cost_on_accept=True,
                completion_requires_elapsed_delay=True, virtual_settlement_counts_as_completion=False):
            raise ValueError('accepted API contract changed')
        self.rows, self.contract, self.model = rows, contract, model
        self.policy = TargetPolicy(policy)
        self.policy.state['target'] = 2
        self.devices = [dict(state='active', jobs={}, due=None) for _ in range(2)]
        self.jobs = {r['request_id']: dict(row=r, dispatched_at=None, completed_at=None, route=None) for r in rows}
        self.queues = {k: deque() for k in ('online', 'offline')}
        self.api, self.api_ids, self.events = {}, [], []
        self.now, self.cursor = 0.0, 0
        self.occupied = self.busy = self.active = self.queue_area = 0.0
        self.starts = self.shutdowns = 0
        self.emit('initial', states=['active', 'active'], queues_empty=True)

    def emit(self, kind, **fields):
        self.events.append(dict(seq=len(self.events), time=self.now, kind=kind, **fields))

    def state(self, gpu, new, due=None):
        device = self.devices[gpu]
        self.emit('state', gpu=gpu, previous=device['state'], state=new, due=due)
        device.update(state=new, due=due)

    def stop(self, gpu):
        assert not self.devices[gpu]['jobs']
        self.shutdowns += 1
        self.state(gpu, 'stopping', self.now + self.model['shutdown_seconds'])

    def actuate(self, target):
        def count():
            return sum(d['state'] in ('active', 'starting') for d in self.devices)
        for gpu in (1, 0):
            if count() > target and self.devices[gpu]['state'] == 'active':
                self.state(gpu, 'draining')
                if not self.devices[gpu]['jobs']:
                    self.stop(gpu)
        for gpu, d in enumerate(self.devices):
            if count() < target and d['state'] == 'draining':
                self.state(gpu, 'active')
        for gpu, d in enumerate(self.devices):
            if count() < target and d['state'] == 'off':
                self.starts += 1
                self.state(gpu, 'starting', self.now + self.model['startup_seconds'])

    def dispatch_local(self, kind):
        queue = self.queues[kind]
        while queue:
            available = [g for g, d in enumerate(self.devices) if d['state'] == 'active' and len(d['jobs']) < 4]
            if not available:
                break
            gpu = min(available, key=lambda g: (len(self.devices[g]['jobs']), g))
            rid = queue.popleft()
            self.devices[gpu]['jobs'][rid] = 1.0
            self.jobs[rid].update(dispatched_at=self.now, route=gpu)
            self.emit('dispatch', request_id=rid, route=gpu)

    def tick(self, tick):
        while self.cursor < len(self.rows) and self.rows[self.cursor]['arrival_s'] <= tick:
            row = self.rows[self.cursor]
            self.queues[row['job_type']].append(row['request_id'])
            self.emit('release', request_id=row['request_id'])
            self.cursor += 1
        observation = dict(time=tick, queue_online=len(self.queues['online']),
                           queue_offline=len(self.queues['offline']),
                           local_running=sum(len(d['jobs']) for d in self.devices),
                           **{s: sum(d['state'] == s for d in self.devices) for s in ('active', 'starting', 'off')},
                           draining=sum(d['state'] in ('draining', 'stopping') for d in self.devices))
        target = self.policy.decide(observation)
        self.emit('decision', target=target, observation=observation)
        self.actuate(target)
        self.dispatch_local('online')
        queue = self.queues['online']
        while queue and len(self.api) < 8 and self.now - self.jobs[queue[0]]['row']['arrival_s'] >= 10:
            rid = queue.popleft()
            self.jobs[rid].update(dispatched_at=self.now, route='synthetic_api')
            self.api[rid] = self.now + 5
            self.api_ids.append(rid)
            self.emit('dispatch', request_id=rid, route='synthetic_api')
        self.dispatch_local('offline')

    def complete(self, rid):
        self.jobs[rid]['completed_at'] = self.now
        self.emit('complete', request_id=rid, route=self.jobs[rid]['route'])

    def advance(self, until):
        while self.now < until:
            rates = {rid: 1 / service_seconds(self.model, self.jobs[rid]['row'], len(d['jobs']))
                     for d in self.devices for rid in d['jobs']}
            due = [until, *self.api.values()]
            due.extend(d['due'] for d in self.devices if d['due'] is not None)
            due.extend(self.now + left/rates[rid] for d in self.devices for rid, left in d['jobs'].items())
            next_time = min(due)
            elapsed = next_time - self.now
            if elapsed < -1e-9:
                raise ValueError('backward model clock')
            self.occupied += elapsed * sum(d['state'] != 'off' for d in self.devices)
            self.active += elapsed * sum(d['state'] == 'active' for d in self.devices)
            self.busy += elapsed * sum(bool(d['jobs']) for d in self.devices)
            self.queue_area += elapsed * sum(map(len, self.queues.values()))
            for d in self.devices:
                for rid in d['jobs']:
                    d['jobs'][rid] -= elapsed * rates[rid]
            self.now = next_time
            for gpu, d in enumerate(self.devices):
                for rid in list(d['jobs']):
                    if d['jobs'][rid] <= 1e-10:
                        del d['jobs'][rid]
                        self.complete(rid)
                if d['state'] == 'draining' and not d['jobs']:
                    self.stop(gpu)
                if d['due'] is not None and d['due'] <= self.now:
                    self.state(gpu, 'active' if d['state'] == 'starting' else 'off')
            for rid in list(self.api):
                if self.api[rid] <= self.now:
                    del self.api[rid]
                    self.complete(rid)

    def run(self):
        for tick in range(self.end):
            self.advance(tick)
            self.tick(tick)
        self.advance(self.end)
        self.emit('end')
        records = {rid: {k: j[k] for k in ('dispatched_at', 'completed_at', 'route')} for rid, j in self.jobs.items()}
        result = metrics(self.rows, records, self.occupied, self.busy, self.active, self.queue_area,
                         self.starts, self.shutdowns, self.api_ids, self.contract)
        # Window cost is the registered ranking metric. Common setup and terminal
        # cleanup are disclosed separately, including residual in-flight release.
        prices = self.contract['accounting']
        cleanup_seconds = sum((d['due']-self.end if d['state'] == 'stopping' else self.model['shutdown_seconds'])
                              for d in self.devices if d['state'] != 'off')
        cleanup_events = sum(d['state'] not in ('off', 'stopping') for d in self.devices)
        setup_cost = 2 * self.model['setup_seconds'] * prices['gpu_second'] + 2 * prices['startup_event']
        cleanup_cost = cleanup_seconds * prices['gpu_second'] + cleanup_events * prices['shutdown_event']
        result['outside_window'] = dict(setup_proxy_cost=setup_cost, cleanup_proxy_cost=cleanup_cost,
                                       cleanup_gpu_seconds=cleanup_seconds,
                                       deployment_total_with_penalty=result['costs']['total']+setup_cost+cleanup_cost)
        return dict(summary=result, requests=records, events=self.events)
