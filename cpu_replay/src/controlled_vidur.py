"""Qwen CPU control integration derived from the accepted engineering adapter. Native Vidur handles every local service event.

The wrapper owns admission/lifecycle only. Two preallocated, persistent replica
schedulers stand for capacity generations; restart is permitted only when empty.
No model load/cache reset time is inferred from this simulation.
"""
import atexit
import copy
import csv
import heapq
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/vidur"
for key, value in {"CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
    os.environ[key] = value
sys.path[:0] = [str(SOURCE), str(ROOT / "vendor/fidelityloop")]
from vidur.config import (ClusterConfig, MetricsConfig, ReplicaConfig, SarathiSchedulerConfig,
                         SimulationConfig, TraceRequestGeneratorConfig, RandomForrestExecutionTimePredictorConfig)
from vidur.simulator import Simulator
from vidur.entities import Request, Replica
from vidur.events import BaseEvent
from vidur.events.replica_schedule_event import ReplicaScheduleEvent
from vidur.types import EventType
from vidur.utils.random import set_seeds
from fidelityloop.framework.policies import TargetPolicyAdapter
from fidelityloop.legacy.maxopt_v3.policy import DeadlinePolicy, public_job, validate_observation, SCHEMA
from fidelityloop.bridge.metrics import evaluate_requests

H_SPEC = {"name": "hysteresis", "up_threshold": 8, "down_threshold": 0, "cooldown_seconds": 60, "min_devices": 0}
SYNTHETIC_MODEL = {"kind": "original_equal_share"}
SYNTHETIC_SAFETY = {"startup_safe_seconds": 2., "startup_max_seconds": 2., "shutdown_seconds": 1.}


from native_qwen import make_simulator


def make_policy(name, arm=None, model=None, safety=None):
    if name not in ('H','guard','all1','all2'):
        raise ValueError('unknown policy name')
    if arm is not None:
        return DeadlinePolicy(arm['policy'], model, safety,
                              scale=arm['scale'], reservation=arm['reservation'])
    if name == 'guard':
        return DeadlinePolicy(H_SPEC, model or SYNTHETIC_MODEL, safety or SYNTHETIC_SAFETY)
    policy = TargetPolicyAdapter(H_SPEC if name == 'H' else name)
    policy.state['target'] = 2
    return policy


class ControlledVidur:
    def __init__(self, rows, out, policy="H", *, horizon=120, startup=2., shutdown=1., initial="off", targets=None, arm=None, guard_model=None, guard_safety=None):
        if (not rows or len(rows) > 2000 or type(horizon) is not int or horizon > 2100 or horizon <= 0
                or initial not in ('off','active') or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (startup,shutdown))):
            raise ValueError("outside bounded CPU contract")
        for r in rows:
            if set(r) != {'request_id','arrival_s','deadline_s','job_type','input_tokens','max_output_tokens'}:
                raise ValueError('unknown/missing public input fields')
            if (type(r['arrival_s']) is not int or r['arrival_s'] < 0 or type(r['deadline_s']) is not int
                    or r['deadline_s'] < r['arrival_s'] or r['job_type'] not in ('online','offline')
                    or not isinstance(r['request_id'],str) or not r['request_id']
                    or type(r['input_tokens']) is not int or not 32 <= r['input_tokens'] <= 2048
                    or type(r['max_output_tokens']) is not int or not 1 <= r['max_output_tokens'] <= 256):
                raise ValueError('input outside registered Qwen domain')
        if len({r['request_id'] for r in rows}) != len(rows):
            raise ValueError("duplicate input request ID")
        self.rows = sorted(copy.deepcopy(rows), key=lambda r: r['arrival_s'])
        self.policy = make_policy(policy, arm, guard_model, guard_safety)
        self.historical_parameters = arm is not None
        self.sim, self.requests = make_simulator(self.rows, out)
        self.sim._event_queue.clear()  # outer release/admission owns arrivals
        self.policy_name, self.targets = policy, targets
        self.horizon, self.startup, self.shutdown = horizon, startup, shutdown
        self.now, self.cursor = 0., 0
        self.devices = [dict(state=initial, since=0., due=None, generation=0, jobs={}) for _ in range(2)]
        self.queue = {k: [] for k in ('online', 'offline')}
        self.records = {r['request_id']: dict(r, dispatched_at=None, completed_at=None, route=None) for r in self.rows}
        self.api = {}
        self.events, self.decisions, self.admissions, self.native_events = [], [], [], []
        self.occupied = 0.
        for gpu, d in enumerate(self.devices):
            self.emit('state', device_id=gpu, generation=0, previous=None, state=d['state'])

    def emit(self, kind, **fields):
        self.events.append(dict(time=self.now, kind=kind, **fields))

    def state(self, gpu, state, due=None):
        d = self.devices[gpu]
        if state in ('starting','off'):
            assert not d['jobs'] and self.sim.scheduler.get_replica_scheduler(gpu).is_empty()
            assert not any(getattr(e, '_replica_id', None) == gpu for _,e in self.sim._event_queue), 'old-generation native event remains'
        if state == 'starting': d['generation'] += 1
        self.emit('state', device_id=gpu, generation=d['generation'], previous=d['state'], state=state)
        d.update(state=state, since=self.now, due=due)

    def clock(self, time):
        assert time >= self.now
        self.occupied += (time - self.now) * sum(d['state'] != 'off' for d in self.devices)
        self.now = time

    def settle(self):
        for gpu, d in enumerate(self.devices):
            for rid in list(d['jobs']):
                request = self.requests[rid]
                if request.completed:
                    assert request.completed_at <= self.now
                    self.records[rid]['completed_at'] = request.completed_at
                    del d['jobs'][rid]
                    self.emit('complete', request_id=rid, device_id=gpu, generation=d['generation'], route='local')
            if d['state'] == 'draining' and not d['jobs']:
                assert self.sim.scheduler.get_replica_scheduler(gpu).is_empty()
                # All native events at this time are processed before settling.
                self.state(gpu, 'stopping', self.now + self.shutdown)
            if d['due'] is not None and d['due'] <= self.now:
                if d['state'] == 'starting': self.state(gpu, 'active')
                elif d['state'] == 'stopping':
                    assert self.sim.scheduler.get_replica_scheduler(gpu).is_empty()
                    self.state(gpu, 'off')
        for rid, due in list(self.api.items()):
            if due <= self.now:
                self.records[rid]['completed_at'] = due
                del self.api[rid]
                self.emit('complete', request_id=rid, route='synthetic_api')

    def advance(self, until):
        if until < self.now: raise ValueError('backward advance')
        while True:
            native = self.sim._event_queue[0][1].time if self.sim._event_queue else float('inf')
            due = min([d['due'] for d in self.devices if d['due'] is not None] + list(self.api.values()) + [float('inf')])
            at = min(native, due)
            if at > until: break
            self.clock(at)
            while self.sim._event_queue and self.sim._event_queue[0][1].time <= at:
                _, event = heapq.heappop(self.sim._event_queue)
                self.sim._set_time(event.time)
                self.sim._add_events(event.handle_event(self.sim.scheduler, self.sim.metric_store))
                self.native_events.append(dict(class_name=type(event).__name__, **event.to_dict()))
            self.settle()
        self.clock(until)
        self.settle()

    def observation(self, tick):
        obs = dict(schema=SCHEMA, time=tick,
            queue_online=[public_job(self.records[r]) for r in self.queue['online']],
            queue_offline=[public_job(self.records[r]) for r in self.queue['offline']],
            devices=[dict(state=d['state'], since=d['since'],
                jobs=[public_job(self.records[r], started_at=t) for r,t in d['jobs'].items()]) for d in self.devices])
        validate_observation(obs)
        return obs

    def actuate(self, target):
        if type(target) is not int or not 0 <= target <= 2: raise ValueError('invalid target')
        def count(): return sum(d['state'] in ('active','starting') for d in self.devices)
        for gpu in (1,0):
            if count() > target and self.devices[gpu]['state'] == 'active': self.state(gpu, 'draining')
        for gpu,d in enumerate(self.devices):
            if count() < target and d['state'] == 'draining': self.state(gpu, 'active')
        for gpu,d in enumerate(self.devices):
            if count() < target and d['state'] == 'off': self.state(gpu, 'starting', self.now + self.startup)
        self.settle()

    def dispatch(self, tick, reserve):
        while True:
            available = [g for g,d in enumerate(self.devices) if d['state'] == 'active' and len(d['jobs']) < 4]
            if not available: break
            kind = self.policy.dispatch_kind(self.observation(tick), reserve)
            if kind is None: break
            gpu = min(available, key=lambda g: (len(self.devices[g]['jobs']),g))
            rid = self.queue[kind].pop(0)
            d = self.devices[gpu]
            assert self.records[rid]['dispatched_at'] is None
            d['jobs'][rid] = self.now
            self.records[rid].update(dispatched_at=self.now, route=gpu)
            self.sim.scheduler.get_replica_scheduler(gpu).add_request(self.requests[rid])
            self.admissions.append(dict(time=self.now, device_id=gpu, request_id=rid, generation=d['generation']))
            self.emit('dispatch', device_id=gpu, generation=d['generation'], request_id=rid, route='local')
        # Schedule once per affected replica to preserve native batching.
        for gpu in sorted({a['device_id'] for a in self.admissions if a['time'] == self.now}):
            self.sim._add_event(ReplicaScheduleEvent(self.now, gpu))
        for rid in list(self.queue['online']):
            if len(self.api) == 8: break
            if self.now - self.records[rid]['arrival_s'] < 10: continue
            self.queue['online'].remove(rid)
            self.api[rid] = self.now + 5
            self.records[rid].update(dispatched_at=self.now, route='synthetic_api')
            self.emit('dispatch', request_id=rid, route='synthetic_api')

    def run(self):
        for tick in range(self.horizon):
            self.advance(tick)
            while self.cursor < len(self.rows) and self.rows[self.cursor]['arrival_s'] <= tick:
                row = self.rows[self.cursor]; self.cursor += 1
                rid = row['request_id']
                self.queue[row['job_type']].append(rid)
                self.sim.metric_store.on_request_arrival(row['arrival_s'], self.requests[rid])
                self.emit('arrival', request_id=rid)
            obs = self.observation(tick)
            decision = self.policy.decide(copy.deepcopy(obs))
            if self.targets is not None:
                decision = dict(decision, target=self.targets(tick), engineering_script_override=True)
            self.decisions.append(dict(time=tick, event_seq_before_action=len(self.events), observation=obs, decision=decision))
            self.actuate(decision['target'])
            self.dispatch(tick, decision['reserve'])
        self.advance(self.horizon)
        visible = [r for r in self.rows if r['arrival_s'] < self.horizon]
        metrics = evaluate_requests(visible, {r['request_id']: self.records[r['request_id']] for r in visible}, horizon=self.horizon)
        return dict(policy=self.policy_name, metrics=metrics, occupied_seconds=self.occupied,
            horizon=self.horizon, native_event_count=len(self.native_events), final_states=[d['state'] for d in self.devices],
            native_pending_events=len(self.sim._event_queue), api_pending=len(self.api),
            P1plus_applicable=False, lifecycle_is_synthetic=not self.historical_parameters, guard_estimator_is_synthetic=not self.historical_parameters,
            lifecycle_source='frozen E development scalars' if self.historical_parameters else 'engineering fixture',
            comparison_status='CPU precheck, not physical validation',
            domain_audit=self.sim.domain_audit)


class AdmissionEvent(BaseEvent):
    """Reference-only input event, run by upstream Simulator.run unchanged."""
    def __init__(self, at, pairs):
        super().__init__(at, EventType.REQUEST_ARRIVAL)
        self.pairs = pairs
    def handle_event(self, scheduler, metric_store):
        for gpu, request in self.pairs:
            metric_store.on_request_arrival(request.arrived_at, request)
            scheduler.get_replica_scheduler(gpu).add_request(request)
        return [ReplicaScheduleEvent(self.time,g) for g in sorted({g for g,r in self.pairs})]


def native_reference(rows, out, admissions=None):
    sim, requests = make_simulator(rows, out)
    if admissions is not None:
        sim._event_queue.clear()
        by_time = {}
        for a in admissions: by_time.setdefault(a['time'], []).append((a['device_id'], requests[a['request_id']]))
        for at,pairs in by_time.items(): sim._add_event(AdmissionEvent(at,pairs))
    sim.run()
    return {rid: dict(r.to_dict()) for rid,r in requests.items()}
