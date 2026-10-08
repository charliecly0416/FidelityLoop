"""Capacity-only step adapter around the unchanged calibrated simulator.

The simulator owns physics, dispatch, events and final settlement. A synchronous
policy callback is its only pause point, after releases and before decisions.
The caller and worker never access simulator state concurrently.
"""
import copy
import math
import queue
import threading

import numpy as np

from scripts.maxopt_v2.calibrated import CalibratedSimulator
from scripts.maxopt_bridge.metrics import cost_breakdown, evaluate_requests


STATES = ('off', 'starting', 'active', 'draining', 'stopping')
CAPACITY_MASK = (1, 1, 1)
OFFLINE_MASK = (1, 0, 0, 0)
OBS_DIM = 30
REWARD_SCALE = 1.0
COOLDOWN_SECONDS = 60
_STOP = object()


class _StopEpisode(Exception):
    pass


def validate_mask(mask, width):
    if mask is None:
        raise ValueError('missing action mask')
    a = np.asarray(mask)
    if a.shape != (width,) or a.dtype.kind not in 'biuf':
        raise ValueError('action mask shape/type mismatch')
    if not np.isfinite(a).all() or not np.isin(a, (0, 1)).all() or not a.any():
        raise ValueError('action mask must contain binary finite entries and a legal action')
    return tuple(int(x) for x in a)


def filter_target(proposal, current, last_change_at, now):
    if type(proposal) is not int or proposal not in (0, 1, 2):
        raise ValueError('proposal must be target capacity 0, 1 or 2')
    target = current if now - last_change_at < COOLDOWN_SECONDS else proposal
    return target, now if target != current else last_change_at


class _VisibleSimulator(CalibratedSimulator):
    """Track only public lifecycle transition timestamps; physics is inherited."""
    def __init__(self, *args, **kwargs):
        self.state_since = [0.0, 0.0]
        super().__init__(*args, **kwargs)

    def state(self, gpu, new, due=None):
        super().state(gpu, new, due)
        self.state_since[gpu] = self.now


def public_observation(sim, target, last_change_at):
    """Construct a whitelist without future rows, ready-at or service progress."""
    def job(rid):
        row = sim.jobs[rid]['row']
        return dict(age_seconds=sim.now - row['arrival_s'],
                    remaining_deadline_seconds=row['deadline_s'] - sim.now)

    return dict(time_seconds=sim.now, target=target,
                since_target_change_seconds=sim.now - last_change_at,
                queues={kind: [job(rid) for rid in sim.queues[kind]]
                        for kind in ('online', 'offline')},
                devices=[dict(state=d['state'], state_age_seconds=sim.now - sim.state_since[g],
                              jobs=[job(rid) for rid in d['jobs']])
                         for g, d in enumerate(sim.devices)],
                api_jobs=[job(rid) for rid in sim.api])


def encode_observation(public):
    """Fixed common scales; no data fitting or environment-specific duration."""
    def jobs(items, count_scale):
        return [len(items) / count_scale,
                max((j['age_seconds'] for j in items), default=0.0) / 300.0,
                min((j['remaining_deadline_seconds'] for j in items), default=0.0) / 300.0]

    values = [public['time_seconds'] / 2100.0, public['target'] / 2.0,
              public['since_target_change_seconds'] / 60.0]
    for kind in ('online', 'offline'):
        values.extend(jobs(public['queues'][kind], 128.0))
    for device in public['devices']:
        values.extend(float(device['state'] == state) for state in STATES)
        values.append(device['state_age_seconds'] / 300.0)
        values.extend(jobs(device['jobs'], 4.0))
    values.extend(jobs(public['api_jobs'], 8.0))
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (OBS_DIM,) or not np.isfinite(vector).all():
        raise ValueError('invalid causal observation')
    return vector


class CapacityEnv:
    """One-second raw PPO environment. Use close() or the context manager.

    step returns (observation, reward, terminated, truncated=False, info).
    A rollout cut is a collector decision, never an episode terminal. Capacity
    masks describe legal proposals; cooldown is a deterministic many-to-one
    filter, so all three proposals retain their own behavior probabilities.
    """
    def __init__(self, rows, contract, model, *, handshake_timeout=30.0):
        if not math.isfinite(handshake_timeout) or handshake_timeout <= 0:
            raise ValueError('handshake timeout must be finite and positive')
        self.rows, self.contract, self.model = copy.deepcopy((rows, contract, model))
        self.handshake_timeout = handshake_timeout
        self._thread = None
        self._sim = None
        self._terminated = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        raise TypeError('checkpoint only model/config/RNG state, never a live environment')

    def close(self):
        if self._thread is not None:
            if self._thread.is_alive():
                self._commands.put(_STOP)
            self._thread.join(self.handshake_timeout)
            if self._thread.is_alive():
                raise RuntimeError('simulator worker did not acknowledge cancellation')
            self._thread = None
        self._terminated = True

    def _ledger(self, terminal=False):
        sim = self._sim
        misses = sum(row['job_type'] == 'offline'
                     and (terminal or row['deadline_s'] <= sim.now)
                     and (sim.jobs[row['request_id']]['completed_at'] is None
                          or sim.jobs[row['request_id']]['completed_at'] > row['deadline_s'])
                     for row in sim.rows)
        return cost_breakdown(sim.rows, {'populations': {'offline': {'not_on_time': misses}}},
                              occupied_seconds=sim.occupied, starts=sim.starts,
                              shutdowns=sim.shutdowns, api_ids=sim.api_ids,
                              prices=sim.contract['accounting'])

    def _snapshot(self, terminal=False):
        public = public_observation(self._sim, self._target, self._last_change_at)
        return dict(observation=encode_observation(public), public=public,
                    capacity_mask=list(CAPACITY_MASK), offline_mask=list(OFFLINE_MASK),
                    ledger=self._ledger(terminal))

    def decide(self, script_observation):
        # Called only by the worker, at the authoritative predecision boundary.
        info = self._snapshot()
        info['script_observation'] = dict(script_observation)
        self._replies.put(('decision', info))
        # An intentional pause may include a PPO update. Only the public wait
        # for simulator progress and cancellation is bounded, not caller think
        # time. close/reset always supplies the cancellation command.
        target = self._commands.get()
        if target is _STOP:
            raise _StopEpisode()
        return target

    def _work(self):
        try:
            result = self._sim.run()
            info = self._snapshot(terminal=True)
            info['result'] = result
            info['request_metrics'] = evaluate_requests(self.rows, result['requests'], horizon=self._sim.end)
            for key, value in result['summary']['costs'].items():
                if not math.isclose(value, info['ledger'][key], rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError('terminal original/bridge ledger mismatch: ' + key)
            self._replies.put(('terminal', info))
        except _StopEpisode:
            pass
        except BaseException as exc:
            self._replies.put(('error', exc))

    def _receive(self):
        try:
            kind, value = self._replies.get(timeout=self.handshake_timeout)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError('simulator failed to reach next decision boundary') from exc
        if kind == 'error':
            self.close()
            raise RuntimeError('simulator worker failed') from value
        self._terminated = kind == 'terminal'
        if self._terminated:
            self._thread.join(self.handshake_timeout)
            if self._thread.is_alive():
                raise RuntimeError('terminal worker did not exit')
            self._thread = None
        self._info = value
        return value

    def reset(self):
        self.close()
        self._target, self._last_change_at = 2, -COOLDOWN_SECONDS
        self._commands, self._replies = queue.Queue(), queue.Queue()
        self._sim = _VisibleSimulator(self.rows, self.contract, self.model, 'all2')
        self._sim.policy = self
        self._previous_ledger = cost_breakdown([], {'populations': {'offline': {'not_on_time': 0}}},
            occupied_seconds=0.0, starts=0, shutdowns=0, api_ids=[], prices=self.contract['accounting'])
        self._thread = threading.Thread(target=self._work, name='maxopt-capacity-episode', daemon=True)
        self._thread.start()
        info = self._receive()
        return info['observation'].copy(), copy.deepcopy(info)

    def step(self, proposal, *, behavior_log_prob, behavior_mask):
        if self._terminated or self._thread is None:
            raise RuntimeError('step requires a live, nonterminal episode')
        mask = validate_mask(behavior_mask, 3)
        if mask != CAPACITY_MASK:
            raise ValueError('behavior mask differs from the decision-time environment mask')
        if not isinstance(behavior_log_prob, (int, float)) or not math.isfinite(behavior_log_prob) or behavior_log_prob > 1e-7:
            raise ValueError('behavior log-probability must be finite and nonpositive')
        now = self._info['public']['time_seconds']
        target, changed = filter_target(proposal, self._target, self._last_change_at, now)
        action = dict(time_seconds=now, proposal=proposal, filtered_target=target,
                      executed_target=target, behavior_mask=list(mask),
                      behavior_log_prob=float(behavior_log_prob), last_change_at=changed)
        self._target, self._last_change_at = target, changed
        self._commands.put(target)
        info = self._receive()
        delta = {key: info['ledger'][key] - self._previous_ledger[key]
                 for key in ('gpu', 'startup', 'shutdown', 'synthetic_api', 'operating', 'offline_miss', 'total')}
        if any(value < -1e-10 for value in delta.values()):
            raise RuntimeError('accrued ledger moved backwards')
        self._previous_ledger = dict(info['ledger'])
        info.update(action=action, cost_delta=delta)
        return info['observation'].copy(), -delta['total'] / REWARD_SCALE, self._terminated, False, copy.deepcopy(info)


def environment_pair(context):
    """Validate the complete model diff before creating training environments."""
    old, fixed = (copy.deepcopy(context['models'][name]) for name in ('C', 'C30'))
    expected = (69.219185665, 30)
    if (old.get('startup_seconds'), fixed.get('startup_seconds')) != expected:
        raise ValueError('unregistered paired startup endpoints')
    left, right = copy.deepcopy(old), copy.deepcopy(fixed)
    left.pop('startup_seconds'); right.pop('startup_seconds')
    if left != right:
        raise ValueError('paired environments differ outside startup_seconds')
    return {'C': old, 'C30': fixed}
