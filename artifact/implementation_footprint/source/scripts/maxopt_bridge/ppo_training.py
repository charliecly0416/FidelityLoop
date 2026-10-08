"""CPU-only bridge PPO execution and evidence; every real run requires a gate.

This driver adds no learning algorithm. It calls the accepted CapacityTrainer
and CapacityEnv, keeps proposal provenance, and selects only by per-window P1+.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from scripts.maxopt_bridge import data, f4, protocol
from scripts.maxopt_bridge.ppo_capacity import COMMON_CONFIG, CapacityNetwork, CapacityTrainer, make_transition
from scripts.maxopt_bridge.ppo_env import CapacityEnv, environment_pair


SOURCE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_WINDOWS = tuple('train_' + regime + '_v2' for regime in ('burst_offline', 'recovery', 'steady'))
VALIDATION_WINDOWS = tuple('validation_' + regime + '_v2' for regime in ('burst_offline', 'recovery', 'steady'))
BENCHMARK_SPEC = dict(schema='maxopt-bridge-ppo-benchmark-v1', seed=20261001,
    train_window='train_steady_v2', environments=['C', 'C30'], execution='serial',
    torch_threads=1, blas_threads=1, script_steps=2100, rollout_steps=2100,
    updates=1, validation_windows=list(VALIDATION_WINDOWS), validation_environment='C30',
    deterministic_validation=True, checkpoint_candidate=False, reusable_initialization=False,
    purpose='throughput, memory and numerical correctness only; no hyperparameter search')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def tensor_digest(state_dict):
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_inputs(root=SOURCE_ROOT):
    root = Path(root).resolve()
    context = f4.load_context(root)
    environment_pair(context)
    manifest = read_json(root / protocol.DOC / 'DATA_SPLIT_MANIFEST.json')
    expected = list(TRAIN_WINDOWS + VALIDATION_WINDOWS)
    files = {name: root / protocol.INPUT / 'development' / (name + '.jsonl') for name in expected}
    digests = {name: file_sha(path) for name, path in files.items()}
    for name, path in files.items():
        if manifest['files'].get(str(path.relative_to(root))) != digests[name]:
            raise ValueError('development file differs from accepted split manifest: ' + name)
    rows = {name: [json.loads(line) for line in path.read_text().splitlines()] for name, path in files.items()}
    partitions = data.inspect_partitions(rows)
    return context, rows, digests, partitions


def selection_contract(context, data_sha256):
    root = context['root']
    return dict(schema='maxopt-bridge-ppo-selection-v1', digest_encoding='canonical JSON UTF-8',
        evaluation_model='C30', model_sha256=object_sha(context['models']['C30']), guard=False,
        deterministic=True, validation_windows=list(VALIDATION_WINDOWS),
        data_sha256={name: data_sha256[name] for name in VALIDATION_WINDOWS},
        simulator_contract_sha256=file_sha(root / protocol.INPUT / 'simulator_contract.json'),
        metrics_code_sha256=file_sha(root / 'scripts/maxopt_bridge/metrics.py'),
        environment_code_sha256=file_sha(root / 'scripts/maxopt_bridge/ppo_env.py'),
        primary_gate='P1plus separately in every validation window',
        eligible_order=['equal_window_mean_total_cost', 'checkpoint_index', 'seed_id'],
        diagnostic_order=['failed_windows', 'unfinished', 'late', 'equal_window_mean_total_cost',
                          'checkpoint_index', 'seed_id'],
        initialization_is_candidate=False, engineering_probe_is_candidate=False)


def summarize_validation(windows, selection):
    if set(windows) != set(selection['validation_windows']) or len(windows) != 3:
        raise ValueError('all three exact validation windows are mandatory')
    failures, unfinished, late, costs = 0, 0, 0, []
    for name in selection['validation_windows']:
        result = windows[name]
        if result.get('prediction_model') != 'C30' or result.get('guard') is not False:
            raise ValueError('selection requires common C30 with guard disabled')
        if result.get('deterministic') is not True:
            raise ValueError('selection inference must be deterministic')
        if result.get('input_sha256') != selection['data_sha256'][name]:
            raise ValueError('validation input identity mismatch')
        populations = result['request_metrics']['populations']
        ontime = True
        for kind in ('online', 'offline'):
            pop = populations[kind]
            keys = ('arrivals', 'completed', 'on_time', 'late', 'unfinished', 'not_on_time')
            if any(type(pop.get(key)) is not int or pop[key] < 0 for key in keys):
                raise ValueError('invalid per-window populations')
            if (pop['completed'] + pop['unfinished'] != pop['arrivals']
                    or pop['on_time'] + pop['late'] != pop['completed']
                    or pop['late'] + pop['unfinished'] != pop['not_on_time']):
                raise ValueError('population denominators or terminal accounting disagree')
            ontime &= pop['not_on_time'] == 0
            unfinished += pop['unfinished']; late += pop['late']
        eligible = ontime and populations['online']['arrivals'] > 0 and populations['offline']['arrivals'] == 120
        if result['request_metrics'].get('P1plus') is not eligible:
            raise ValueError('stored P1plus disagrees with full populations')
        failures += not eligible
        cost = result['costs']['total']
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError('invalid validation scenario cost')
        costs.append(cost)
    return dict(eligible=failures == 0, failed_windows=failures, unfinished=unfinished,
                late=late, equal_window_mean_total_cost=sum(costs) / len(costs))


def rank_candidates(candidates, selection):
    eligible, diagnostic, seen = [], [], set()
    for candidate in candidates:
        identity = (candidate['training_environment'], candidate['seed'], candidate['checkpoint_index'])
        if identity in seen:
            raise ValueError('duplicate checkpoint candidate')
        seen.add(identity)
        if (candidate.get('kind') != 'training_checkpoint' or candidate.get('actual_steps', 0) <= 0
                or type(candidate.get('checkpoint_index')) is not int or candidate['checkpoint_index'] <= 0
                or candidate.get('evaluation_due') is not True
                or candidate.get('selection_contract_sha256') != object_sha(selection)):
            raise ValueError('initial/probe/unregistered checkpoint cannot enter selection')
        summary = summarize_validation(candidate['validation'], selection)
        row = {**candidate, 'selection_summary': summary}
        diagnostic.append(row)
        if summary['eligible']:
            eligible.append(row)
    eligible.sort(key=lambda c: (c['selection_summary']['equal_window_mean_total_cost'], c['checkpoint_index'], c['seed']))
    diagnostic.sort(key=lambda c: tuple(c['selection_summary'][key] for key in
        ('failed_windows', 'unfinished', 'late', 'equal_window_mean_total_cost')) + (c['checkpoint_index'], c['seed']))
    return dict(eligible=eligible, diagnostic=diagnostic,
                selected=eligible[0] if eligible else None,
                diagnostic_selected=diagnostic[0] if diagnostic else None)


def episode_order(seed, episodes):
    if type(seed) is not int or type(episodes) is not int or episodes <= 0:
        raise ValueError('seed and episode budget must be explicit integers')
    rng, result = random.Random(seed), []
    while len(result) < episodes:
        block = list(TRAIN_WINDOWS)
        rng.shuffle(block)
        result.extend(block)
    return result[:episodes]


def checkpoint_schedule(episodes, interval):
    if type(interval) is not int or interval <= 0 or type(episodes) is not int or episodes <= 0:
        raise ValueError('invalid checkpoint schedule')
    return sorted(set(range(interval, episodes + 1, interval)) | {episodes})


def verify_s04(acceptance_path):
    acceptance_path = Path(acceptance_path).resolve()
    acceptance = read_json(acceptance_path)
    if acceptance.get('phase') != 'S04' or acceptance.get('decision') != 'ACCEPTED':
        raise ValueError('S04 coordinator acceptance is required before any actual run')
    delivery_path = acceptance_path.parent / 'executor/s04/DELIVERY_SHA256.json'
    if file_sha(delivery_path) != acceptance.get('executor_delivery_sha256'):
        raise ValueError('S04 accepted delivery SHA mismatch')
    delivery = read_json(delivery_path)
    if Path(delivery['workspace']).resolve() != SOURCE_ROOT.resolve():
        raise ValueError('S04 acceptance refers to another workspace')
    for name, meta in delivery['owned_files'].items():
        if file_sha(SOURCE_ROOT / name) != meta['sha256']:
            raise ValueError('accepted S04 file changed: ' + name)
    receipt_path = acceptance_path.parent / 'executor/r0/R0_RECEIPT.json'
    if file_sha(receipt_path) != delivery.get('shared_source_receipt_sha256'):
        raise ValueError('original-source receipt identity mismatch')
    for name, meta in read_json(receipt_path)['source_inventory'].items():
        if file_sha(SOURCE_ROOT / name) != meta['sha256']:
            raise ValueError('received original source changed: ' + name)
    config_sha = file_sha(Path(__file__).with_name('ppo_common.json'))
    if config_sha != acceptance.get('common_config_sha256'):
        raise ValueError('accepted common algorithm config changed')
    review_path = Path(acceptance['review_path'])
    if file_sha(review_path) != acceptance.get('review_sha256'):
        raise ValueError('independent S04 review binding mismatch')
    # protocol.check recomputes original input and protected-source identities.
    protocol.check(SOURCE_ROOT)
    return dict(s04_acceptance_sha256=file_sha(acceptance_path),
                s04_delivery_sha256=file_sha(delivery_path), common_config_sha256=config_sha)


def validate_run_lock(lock, mode, bindings, selection, data_sha256):
    seeds = [0, 1, 2] if mode == 'pilot' else [101, 102, 103, 104, 105]
    episode_cap, wall_cap = (20, 21600) if mode == 'pilot' else (100, 86400)
    if mode not in ('pilot', 'formal') or lock.get('decision') != 'AUTHORIZED' or lock.get('mode') != mode:
        raise ValueError('explicit phase-specific run lock is required')
    if any(lock.get(key) != value for key, value in bindings.items()):
        raise ValueError('run lock has stale S04/config acceptance bindings')
    if (lock.get('driver_sha256') != file_sha(__file__)
            or lock.get('selection_contract_sha256') != object_sha(selection)
            or lock.get('selection_contract') != selection or lock.get('data_sha256') != data_sha256):
        raise ValueError('run lock driver, selection or data binding mismatch')
    episodes = lock.get('episodes_per_run')
    if type(episodes) is not int or not 1 <= episodes <= episode_cap:
        raise ValueError('episode budget exceeds common cap')
    schedule = checkpoint_schedule(episodes, lock.get('checkpoint_every_episodes'))
    if lock.get('checkpoint_episodes') != schedule:
        raise ValueError('checkpoint frequency not completely frozen')
    expected_runs = [dict(seed=seed, training_environment=env) for seed in seeds for env in ('C', 'C30')]
    if lock.get('runs') != expected_runs:
        raise ValueError('run order must contain paired seeds and both environments exactly once')
    if lock.get('episode_order') != {str(seed): episode_order(seed, episodes) for seed in seeds}:
        raise ValueError('paired episode order mismatch')
    seconds = lock.get('total_wall_seconds')
    if type(seconds) is not int or not 1 <= seconds <= wall_cap:
        raise ValueError('total wall budget exceeds registered hard cap')
    if (lock.get('torch_threads') != 1 or lock.get('blas_threads') != 1
            or lock.get('execution') != 'serial' or lock.get('from_scratch') is not True):
        raise ValueError('CPU/serial/from-scratch contract changed')
    return schedule


class WallBudgetExceeded(RuntimeError):
    pass


class WallBudget:
    def __init__(self, cap_seconds, elapsed_seconds=0.0, clock=time.monotonic):
        if not math.isfinite(cap_seconds) or not math.isfinite(elapsed_seconds) or not 0 <= elapsed_seconds <= cap_seconds:
            raise ValueError('invalid accumulated wall budget')
        self.cap, self.previous, self.clock = cap_seconds, elapsed_seconds, clock
        self.start = clock()

    def elapsed(self):
        return self.previous + self.clock() - self.start

    def check(self):
        if self.elapsed() >= self.cap:
            raise WallBudgetExceeded('whole-study cumulative wall cap reached')


def save_checkpoint(path, trainer, metadata):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise ValueError('checkpoint is immutable and cannot be overwritten')
    payload = dict(model_state_dict=trainer.model.state_dict(), optimizer_state_dict=trainer.optimizer.state_dict(),
        torch_rng_state=torch.random.get_rng_state(), current_iteration=trainer.current_iteration,
        early_stop_count=trainer.early_stop_count, current_lr=trainer.current_lr,
        adaptive_kl_coef=trainer.adaptive_kl_coef, metadata=copy.deepcopy(metadata))
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    receipt = dict(metadata=metadata, checkpoint_sha256=file_sha(path),
                   tensor_sha256=tensor_digest(trainer.model.state_dict()))
    atomic_json(path.with_suffix('.json'), receipt)
    return receipt


def restore_checkpoint(path, trainer, expected):
    path = Path(path)
    receipt = read_json(path.with_suffix('.json'))
    if file_sha(path) != receipt['checkpoint_sha256']:
        raise ValueError('checkpoint byte identity mismatch')
    metadata = receipt['metadata']
    if metadata.get('kind') != 'training_checkpoint' or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('probe/initial/stale checkpoint cannot resume a training run')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['metadata'] != metadata or tensor_digest(payload['model_state_dict']) != receipt['tensor_sha256']:
        raise ValueError('checkpoint receipt/payload/tensor mismatch')
    trainer.model.load_state_dict(payload['model_state_dict'])
    trainer.optimizer.load_state_dict(payload['optimizer_state_dict'])
    for name in ('current_iteration', 'early_stop_count', 'current_lr', 'adaptive_kl_coef'):
        setattr(trainer, name, payload[name])
    torch.random.set_rng_state(payload['torch_rng_state'])
    return metadata


def runtime_identity():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
    except ImportError:
        peak_bytes = None
    return dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), numpy=np.__version__,
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                environment_threads={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                peak_resident_bytes=peak_bytes, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                device='cpu', execution='serial')


def configure_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(key) != '1'
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('set CPU-only device and all three BLAS thread variables before launching Python')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def new_trainer(seed):
    torch.random.default_generator.manual_seed(seed)
    return CapacityTrainer(CapacityNetwork())


def run_episode(context, rows, prediction_model, model, *, sample, budget, trace_path, script=False):
    """Called only after authorization by benchmark/study entrypoints.

    One JSONL row is flushed per completed transition, including the original
    proposal/mask/log-prob. An interrupted episode is evidence, not a completed
    optimizer input or an automatically restarted scientific run.
    """
    trajectory = []
    trace_path = Path(trace_path)
    if trace_path.exists():
        raise ValueError('trace is immutable; cannot repeat or overwrite an episode')
    with trace_path.open('x', buffering=1) as log, CapacityEnv(
            rows, context['simulator_contract'], context['models'][prediction_model]) as env:
        obs, info = env.reset()
        for index in range(COMMON_CONFIG['episode_seconds']):
            budget.check()
            if script:
                proposal, logprob, value = 2, 0.0, 0.0
            else:
                with torch.no_grad():
                    action, _, lp, v, _ = model.get_action(torch.as_tensor(obs)[None],
                        torch.tensor([info['capacity_mask']]), torch.tensor([info['offline_mask']]),
                        deterministic=not sample)
                proposal, logprob, value = int(action.item()), float(lp.item()), float(v.item())
            next_obs, reward, terminal, truncated, next_info = env.step(proposal,
                behavior_log_prob=logprob, behavior_mask=info['capacity_mask'])
            if truncated or terminal != (index + 1 == COMMON_CONFIG['episode_seconds']):
                raise RuntimeError('episode timing deviates from the common 2100-second contract')
            if sample:
                trajectory.append(make_transition(obs, value, reward, terminal, next_info))
            record = dict(step=index + 1, observation=obs.tolist(), value=value, reward=reward,
                terminated=terminal, action=next_info['action'], cost_delta=next_info['cost_delta'])
            log.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
            obs, info = next_obs, next_info
        budget.check()
    if any(t.name == 'maxopt-capacity-episode' for t in threading.enumerate()):
        raise RuntimeError('capacity worker remains after episode; refuse the next episode')
    return trajectory, info


def write_result(path, info):
    payload = dict(result=info['result'], request_metrics=info['request_metrics'], ledger=info['ledger'])
    path = Path(path)
    if path.exists():
        raise ValueError('raw result is immutable')
    with path.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
        compressed.write(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode())
    return file_sha(path)


def evaluate_model(context, partitions, digests, model, prediction_model, output, budget):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {}
    for name in VALIDATION_WINDOWS:
        started = time.monotonic()
        _, info = run_episode(context, partitions[name], prediction_model, model, sample=False, budget=budget,
                              trace_path=output / (name + '.jsonl'))
        raw_path = output / (name + '.json.gz')
        raw_sha = write_result(raw_path, info)
        result[name] = dict(prediction_model=prediction_model, guard=False, deterministic=True,
            input_sha256=digests[name], request_metrics=info['request_metrics'], costs=info['ledger'],
            outside_window=info['result']['summary']['outside_window'], raw_result_path=str(raw_path),
            raw_result_sha256=raw_sha, elapsed_seconds=time.monotonic() - started)
    atomic_json(output / 'EVALUATION.json', result)
    return result


def finite_update(metrics):
    if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('nonfinite PPO update diagnostic')
    if any('nonfinite_count' in key and value != 0 for key, value in metrics.items()):
        raise RuntimeError('PPO update reported nonfinite intermediate values')


def gradient_snapshot(trainer):
    result = {name: dict(finite=bool(torch.isfinite(parameter.grad).all()),
                         norm=float(parameter.grad.norm()), nonzero=int(torch.count_nonzero(parameter.grad)))
              for name, parameter in trainer.model.named_parameters() if parameter.grad is not None}
    if not result or any(not item['finite'] for item in result.values()):
        raise RuntimeError('missing or nonfinite gradients after PPO update')
    return result


def benchmark(acceptance_path, output, *, wall_seconds=600):
    bindings = verify_s04(acceptance_path)
    configure_cpu()
    context, partitions, digests, _ = load_inputs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = {**BENCHMARK_SPEC, **bindings, 'driver_sha256': file_sha(__file__),
            'data_sha256': digests, 'selection_contract_sha256': object_sha(selection_contract(context, digests))}
    atomic_json(output / 'BENCHMARK_SPEC.json', spec)
    budget = WallBudget(wall_seconds)
    results = []
    started = time.monotonic()
    try:
        for environment in BENCHMARK_SPEC['environments']:
            run_dir = output / environment
            run_dir.mkdir()
            name = BENCHMARK_SPEC['train_window']
            trainer = new_trainer(BENCHMARK_SPEC['seed'])
            initialization = tensor_digest(trainer.model.state_dict())
            before = runtime_identity()
            start = time.monotonic()
            _, info = run_episode(context, partitions[name], environment, trainer.model,
                sample=False, budget=budget, trace_path=run_dir / 'script.jsonl', script=True)
            script_seconds = time.monotonic() - start
            write_result(run_dir / 'script.json.gz', info)
            start = time.monotonic()
            trajectory, info = run_episode(context, partitions[name], environment, trainer.model,
                sample=True, budget=budget, trace_path=run_dir / 'rollout.jsonl')
            rollout_seconds = time.monotonic() - start
            write_result(run_dir / 'rollout.json.gz', info)
            budget.check()
            start = time.monotonic()
            metrics = trainer.update([trajectory], iteration=0, bootstrap_values=[0.0])
            finite_update(metrics)
            gradients = gradient_snapshot(trainer)
            update_seconds = time.monotonic() - start
            budget.check()
            save_checkpoint(run_dir / 'PROBE_NOT_A_CANDIDATE.pt', trainer, dict(kind='engineering_probe',
                seed=BENCHMARK_SPEC['seed'], actual_steps=len(trajectory), training_environment=environment,
                candidate=False, reusable_initialization=False, **bindings))
            start = time.monotonic()
            validation = evaluate_model(context, partitions, digests, trainer.model, 'C30', run_dir / 'validation', budget)
            item = dict(training_environment=environment, initialization_tensor_sha256=initialization,
                script_seconds=script_seconds, rollout_seconds=rollout_seconds, update_seconds=update_seconds,
                validation_seconds=time.monotonic()-start, actual_rollout_steps=len(trajectory),
                optimizer_minibatch_updates=metrics['n_updates'], update_metrics=metrics,
                gradients=gradients, thread_cleanup_verified=True,
                validation=validation, runtime_before=before, runtime_after=runtime_identity())
            results.append(item)
            atomic_json(output / 'BENCHMARK_PROGRESS.json', results)
        assert results[0]['initialization_tensor_sha256'] == results[1]['initialization_tensor_sha256']
        final = dict(status='ENGINEERING_BENCHMARK_COMPLETE', spec=spec, environments=results,
                     elapsed_seconds=time.monotonic()-started, candidate_checkpoints=0,
                     claim='single fixed train window probe; budget must allow for all window sizes and variability')
        atomic_json(output / 'BENCHMARK_RESULTS.json', final)
        return final
    except BaseException as exc:
        atomic_json(output / 'INTERRUPTION.json', dict(kind=type(exc).__name__, message=str(exc),
            completed_environments=len(results), elapsed_seconds=time.monotonic()-started,
            status='FAILED_OR_INTERRUPTED_NOT_A_SCIENTIFIC_RESULT'))
        raise


def _candidate_metadata(mode, run, episode, lock, lock_sha, selection, bindings):
    return dict(kind='training_checkpoint', mode=mode, seed=run['seed'],
        training_environment=run['training_environment'], checkpoint_index=episode,
        episode_completed=episode, actual_steps=episode * COMMON_CONFIG['episode_seconds'],
        evaluation_due=episode in lock['checkpoint_episodes'], lock_sha256=lock_sha,
        selection_contract_sha256=object_sha(selection), **bindings)


def run_one(context, partitions, digests, run, mode, lock, lock_sha, bindings, selection, output, budget, resume):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    status_path = output / 'RUN_STATE.json'
    trainer = new_trainer(run['seed'])
    initialization = tensor_digest(trainer.model.state_dict())
    state = dict(status='READY', phase='boundary', run=run, episode_completed=0, actual_steps=0,
        initialization_tensor_sha256=initialization, lock_sha256=lock_sha, checkpoints=[], updates=[],
        starting_study_wall_seconds=budget.elapsed(), elapsed_seconds=0.0, selection_contract_sha256=object_sha(selection))
    if resume and status_path.exists():
        state = read_json(status_path)
        if state['lock_sha256'] != lock_sha or state['run'] != run:
            raise ValueError('resume identity mismatch')
        if state['phase'] != 'boundary':
            raise ValueError('partial rollout/update/evaluation requires a recorded technical-retry decision; no silent replay')
        if state['status'] == 'COMPLETE':
            return state
        if state['episode_completed']:
            restore_checkpoint(output / ('episode_%04d.pt' % state['episode_completed']), trainer,
                dict(mode=mode, seed=run['seed'], training_environment=run['training_environment'], lock_sha256=lock_sha,
                     selection_contract_sha256=object_sha(selection), **bindings))
        elif any(output.glob('episode_*.jsonl')):
            raise ValueError('uncommitted rollout exists; cannot restart as a fresh run')
    elif not resume or not status_path.exists():
        save_checkpoint(output / 'initial.pt', trainer, dict(kind='initialization', candidate=False,
                        seed=run['seed'], training_environment=run['training_environment'], lock_sha256=lock_sha, **bindings))
    atomic_json(status_path, state)
    start = time.monotonic()
    prior_elapsed = state['elapsed_seconds']
    try:
        order = lock['episode_order'][str(run['seed'])]
        for index in range(state['episode_completed'], lock['episodes_per_run']):
            budget.check()
            episode, name = index + 1, order[index]
            state.update(status='RUNNING', phase='collecting', current_episode=episode, current_window=name)
            atomic_json(status_path, state)
            trajectory, info = run_episode(context, partitions[name], run['training_environment'], trainer.model,
                sample=True, budget=budget, trace_path=output / ('episode_%04d.jsonl' % episode))
            write_result(output / ('episode_%04d.json.gz' % episode), info)
            state.update(phase='updating', actual_steps=episode * COMMON_CONFIG['episode_seconds'])
            atomic_json(status_path, state)
            budget.check()
            update = trainer.update([trajectory], iteration=index, bootstrap_values=[0.0])
            finite_update(update)
            state['updates'].append(dict(episode=episode, window=name, metrics=update,
                reward_sum=sum(t.reward for t in trajectory), request_metrics=info['request_metrics'], costs=info['ledger']))
            budget.check()
            metadata = _candidate_metadata(mode, run, episode, lock, lock_sha, selection, bindings)
            receipt = save_checkpoint(output / ('episode_%04d.pt' % episode), trainer, metadata)
            candidate = {**metadata, 'path': str(output / ('episode_%04d.pt' % episode)),
                         'checkpoint_sha256': receipt['checkpoint_sha256'], 'tensor_sha256': receipt['tensor_sha256']}
            if metadata['evaluation_due']:
                state['phase'] = 'evaluating'
                atomic_json(status_path, state)
                candidate['validation'] = evaluate_model(context, partitions, digests, trainer.model, 'C30',
                    output / ('validation_%04d' % episode), budget)
                candidate['selection_summary'] = summarize_validation(candidate['validation'], selection)
            state['checkpoints'].append(candidate)
            state.update(phase='boundary', episode_completed=episode,
                         elapsed_seconds=prior_elapsed + time.monotonic() - start,
                         study_wall_seconds=budget.elapsed())
            atomic_json(status_path, state)
        state.update(status='COMPLETE', runtime=runtime_identity())
        atomic_json(status_path, state)
        return state
    except BaseException as exc:
        current_episode = state.get('current_episode', 0)
        trace = output / ('episode_%04d.jsonl' % current_episode)
        # A boundary stop still names the last committed episode. Its samples
        # are already in episode_completed; only an uncommitted trace is extra.
        trace_steps = 0
        if current_episode > state['episode_completed'] and trace.exists():
            with trace.open() as stream:
                trace_steps = sum(1 for _ in stream)
        state.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, WallBudgetExceeded)) else 'FAILED',
            exception=dict(type=type(exc).__name__, message=str(exc)), partial_episode_trace_steps=trace_steps,
            observed_collected_steps=state['episode_completed'] * COMMON_CONFIG['episode_seconds'] + trace_steps,
            elapsed_seconds=prior_elapsed + time.monotonic()-start, study_wall_seconds=budget.elapsed())
        atomic_json(status_path, state)
        raise


def verify_formal(authorization_path, lock):
    authorization = read_json(authorization_path)
    if (authorization.get('phase') != 'S07' or authorization.get('decision') != 'AUTHORIZED'
            or file_sha(authorization_path) != lock.get('formal_authorization_sha256')):
        raise ValueError('formal training requires its bound S07 authorization')
    s06_path = Path(authorization['s06_acceptance_path'])
    s06 = read_json(s06_path)
    if (file_sha(s06_path) != authorization.get('s06_acceptance_sha256') or s06.get('phase') != 'S06'
            or s06.get('decision') != 'ACCEPTED' or s06.get('continue_formal_ppo') is not True):
        raise ValueError('S06 must explicitly accept continuation to formal PPO')


def shared_rule_reference(context):
    output = {}
    for name in VALIDATION_WINDOWS:
        path = context['root'] / 'artifacts/maxopt_bridge_20260930/f4/C30' / protocol.F4_IDS[1] / (name + '.json.gz')
        with gzip.open(path, 'rt') as stream:
            cell = json.load(stream)
        output[name] = dict(policy_id=protocol.F4_IDS[1], prediction_model='C30',
            request_metrics=cell['request_metrics'], costs=cell['costs'],
            source_path=str(path), source_sha256=file_sha(path),
            source='accepted unchanged F4 replay; reused common-rule reference, not a new run')
    return output


def write_results_csv(path, runs):
    columns = ['training_environment', 'seed', 'checkpoint_index', 'actual_steps', 'window',
        'evaluation_model', 'P1plus', 'online_arrivals', 'online_late', 'online_unfinished',
        'offline_arrivals', 'offline_late', 'offline_unfinished', 'operating', 'offline_miss', 'total']
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            for checkpoint in run['checkpoints']:
                for window, result in checkpoint.get('validation', {}).items():
                    item = {key: checkpoint[key] for key in ('training_environment', 'seed', 'checkpoint_index', 'actual_steps')}
                    item.update(window=window, evaluation_model='C30', P1plus=result['request_metrics']['P1plus'])
                    for kind in ('online', 'offline'):
                        for field in ('arrivals', 'late', 'unfinished'):
                            item[kind + '_' + field] = result['request_metrics']['populations'][kind][field]
                    item.update({key: result['costs'][key] for key in ('operating', 'offline_miss', 'total')})
                    writer.writerow(item)


def run_study(mode, acceptance_path, lock_path, output, *, resume=False, elapsed_before=0.0, formal_authorization=None):
    bindings = verify_s04(acceptance_path)
    configure_cpu()
    context, partitions, digests, _ = load_inputs()
    selection = selection_contract(context, digests)
    lock = read_json(lock_path)
    validate_run_lock(lock, mode, bindings, selection, digests)
    if mode == 'formal':
        verify_formal(formal_authorization, lock)
    lock_sha = file_sha(lock_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    state_path = output / 'STUDY_STATE.json'
    if resume and state_path.exists():
        state = read_json(state_path)
        if state['lock_sha256'] != lock_sha:
            raise ValueError('resume cannot change the original common lock or budget')
        elapsed_before = max(elapsed_before, state.get('elapsed_seconds', 0.0))
    else:
        state = dict(status='RUNNING', mode=mode, lock_sha256=lock_sha, runs=[], cross_evaluation={},
                     elapsed_seconds=elapsed_before, planned_runs=lock['runs'], selection_contract=selection)
    budget = WallBudget(lock['total_wall_seconds'], elapsed_before)
    try:
        completed = []
        for run in lock['runs']:
            budget.check()
            run_name = run['training_environment'] + '_seed_' + str(run['seed'])
            result = run_one(context, partitions, digests, run, mode, lock, lock_sha, bindings,
                             selection, output / run_name, budget, resume=resume)
            completed.append(result)
            counterparts = [r for r in completed if r['run']['seed'] == run['seed']]
            if len(counterparts) == 2 and counterparts[0]['initialization_tensor_sha256'] != counterparts[1]['initialization_tensor_sha256']:
                raise RuntimeError('paired initial tensor identities differ')
            state.update(runs=completed, elapsed_seconds=budget.elapsed())
            atomic_json(state_path, state)
        for run in completed:
            name = run['run']['training_environment'] + '_seed_' + str(run['run']['seed'])
            if name in state['cross_evaluation']:
                continue
            candidates = [c for c in run['checkpoints'] if c['evaluation_due']]
            ranked = rank_candidates(candidates, selection)
            chosen = ranked['selected'] or ranked['diagnostic_selected']
            trainer = new_trainer(run['run']['seed'])
            restore_checkpoint(chosen['path'], trainer, dict(lock_sha256=lock_sha,
                seed=run['run']['seed'], training_environment=run['run']['training_environment'], **bindings))
            evaluation = evaluate_model(context, partitions, digests, trainer.model, 'C', output / ('cross_' + name), budget)
            state['cross_evaluation'][name] = dict(checkpoint_index=chosen['checkpoint_index'],
                deployment_eligible=ranked['selected'] is not None, C=evaluation, C30=chosen['validation'],
                scope='known validation diagnostics; C30 values reused from exact same checkpoint')
            state['elapsed_seconds'] = budget.elapsed()
            atomic_json(state_path, state)
        all_candidates = [c for run in completed for c in run['checkpoints'] if c['evaluation_due']]
        state['selection_by_training_environment'] = {environment: rank_candidates(
            [c for c in all_candidates if c['training_environment'] == environment], selection)
            for environment in ('C', 'C30')}
        state['both_groups_have_eligible_checkpoint'] = all(
            result['selected'] is not None for result in state['selection_by_training_environment'].values())
        state['physical_arm_status'] = ('NOT_YET_AUTHORIZED' if state['both_groups_have_eligible_checkpoint']
            else 'NOT_RUN_TRAINING_GATE' if mode == 'formal' else 'PILOT_DIAGNOSTIC_ONLY')
        state['shared_rule_reference'] = shared_rule_reference(context)
        state.update(status='COMPLETE', elapsed_seconds=budget.elapsed(), runtime=runtime_identity())
        write_results_csv(output / ('PILOT_RESULTS.csv' if mode == 'pilot' else 'TRAINING_RESULTS.csv'), completed)
        atomic_json(output / 'CHECKPOINT_SELECTION.json', dict(selection_contract=selection,
            selection_contract_sha256=object_sha(selection), lock_sha256=lock_sha,
            groups=state['selection_by_training_environment'], physical_arm_status=state['physical_arm_status']))
        atomic_json(state_path, state)
        return state
    except BaseException as exc:
        state.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, WallBudgetExceeded)) else 'FAILED',
            error=dict(type=type(exc).__name__, message=str(exc)), elapsed_seconds=budget.elapsed())
        atomic_json(state_path, state)
        raise


def supervised_run(mode, acceptance_path, output, *, lock_path=None, resume=False, formal_authorization=None):
    """The supervisor enforces wall time even while original PPO update blocks.

    The child checks the same proof again. Each invocation's full startup,
    training, validation, cross-validation and interruption time is accumulated.
    There is no automatic retry or scientific-outcome-dependent extension.
    """
    bindings = verify_s04(acceptance_path)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'benchmark':
        cap = 600
        if resume:
            raise ValueError('engineering probes are never resumed or reused')
    else:
        context, _, digests, _ = load_inputs()
        selection = selection_contract(context, digests)
        lock = read_json(lock_path)
        validate_run_lock(lock, mode, bindings, selection, digests)
        if mode == 'formal':
            verify_formal(formal_authorization, lock)
        cap = lock['total_wall_seconds']
    supervision = output.with_name(output.name + '.SUPERVISION.json')
    previous = read_json(supervision) if supervision.exists() else None
    if previous and not resume:
        raise ValueError('execution record already exists; no fresh-run budget reset')
    elapsed_before = previous['cumulative_elapsed_seconds'] if previous else 0.0
    remaining = cap - elapsed_before
    if remaining <= 0:
        raise WallBudgetExceeded('no cumulative study wall budget remains')
    command = [sys.executable, '-m', 'scripts.maxopt_bridge.ppo_training', '_worker', '--mode', mode,
               '--acceptance', str(Path(acceptance_path).resolve()), '--output', str(output),
               '--elapsed-before', str(elapsed_before)]
    if lock_path:
        command += ['--lock', str(Path(lock_path).resolve())]
    if resume:
        command.append('--resume')
    if formal_authorization:
        command += ['--formal-authorization', str(Path(formal_authorization).resolve())]
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    attempt = len(previous['invocations']) + 1 if previous else 1
    stdout_path = output.with_name(output.name + '.attempt_%02d.stdout.txt' % attempt)
    stderr_path = output.with_name(output.name + '.attempt_%02d.stderr.txt' % attempt)
    start = time.monotonic()
    timed_out = False
    with stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
        process = subprocess.Popen(command, cwd=SOURCE_ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            returncode = process.returncode
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            returncode = process.returncode
            raise
        finally:
            elapsed = time.monotonic() - start
            record = dict(command=command, pid=process.pid, elapsed_seconds=elapsed,
                cumulative_elapsed_seconds=elapsed_before + elapsed, timed_out=timed_out,
                returncode=process.poll(), stdout_path=str(stdout_path), stderr_path=str(stderr_path))
            atomic_json(supervision, dict(mode=mode, total_wall_seconds=cap,
                cumulative_elapsed_seconds=elapsed_before + elapsed,
                invocations=(previous['invocations'] if previous else []) + [record]))
    if timed_out:
        atomic_json(output.with_name(output.name + '.WALL_CAP_TERMINATED.json'), record)
    if returncode != 0:
        raise RuntimeError('CPU worker stopped; preserve logs and partial state before any authorized recovery')
    return read_json(output / ('BENCHMARK_RESULTS.json' if mode == 'benchmark' else 'STUDY_STATE.json'))


def prepare(output):
    """Read-only scientific inputs; emit proposed engineering records, no run."""
    context, _, digests, partitions = load_inputs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = {**BENCHMARK_SPEC, 'data_sha256': digests,
            'common_config_sha256': file_sha(Path(__file__).with_name('ppo_common.json'))}
    path = output / 'BENCHMARK_SPEC.json'
    if path.exists() and read_json(path) != spec:
        raise ValueError('frozen benchmark specification cannot be silently changed')
    atomic_json(path, spec)
    selection = selection_contract(context, digests)
    atomic_json(output / 'SELECTION_CONTRACT.json', selection)
    atomic_json(output / 'DRIVER_PREPARATION.json', dict(state='ENGINEERING_ONLY_NO_RUN',
        benchmark_spec_sha256=file_sha(path), selection_contract_sha256=object_sha(selection),
        driver_sha256=file_sha(__file__), partitions=partitions, training_steps=0, optimizer_steps=0,
        proposed_checkpoint_every_episodes=5, budget='not locked; requires worst-environment measurement plus all-window size margin'))
    return dict(benchmark_spec_sha256=file_sha(path), selection_contract_sha256=object_sha(selection))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'benchmark', 'pilot', 'formal', '_worker'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--formal-authorization', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--mode', choices=['benchmark', 'pilot', 'formal'])
    parser.add_argument('--elapsed-before', type=float, default=0.0)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output)
    elif args.command == '_worker':
        if args.mode == 'benchmark':
            result = benchmark(args.acceptance, args.output)
        else:
            result = run_study(args.mode, args.acceptance, args.lock, args.output,
                resume=args.resume, elapsed_before=args.elapsed_before, formal_authorization=args.formal_authorization)
    else:
        result = supervised_run(args.command, args.acceptance, args.output,
            lock_path=args.lock, resume=args.resume, formal_authorization=args.formal_authorization)
    print(json.dumps(dict(status=result.get('status', 'PREPARED'), output=str(args.output))))


if __name__ == '__main__':
    main()
