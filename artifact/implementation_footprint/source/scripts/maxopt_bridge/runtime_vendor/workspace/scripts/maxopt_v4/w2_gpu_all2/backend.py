"""Generation-tagged accepted worker plus asynchronous, ownership-bound backend."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from . import evidence
    from . import llama_tp1_worker as accepted_worker
except ImportError:
    from scripts.maxopt_v4.w2_gpu_all2 import evidence
    from scripts.maxopt_v4.w2_gpu_all2 import llama_tp1_worker as accepted_worker

RUNTIME_EXPECTED = {
    'effective_dtype': 'torch.bfloat16', 'effective_max_model_len': 4096,
    'effective_tensor_parallel_size': 1, 'effective_pipeline_parallel_size': 1,
    'effective_data_parallel_size': 1, 'effective_enforce_eager': True,
    'effective_gpu_memory_utilization': .8, 'effective_enable_prefix_caching': False,
    'effective_max_num_batched_tokens': 16384, 'effective_max_num_seqs': 1024,
    'effective_enable_chunked_prefill': True,
    'actual_engine_class': 'vllm.v1.engine.async_llm.AsyncLLM',
}


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prepare_start_files(folder):
    """Create a generation directory and exclusive logs outside the controller."""
    folder.mkdir(parents=True)
    stdout = (folder / 'stdout.log').open('xb')
    try:
        return stdout, (folder / 'stderr.log').open('xb')
    except BaseException:
        stdout.close()
        raise


def _persist_ready_evidence(folder, snapshot, members, out):
    """Persist both ready proofs before the controller may publish ``ready``."""
    nvidia = folder / 'nvidia_ready.json'
    groups = folder / 'process_groups_ready.json'
    save(nvidia, snapshot)
    save(groups, members)
    return dict(nvidia_path=nvidia.relative_to(out).as_posix(), nvidia_sha256=sha(nvidia),
                groups_path=groups.relative_to(out).as_posix(), groups_sha256=sha(groups))


def _persist_release_evidence(folder, cleanup, snapshot, out):
    """Persist both release proofs before the controller may publish release."""
    cleanup_path = folder / 'cleanup.json'
    nvidia = folder / 'nvidia_released.json'
    save(cleanup_path, cleanup)
    save(nvidia, snapshot)
    return dict(cleanup_path=cleanup_path.relative_to(out).as_posix(), cleanup_sha256=sha(cleanup_path),
                nvidia_path=nvidia.relative_to(out).as_posix(), nvidia_sha256=sha(nvidia))


def _tail_readline(stream):
    """Read one tail record atomically with its seek-back for a partial line."""
    position = stream.tell()
    line = stream.readline()
    if not line.endswith('\n'):
        stream.seek(position)
    return line


def _close_streams(context):
    for key in ('stdout', 'stderr'):
        stream = context.get(key)
        if stream is not None and not stream.closed:
            stream.close()
    context['closed_files'] = True


async def _thread_operation(callable_, *args, **kwargs):
    """Finish a filesystem operation before a cancelled pump can close its stream."""
    task = asyncio.create_task(asyncio.to_thread(callable_, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


class GenerationWriter:
    def __init__(self, path, generation):
        self.writer = accepted_worker.EventWriter(path)
        self.generation, self.seq = generation, 0

    def __call__(self, event):
        self.writer(dict(event, generation=self.generation, worker_seq=self.seq))
        self.seq += 1

    def close(self):
        self.writer.close()


class AsyncPopenProcess:
    """Async facade over a ``Popen`` whose pipe operations run in a thread.

    ``asyncio``'s ``connect_write_pipe`` is not implemented for the pipe
    transport used by this environment.  Keeping the native ``Popen.stdin``
    handle and offloading write/flush/close preserves controller-clock
    progress without relying on an unsupported transport method.
    """

    def __init__(self, process, stdin):
        self._process = process
        self.stdin = stdin
        self.pid = process.pid

    @property
    def returncode(self):
        return self._process.poll()

    async def wait(self):
        return await asyncio.to_thread(self._process.wait)

    async def send(self, payload):
        """Write and flush without blocking the controller event loop."""
        def write_and_flush():
            self.stdin.write(payload)
            self.stdin.flush()
        await asyncio.to_thread(write_and_flush)

    async def close_stdin(self):
        """Close stdin at an explicit awaitable boundary."""
        await asyncio.to_thread(self.stdin.close)

    def terminate(self):
        return self._process.terminate()

    def kill(self):
        return self._process.kill()


class AsyncWorkerBackend:
    """Each generation has independent startup, pump, writer and release tasks."""
    def __init__(self, controller):
        self.controller = controller
        self.plan, self.out = controller.plan, controller.out
        self.contexts = {}

    def schedule(self, coroutine):
        task = asyncio.create_task(coroutine)
        def finished(done):
            if not done.cancelled() and done.exception() is not None:
                self.controller.fail(done.exception())
        task.add_done_callback(finished)
        return task

    async def before(self):
        return await asyncio.to_thread(evidence.snapshot_gpu, 5)

    async def after(self):
        return await asyncio.to_thread(evidence.snapshot_gpu, 5)

    async def _create_process(self, argv, env, stdout, stderr):
        """Fork/exec in a worker thread so page-table setup cannot stall ticks."""
        process = await asyncio.to_thread(
            subprocess.Popen, argv, env=env, stdin=subprocess.PIPE,
            stdout=stdout, stderr=stderr, start_new_session=True)
        return AsyncPopenProcess(process, process.stdin)

    def launch(self, gpu, generation):
        key = (gpu, generation)
        if key in self.contexts:
            raise ValueError('duplicate generation launch')
        folder = self.out / ('gpu' + str(gpu)) / ('generation_' + format(generation, '04d'))
        context = dict(gpu=gpu, generation=generation, folder=folder, process=None, ownership=None,
                       events=[], pump_done=False, pump=None, writer=None, commands=asyncio.Queue(),
                       release_task=None, closed_files=False)
        self.contexts[key] = context
        context['startup_task'] = self.schedule(self._start(context))

    def shutdown(self, gpu, generation, forced):
        context = self.contexts[(gpu, generation)]
        if context['release_task'] is not None:
            raise ValueError('duplicate generation shutdown')
        context['release_task'] = self.schedule(self._stop(context, forced))

    def submit(self, gpu, generation, row):
        context = self.contexts[(gpu, generation)]
        context['commands'].put_nowait(dict(command='submit', request_id=row['request_id'],
                                           prompt_token_ids=row['prompt_token_ids'],
                                           max_output_tokens=row['max_output_tokens']))

    async def _send(self, context, command):
        process = context['process']
        if process is None or process.returncode is not None:
            raise RuntimeError('cannot write to exited generation')
        self.controller.emit('worker_command', gpu=context['gpu'], generation=context['generation'],
                             pid=process.pid, command=command)
        payload = (json.dumps(command) + '\n').encode()
        if hasattr(process, 'send'):
            await asyncio.wait_for(process.send(payload), timeout=5)
        else:
            process.stdin.write(payload)
            await asyncio.wait_for(process.stdin.drain(), timeout=5)

    async def _terminate_unowned_process(self, context, error, deadline):
        """Boundedly terminate a child before its ownership token exists.

        The group cleanup primitive requires a captured identity token.  This
        path therefore uses only the child handle returned by
        ``create_subprocess_exec`` and never guesses a process group.
        """
        context['ownership_capture_error'] = {
            'error_type': type(error).__name__, 'error': str(error),
        }
        process = context.get('process')
        if process is not None:
            termination = {'pid': process.pid, 'terminate_sent': False,
                           'kill_sent': False, 'wait_timeout': False}
            if process.returncode is None:
                try:
                    process.terminate()
                    termination['terminate_sent'] = True
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(asyncio.shield(process.wait()),
                                           timeout=max(.01, min(5.0, deadline - time.monotonic())))
                except asyncio.TimeoutError:
                    termination['wait_timeout'] = True
                    try:
                        process.kill()
                        termination['kill_sent'] = True
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(asyncio.shield(process.wait()),
                                               timeout=max(.01, min(5.0, deadline - time.monotonic())))
                    except asyncio.TimeoutError:
                        termination['kill_wait_timeout'] = True
            context['unowned_termination'] = termination
        await asyncio.to_thread(_close_streams, context)

    async def _writer(self, context):
        while True:
            command = await context['commands'].get()
            await self._send(context, command)

    async def _pump(self, context):
        path = context['folder'] / 'events.jsonl'
        while not await _thread_operation(path.exists):
            if context['pump_done']:
                return
            await asyncio.sleep(.005)
        stream = await _thread_operation(path.open, encoding='utf-8')
        try:
            # A worker can emit a dense startup burst.  ``sleep(0)`` only
            # requeues this task and can starve the observation clock when
            # every iteration finds another line.  Bound each pump batch and
            # yield for a positive interval so the one-second controller task
            # always gets a scheduling opportunity without changing event
            # order or contents.
            batch = 0
            while True:
                line = await _thread_operation(_tail_readline, stream)
                if not line.endswith('\n'):
                    if context['pump_done']:
                        if line:
                            raise ValueError('truncated generation worker event')
                        return
                    await asyncio.sleep(.005)
                    continue
                event = json.loads(line)
                if (event.get('generation') != context['generation'] or event.get('gpu_index') != context['gpu']
                        or event.get('pid') != context['process'].pid
                        or event.get('worker_seq') != len(context['events'])):
                    raise ValueError('worker identity/sequence changed')
                context['events'].append(event)
                self.controller.receive(context['gpu'], context['generation'], event)
                batch += 1
                if batch >= 32:
                    batch = 0
                    await asyncio.sleep(0.001)
        finally:
            await _thread_operation(stream.close)

    async def _until(self, context, predicate, deadline, description):
        while not predicate():
            self.controller.check_health()
            if context['process'].returncode is not None:
                raise RuntimeError('worker exited while waiting for ' + description)
            if time.monotonic() >= deadline:
                raise TimeoutError(description)
            await asyncio.sleep(.005)

    async def _start(self, context):
        gpu, generation, folder = context['gpu'], context['generation'], context['folder']
        # W2's immutable plan names the complete setup budget
        # ``setup_seconds``; worker startup is one part of that budget.
        deadline = time.monotonic() + self.plan['limits']['setup_seconds']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', VLLM_USE_V1='1')
        argv = [self.plan['python'], '-B', str(Path(__file__).resolve()), '--gpu-index', str(gpu),
                '--generation', str(generation), '--model', self.plan['model'],
                '--events', str(folder / 'events.jsonl')]
        context['stdout'], context['stderr'] = await asyncio.to_thread(_prepare_start_files, folder)
        process = await self._create_process(argv, env, context['stdout'], context['stderr'])
        context['process'] = process
        try:
            context['ownership'] = await asyncio.to_thread(
                evidence.capture_group_ownership, process.pid)
        except BaseException as exc:
            await self._terminate_unowned_process(context, exc, deadline)
            raise
        self.controller.actuator.spawned(gpu, generation, process.pid, context['ownership'], argv=argv,
                                         environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES',
                                             'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
                                             'VLLM_USE_V1', 'VLLM_NO_USAGE_STATS')})
        context['pump'] = self.schedule(self._pump(context))
        context['writer'] = self.schedule(self._writer(context))
        await self._until(context, lambda: any(e['kind'] == 'ready' for e in context['events']), deadline, 'engine ready')
        engine_ready = next(e for e in context['events'] if e['kind'] == 'ready')
        runtime = engine_ready['runtime']
        differences = {k: (v, runtime.get(k)) for k, v in RUNTIME_EXPECTED.items() if runtime.get(k) != v}
        if differences:
            raise ValueError('effective runtime differs: ' + repr(differences))
        rid = 'setup_g' + str(gpu) + '_gen' + str(generation)
        probe = self.plan['setup_probe']['prompt_token_ids']
        await self._send(context, dict(command='submit', request_id=rid, prompt_token_ids=probe, max_output_tokens=32))
        await self._until(context, lambda: any(e['kind'] in ('finished', 'failed', 'cancelled') and e['request_id'] == rid
                                             for e in context['events']), deadline, 'health probe')
        terminal = next(e for e in context['events'] if e['kind'] in ('finished', 'failed', 'cancelled') and e['request_id'] == rid)
        if terminal['kind'] != 'finished' or terminal.get('prompt_token_ids') != probe or len(terminal.get('output_token_ids', [])) != 32:
            raise ValueError('health probe evidence differs')
        marker = len(context['events'])
        await self._send(context, dict(command='status'))
        await self._until(context, lambda: any(e['kind'] == 'status' and e.get('active_count') == 0
                           and e.get('active_request_ids') == [] for e in context['events'][marker:]), deadline, 'empty backend')
        empty = next(e for e in context['events'][marker:] if e['kind'] == 'status' and e.get('active_count') == 0)
        snapshot = await asyncio.to_thread(evidence.snapshot_gpu, 5)
        members = await asyncio.to_thread(evidence.process_group_members, process.pid, True)
        expected_uuid = self.controller.gpu_uuid(gpu)
        rows = snapshot['inventory']['rows']
        apps = snapshot['compute_apps']['rows']
        if (snapshot['inventory']['returncode'] != 0 or snapshot['compute_apps']['returncode'] != 0
                or {row['index']: row['uuid'] for row in rows}.get(gpu) != expected_uuid
                or not any(row['gpu_uuid'] == expected_uuid for row in apps)
                or any(row['pid'] not in {m['pid'] for m in members} for row in apps if row['gpu_uuid'] == expected_uuid)):
            raise ValueError('ready GPU process ownership/identity unproven')
        ready_evidence = await asyncio.to_thread(_persist_ready_evidence, folder, snapshot, members, self.out)
        self.controller.actuator.ready(gpu, generation, health_probe_completed=True, queues_empty=True,
             health_probe_request_id=rid, probe_worker_seq=terminal['worker_seq'],
             empty_status_worker_seq=empty['worker_seq'], runtime=runtime, evidence=ready_evidence)

    async def _stop(self, context, forced):
        deadline = time.monotonic() + self.plan['limits']['release_seconds']
        if getattr(self.controller, 'cleanup_deadline', None) is not None:
            deadline = min(deadline, self.controller.cleanup_deadline-8)
        # Startup is never cancelled by a policy decision. At terminal cleanup only,
        # retain the launch task until it has registered any created process.
        while context['process'] is None and not context['startup_task'].done() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        process = context['process']
        if process is None:
            raise RuntimeError('launch has no owned process; release cannot be asserted')
        startup = context.get('startup_task')
        if startup is not None and not startup.done():
            # Do not race shutdown against the startup coroutine.  It owns the
            # transition from spawned -> ready and event-pump wiring; stopping
            # first can produce a stale ready callback or an unowned process.
            # Shield keeps a budget timeout from cancelling it prematurely.
            try:
                await asyncio.wait_for(asyncio.shield(startup),
                                       timeout=max(.01, deadline-time.monotonic()))
            except asyncio.TimeoutError:
                context['startup_wait_timeout'] = True
                self.controller.fail(TimeoutError(
                    f"startup wait timeout: gpu={context['gpu']} generation={context['generation']}"))
                # The release budget is exhausted.  This is a last-resort
                # technical failure; normal forced cleanup never takes it.
                startup.cancel()
                await asyncio.gather(startup, return_exceptions=True)
            except BaseException as exc:
                context['startup_error'] = {'error_type': type(exc).__name__, 'error': str(exc)}
        if context['writer'] is not None:
            context['writer'].cancel()
            await asyncio.gather(context['writer'], return_exceptions=True)
        if process.returncode is None:
            try:
                await self._send(context, dict(command='shutdown'))
                if hasattr(process, 'close_stdin'):
                    await process.close_stdin()
                else:
                    await asyncio.to_thread(process.stdin.close)
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=min(30, max(.01, deadline-time.monotonic())))
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                pass
        ownership = context.get('ownership')
        if ownership is None:
            # Without an identity token, group signalling is unsafe.  The
            # child was terminated through its direct handle in _start; this
            # branch only inspects residual session members.
            try:
                members_after = await asyncio.to_thread(evidence.process_group_members, process.pid, True)
            except (OSError, ValueError) as exc:
                members_after = [{'inspection_error': type(exc).__name__, 'error': str(exc)}]
            cleanup = {
                'complete': process.returncode is not None and members_after == [],
                'ownership_capture_failed': context.get('ownership_capture_error',
                    {'error_type': 'OwnershipUnavailable', 'error': 'ownership token missing'}),
                'unowned_termination': context.get('unowned_termination'),
                'members_after': members_after, 'signals': [],
                'ownership_unavailable': True,
            }
        else:
            cleanup = await asyncio.to_thread(evidence.cleanup_group, process.pid, deadline, ownership)
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=max(.01, deadline-time.monotonic()))
        except asyncio.TimeoutError:
            cleanup['worker_wait_timeout'] = True
        cleanup['members_after'] = await asyncio.to_thread(evidence.process_group_members, process.pid, True)
        cleanup['worker_returncode'] = process.returncode
        if context.get('startup_wait_timeout'):
            cleanup['startup_wait_timeout'] = True
        if context.get('startup_error'):
            cleanup['startup_error'] = context['startup_error']
        context['pump_done'] = True
        if context['pump'] is not None:
            await context['pump']
        cleanup['worker_closed_event'] = any(e['kind'] == 'closed' for e in context['events'])
        cleanup['forced_cleanup'] = forced
        await asyncio.to_thread(_close_streams, context)
        folder = context['folder']
        snapshot = await asyncio.to_thread(evidence.snapshot_gpu, 5)
        release_evidence = await asyncio.to_thread(
            _persist_release_evidence, folder, cleanup, snapshot, self.out)
        expected_uuid = self.controller.gpu_uuid(context['gpu'])
        row = next((r for r in snapshot['inventory']['rows'] if r['index'] == context['gpu']), {})
        released = (cleanup.get('complete') is True and cleanup['members_after'] == []
                    and process.returncode is not None and snapshot['inventory']['returncode'] == 0
                    and snapshot['compute_apps']['returncode'] == 0 and row.get('uuid') == expected_uuid
                    and type(row.get('memory_used_mib')) is int
                    and row['memory_used_mib'] <= self.controller.baseline_memory[context['gpu']] + 16
                    and not any(r['gpu_uuid'] == expected_uuid for r in snapshot['compute_apps']['rows']))
        if not released:
            raise RuntimeError('owned generation process/GPU release not verified')
        if ownership is None:
            # Physical release is observable, but ownership was never proven.
            # Keep lifecycle state coherent while forcing TECHNICAL_INVALID.
            self.controller.fail(RuntimeError('worker ownership capture failed before spawn registration'))
        if not forced and (process.returncode != 0 or not cleanup['worker_closed_event']):
            self.controller.fail(RuntimeError('normal lifecycle shutdown lacked clean worker exit'))
        self.controller.actuator.released(context['gpu'], context['generation'], release_verified=True,
             **release_evidence, worker_returncode=process.returncode)

    def check_health(self):
        for context in self.contexts.values():
            process = context['process']
            state = self.controller.actuator.devices[context['gpu']]['state']
            if process is not None and process.returncode is not None and context['release_task'] is None and state != 'off':
                raise RuntimeError('unexpected generation worker exit')

    async def finish(self, deadline):
        tasks = [c['release_task'] for c in self.contexts.values() if c['release_task'] is not None]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=max(0, deadline-time.monotonic()))
            if pending:
                raise TimeoutError('owned generation cleanup deadline exhausted')
            for task in done:
                task.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--events', required=True)
    args = parser.parse_args()
    if args.generation < 1:
        parser.error('generation must be positive')
    writer = GenerationWriter(args.events, args.generation)
    try:
        return asyncio.run(accepted_worker.run_main(args, writer))
    finally:
        writer.close()


if __name__ == '__main__':
    raise SystemExit(main())
