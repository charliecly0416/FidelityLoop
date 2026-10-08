"""Two-device lifecycle state machine. Backend operations are scheduled, never awaited."""


class Actuator:
    def __init__(self, emit, launch, shutdown):
        self.emit, self.launch, self.shutdown = emit, launch, shutdown
        self.devices = [dict(state='off', generation=0, pid=None, ownership=None, jobs=set(),
                             first_service=False, ready_seen=False) for _ in range(2)]
        self.target = 2

    def identity(self, gpu):
        device = self.devices[gpu]
        return dict(gpu=gpu, generation=device['generation'], pid=device['pid'], ownership=device['ownership'])

    def event(self, kind, gpu, **fields):
        return self.emit(kind, **self.identity(gpu), **fields)

    def state(self, gpu, new):
        device = self.devices[gpu]
        old = device['state']
        allowed = {'off': {'starting'}, 'starting': {'active', 'stopping'},
                   'active': {'draining', 'stopping'}, 'draining': {'active', 'stopping'},
                   'stopping': {'off'}}
        if new not in allowed[old]:
            raise ValueError('invalid lifecycle transition: ' + old + ' -> ' + new)
        self.event('lifecycle_state', gpu, previous=old, state=new, local_running=len(device['jobs']))
        device['state'] = new

    def start(self, gpu):
        device = self.devices[gpu]
        if device['state'] != 'off' or device['jobs']:
            raise ValueError('launch requires verified off and no jobs')
        device.update(generation=device['generation'] + 1, pid=None, ownership=None, first_service=False, ready_seen=False)
        self.event('launch_issued', gpu)
        self.state(gpu, 'starting')
        self.launch(gpu, device['generation'])

    def spawned(self, gpu, generation, pid, ownership, **fields):
        device = self._generation(gpu, generation)
        if device['state'] not in ('starting', 'stopping') or device['pid'] is not None or type(pid) is not int or pid <= 1:
            raise ValueError('invalid/duplicate worker spawn')
        device.update(pid=pid, ownership=ownership)
        self.event('worker_spawned', gpu, **fields)

    def ready(self, gpu, generation, **evidence):
        device = self._generation(gpu, generation)
        if device['state'] not in ('starting', 'stopping') or device['jobs'] or device['pid'] is None or device['ready_seen']:
            raise ValueError('ready requires starting empty generation')
        if evidence.get('queues_empty') is not True or evidence.get('health_probe_completed') is not True:
            raise ValueError('ready requires completed health probe and empty status')
        device['ready_seen'] = True
        if device['state'] == 'stopping':
            self.event('ready_during_stopping', gpu, **evidence)
            return
        self.event('ready', gpu, **evidence)
        self.state(gpu, 'active')

    def stop(self, gpu, *, forced=False):
        device = self.devices[gpu]
        if device['state'] in ('off', 'stopping'):
            return
        if not forced and (device['state'] != 'draining' or device['jobs']):
            raise ValueError('normal shutdown requires empty draining generation')
        self.event('shutdown_issued', gpu, forced=forced, active_request_ids=sorted(device['jobs']),
                   reason='observation_cutoff_cleanup' if forced else 'drain_completed')
        self.state(gpu, 'stopping')
        self.shutdown(gpu, device['generation'], forced)

    def released(self, gpu, generation, **evidence):
        device = self._generation(gpu, generation)
        if device['state'] != 'stopping' or evidence.get('release_verified') is not True:
            raise ValueError('off requires verified owned process and GPU release')
        self.event('verified_release', gpu, **evidence)
        device['jobs'].clear()
        self.state(gpu, 'off')

    def apply(self, target):
        if type(target) is not int or target not in (0, 1, 2):
            raise ValueError('target must be integer 0..2')
        self.target = target
        count = lambda: sum(d['state'] in ('active', 'starting') for d in self.devices)
        for gpu in (1, 0):
            device = self.devices[gpu]
            if count() > target and device['state'] == 'active':
                self.state(gpu, 'draining')
                if not device['jobs']:
                    self.stop(gpu)
        for gpu, device in enumerate(self.devices):
            if count() < target and device['state'] == 'draining':
                self.event('drain_cancelled', gpu)
                self.state(gpu, 'active')
        for gpu, device in enumerate(self.devices):
            if count() < target and device['state'] == 'off':
                self.start(gpu)

    def dispatch(self, gpu, rid):
        device = self.devices[gpu]
        if device['state'] != 'active' or len(device['jobs']) >= 4 or any(rid in d['jobs'] for d in self.devices):
            raise ValueError('dispatch must use one active slot exactly once')
        device['jobs'].add(rid)
        if not device['first_service']:
            device['first_service'] = True
            self.event('first_service', gpu, request_id=rid)
        return self.identity(gpu)

    def completed(self, gpu, generation, rid):
        device = self._generation(gpu, generation)
        if rid not in device['jobs']:
            raise ValueError('duplicate or unowned completion')
        device['jobs'].remove(rid)
        if device['state'] == 'draining' and not device['jobs']:
            self.stop(gpu)

    def observation(self, tick, queue_online, queue_offline):
        return dict(time=tick, queue_online=queue_online, queue_offline=queue_offline,
                    local_running=sum(len(d['jobs']) for d in self.devices),
                    active=sum(d['state'] == 'active' for d in self.devices),
                    starting=sum(d['state'] == 'starting' for d in self.devices),
                    draining=sum(d['state'] in ('draining', 'stopping') for d in self.devices),
                    off=sum(d['state'] == 'off' for d in self.devices))

    def available(self):
        return [g for g, d in enumerate(self.devices) if d['state'] == 'active' and len(d['jobs']) < 4]

    def _generation(self, gpu, generation):
        device = self.devices[gpu]
        if device['generation'] != generation:
            raise ValueError('stale generation callback')
        return device
