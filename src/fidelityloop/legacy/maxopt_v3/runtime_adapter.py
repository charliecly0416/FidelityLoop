"""V3 physical controller; GPU imports/launches occur only in explicit run CLI.

Accepted N5 process ownership and asynchronous worker are isolated under vendor;
V3 replaces policy/observation/admission, not process termination semantics.
"""
import argparse
import asyncio
from pathlib import Path
import time
import scripts

VENDOR=Path(__file__).resolve().parent/'vendor/scripts'
if str(VENDOR) not in scripts.__path__:
    scripts.__path__.append(str(VENDOR))
from scripts.maxopt_n5.runner import Controller as AcceptedController, supervised_run, local_model_check
from scripts.maxopt_n5.worker import AsyncWorkerBackend
from .policy import DeadlinePolicy, SCHEMA, public_job
from .common import read, lines, digest


class Controller(AcceptedController):
    def __init__(self,plan,out,*,backend_factory=AsyncWorkerBackend,engineering=False):
        if plan.get('schema')!=('maxopt-v3-engineering-v1' if engineering else 'maxopt-v3-run-v1'):
            raise ValueError('V3 plan required')
        arm=plan['arm']
        self.guard=DeadlinePolicy(arm['policy'],plan['guard_model'],plan['safety'],
                                  scale=arm['scale'],reservation=arm['reservation'])
        self.enabled=arm['scale'] or arm['reservation']
        self.state_times=[0,0]
        self.dispatch_times={}
        self.tick_number=0
        self.decision=None
        super().__init__(plan,out,backend_factory=backend_factory,policy=self.guard.base,formal=False)
        offline=[r for r in self.expected if r['job_type']=='offline']
        if not engineering and (plan['window_seconds']!=1800 or plan['drain_seconds']!=300 or len(offline)!=120
            or [r['arrival_s'] for r in offline]!=list(range(0,1800,15))
            or any(r['input_tokens']!=256 or r['max_output_tokens']!=256 for r in offline)):
            self.raw.close();self.ledger.close()
            raise ValueError('V3 immutable workload/horizon mismatch')
        self.formal=not engineering

    def emit(self,kind,**fields):
        e=super().emit(kind,**fields)
        if kind=='lifecycle_state':
            self.state_times[e['gpu']]=e['controller_monotonic_ns']
        elif kind=='local_dispatch':
            self.dispatch_times[e['request_id']]=e['controller_monotonic_ns']
        return e

    def causal_observation(self,tick):
        return dict(schema=SCHEMA,time=tick,
            **{'queue_'+kind:[public_job(self.by_id[r]) for r in self.queues[kind]] for kind in ('online','offline')},
            devices=[dict(state=d['state'],since=(self.state_times[g]-self.origin_ns)/1e9,
                          jobs=[public_job(self.by_id[r],(t-self.origin_ns)/1e9)
                                for r,t in self.dispatch_times.items() if r in d['jobs']])
                     for g,d in enumerate(self.actuator.devices)])

    def tick(self,tick,*,target_override=None):
        if target_override is not None:
            raise ValueError('no formal scripted targets')
        if not self.enabled:
            return super().tick(tick)
        self.check_health()
        if self.window_closed or time.monotonic_ns()>=self.cutoff_ns:
            raise ValueError('no admission after observation cutoff')
        self.tick_number=tick
        while self.cursor<len(self.expected) and self.expected[self.cursor]['arrival_s']<=tick:
            row=self.expected[self.cursor];rid=row['request_id']
            self.queues[row['job_type']].append(rid);self.released.add(rid)
            self.record('release',rid,self.emit('request_release',request_id=rid,scheduled_arrival_s=row['arrival_s']))
            self.cursor+=1
        obs=self.causal_observation(tick)
        self.decision=self.guard.decide(obs)
        self.emit('policy_tick',tick=tick,observation=obs,target=self.decision['target'],guard=self.decision,
                  policy_state_before=self.decision['policy_state_before'],
                  policy_state_after=self.decision['policy_state_after'],scripted=False)
        self.actuator.apply(self.decision['target'])
        self.admit()
        self.emit('admission_complete',tick=tick)

    def admit(self):
        if not self.enabled:
            return super().admit()
        while self.actuator.available() and time.monotonic_ns()<self.cutoff_ns:
            kind=self.guard.dispatch_kind(self.causal_observation(self.tick_number),self.decision['reserve'])
            if kind is None:
                break
            gpu=min(self.actuator.available(),key=lambda g:(len(self.actuator.devices[g]['jobs']),g))
            rid=self.queues[kind].popleft();row=self.by_id[rid]
            identity=self.actuator.dispatch(gpu,rid)
            self.local[rid]=dict(identity)
            raw=self.emit('local_dispatch',request_id=rid,input_tokens=row['input_tokens'],
                          output_budget=row['max_output_tokens'],**identity)
            self.record('dispatch',rid,raw,route='gpu'+str(gpu),**identity)
            self.backend.submit(gpu,identity['generation'],row)
        queue=self.queues['online']
        while queue and len(self.api_active)<8:
            now=time.monotonic_ns()
            if now>=self.cutoff_ns or (now-self.origin_ns)/1e9-self.by_id[queue[0]]['arrival_s']<10:
                break
            self.accept_api(queue.popleft())


def preflight(out):
    from .freeze import verify_lock
    plan=read(out/'run_plan.json')
    locked=Path(plan['locked_root'])
    verify_lock(locked)
    matrix=read(locked/'matrix.json')
    cell=next(c for c in matrix['runs'] if c['run_id']==plan['run_id'])
    arm=next(a for a in matrix['arms'] if a['id']==cell['policy'])
    if plan['arm']!=arm or plan['safety']!=matrix['safety'] or plan['guard_model']!=matrix['guard_models'][arm['guard_model']]:
        raise ValueError('plan policy/safety differs from frozen cell')
    if read(out/'expected_requests.json')!=lines(locked/'inputs'/(cell['window']+'.jsonl')):
        raise ValueError('runtime requests differ from frozen window')
    if digest(out/'expected_requests.json')!=plan['expected_requests_sha256']:
        raise ValueError('request binding')
    env=read(out/'environment_model_snapshot.json');expected=read(locked/'expected_environment.json')
    if ({r['name']:r['sha256'] for r in env['model_files']}!={r['name']:r['sha256'] for r in expected['model_files']}
            or env['packages']!=expected['packages']):
        raise ValueError('runtime/weight identity mismatch')
    local_model_check(read(out/'environment_model_snapshot.json'))
    if plan['model']!=read(out/'environment_model_snapshot.json')['model_path']:
        raise ValueError('model path binding')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=preflight(args.output)
    result=asyncio.run(supervised_run(Controller(plan,args.output)))
    return 0 if result['technical_valid'] else 2


if __name__=='__main__':
    raise SystemExit(main())
