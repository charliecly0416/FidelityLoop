"""Causal deadline protection shared by prediction and physical adapters.

The snapshot contains only released queue rows and observable running/start/stop
ages. It never contains simulator progress, true ready-at, or future arrivals.
"""
import heapq
import math

from scripts.maxopt_v2.engine import TargetPolicy
from scripts.maxopt_v2.calibrated import service_seconds

SCHEMA = 'maxopt-v3-observation-v1'
JOB_FIELDS = {'request_id','job_type','arrival_s','deadline_s','input_tokens','max_output_tokens','started_at'}


def public_job(row, started_at=None):
    return {k: (started_at if k=='started_at' else row[k]) for k in JOB_FIELDS}


def validate_observation(obs):
    if set(obs)!={'schema','time','queue_online','queue_offline','devices'} or obs['schema']!=SCHEMA:
        raise ValueError('unknown observation fields/schema; future data forbidden')
    now=obs['time']
    if type(now) is not int or now<0 or len(obs['devices'])!=2:
        raise ValueError('invalid observation time/devices')
    jobs=[]
    for kind in ('online','offline'):
        jobs.extend(obs['queue_'+kind])
        if any(j['job_type']!=kind or j['started_at'] is not None for j in obs['queue_'+kind]):
            raise ValueError('queue kind/status mismatch')
    for d in obs['devices']:
        if set(d)!={'state','since','jobs'} or d['state'] not in ('active','starting','draining','stopping','off'):
            raise ValueError('unknown device state/fields')
        if not math.isfinite(d['since']) or d['since']>now+1:
            raise ValueError('invalid observed state timestamp')
        if len(d['jobs'])>4 or (d['state'] in ('off','starting','stopping') and d['jobs']):
            raise ValueError('invalid device jobs')
        if any(j['started_at'] is None for j in d['jobs']):
            raise ValueError('missing dispatch time')
        jobs.extend(d['jobs'])
    if len({j['request_id'] for j in jobs})!=len(jobs):
        raise ValueError('duplicate visible job')
    for j in jobs:
        if (set(j)!=JOB_FIELDS or j['arrival_s']>now or j['deadline_s']<j['arrival_s']
                or j['job_type'] not in ('online','offline')
                or (j['started_at'] is not None and not j['arrival_s']<=j['started_at']<=now+1)):
            raise ValueError('invalid/future job observation')


def base_observation(obs):
    return dict(time=obs['time'],queue_online=len(obs['queue_online']),queue_offline=len(obs['queue_offline']),
                local_running=sum(len(d['jobs']) for d in obs['devices']),
                **{s:sum(d['state']==s for d in obs['devices']) for s in ('active','starting','off')},
                draining=sum(d['state'] in ('draining','stopping') for d in obs['devices']))


class DeadlinePolicy:
    def __init__(self, spec, model, safety, *, scale=True, reservation=True, margin=5.):
        self.base=TargetPolicy(spec)
        self.base.state['target']=2
        self.spec=self.base.spec
        self.model,self.safety=model,safety
        self.scale,self.reservation,self.margin=scale,reservation,margin
        self.latched=set()

    @property
    def state(self):
        return self.base.state

    def duration(self,j):
        return service_seconds(self.model,j,4)

    def remaining(self,j,now):
        full=self.duration(j)
        # Past the bound, retain a positive full-service fallback: no zero ETA lie.
        left=full-(now-j['started_at'])
        return left if left>0 else full

    def delay(self,d,now):
        if d['state'] in ('active','draining'):
            return 0.
        start=self.safety['startup_safe_seconds']
        if d['state']=='starting':
            elapsed=max(0,now-d['since'])
            maximum=self.safety['startup_max_seconds']
            return max(1.,start-elapsed) if elapsed<start else max(1.,maximum-elapsed) if elapsed<maximum else start
        if d['state']=='stopping':
            left=self.safety['shutdown_seconds']-max(0,now-d['since'])
            return (left if left>0 else self.safety['shutdown_seconds'])+start
        return start

    def projected(self,obs,k,reserve):
        """Conservative fixed-concurrency list schedule, not future outcome access."""
        if k==0:
            return {j['request_id']:math.inf for j in obs['queue_offline']}
        now=obs['time']
        devices=sorted(enumerate(obs['devices']),key=lambda item:(self.delay(item[1],now),item[0]))[:k]
        slots=[]
        for gpu,d in devices:
            jobs=d['jobs']
            for i in range(4):
                delay=self.remaining(jobs[i],now) if i<len(jobs) else 0.
                slots.append((now+self.delay(d,now)+delay,gpu,i))
        heapq.heapify(slots)
        finish={}
        offline=sorted(obs['queue_offline'],key=lambda j:(j['deadline_s'],j['request_id']))
        if reserve and obs['queue_online']:
            # Protect one concurrent offline slot. If one is occupied already,
            # new risk jobs wait for it; the other slots remain online-first.
            offline_slots=[(now+self.remaining(j,now),gpu,i) for gpu,d in devices
                           for i,j in enumerate(d['jobs']) if j['job_type']=='offline']
            at,_,_=min(offline_slots or slots)
            for j in offline:
                at+=self.duration(j)
                finish[j['request_id']]=at
            return finish
        # No-reservation predicts current online backlog ahead of offline. Future
        # arrivals remain unknown. Reservation guarantees only the next free slot.
        order=offline+obs['queue_online'] if reserve else obs['queue_online']+offline
        for j in order:
            at,gpu,slot=heapq.heappop(slots)
            end=at+self.duration(j)
            heapq.heappush(slots,(end,gpu,slot))
            if j['job_type']=='offline':
                finish[j['request_id']]=end
        return finish

    def decide(self,obs):
        validate_observation(obs)
        before=dict(self.base.state)
        base=self.base.decide(base_observation(obs))
        pending=obs['queue_offline']
        ids={j['request_id'] for j in pending}
        self.latched.intersection_update(ids)
        projected=self.projected(obs,max(1,base),self.reservation)
        # Trigger before waiting another tick would make the conservative finish late.
        risky={j['request_id'] for j in pending if projected[j['request_id']]+self.margin+1>=j['deadline_s']}
        self.latched.update(risky)
        active=bool(self.latched)
        target=base
        feasible=[]
        if active and self.scale:
            for k in (1,2):
                f=self.projected(obs,k,self.reservation)
                if all(f[j['request_id']]+self.margin<=j['deadline_s'] for j in pending):
                    feasible.append(k)
            target=max(base,min(feasible) if feasible else 2)
            if target!=base:
                self.base.state.update(target=target,last_change_at=obs['time'])
        return dict(base_target=base,target=target,reserve=active and self.reservation,
                    risky_ids=sorted(self.latched),infeasible=active and self.scale and not feasible,
                    policy_state_before=before,policy_state_after=dict(self.base.state),
                    oldest_offline_age=max((obs['time']-j['arrival_s'] for j in pending),default=0),
                    min_offline_slack=min((j['deadline_s']-obs['time'] for j in pending),default=None),
                    offline_work_remaining=sum(self.duration(j) for j in pending)
                        +sum(self.remaining(j,obs['time']) for d in obs['devices'] for j in d['jobs'] if j['job_type']=='offline'),
                    ready_eta_safe=[self.delay(d,obs['time']) for d in obs['devices']],
                    schema=SCHEMA)

    def dispatch_kind(self,obs,reserve):
        """Select next free slot without preempting anything already running."""
        online,offline=obs['queue_online'],obs['queue_offline']
        if not online:
            return 'offline' if offline else None
        if not offline or not reserve:
            return 'online'
        if any(j['job_type']=='offline' for d in obs['devices'] for j in d['jobs']):
            return 'online'
        now=obs['time']
        on=online[0]['deadline_s']-now-self.duration(online[0])
        off=offline[0]['deadline_s']-now-self.duration(offline[0])
        return 'online' if on<=self.margin and on<=off else 'offline'
