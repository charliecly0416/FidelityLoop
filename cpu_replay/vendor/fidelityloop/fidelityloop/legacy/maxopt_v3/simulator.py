"""V3 prediction: frozen V2 physics with shared causal protection/admission."""
from fidelityloop.legacy.maxopt_v2.calibrated import CalibratedSimulator
from .policy import DeadlinePolicy, SCHEMA, public_job


class Simulator(CalibratedSimulator):
    def __init__(self, rows, contract, model, policy, *, safety, guard_model=None,
                 scale=False, reservation=False, allow_locked_test=False):
        super().__init__(rows,contract,model,policy,allow_locked_test=allow_locked_test)
        self.guard=DeadlinePolicy(policy,guard_model or model,safety,scale=scale,reservation=reservation)
        self.enabled=scale or reservation
        self.since=[0.,0.]

    def state(self,gpu,new,due=None):
        super().state(gpu,new,due)
        self.since[gpu]=self.now

    def observation(self,tick):
        return dict(schema=SCHEMA,time=tick,
                    **{'queue_'+kind:[public_job(self.jobs[r]['row']) for r in self.queues[kind]] for kind in ('online','offline')},
                    devices=[dict(state=d['state'],since=self.since[g],
                             jobs=[public_job(self.jobs[r]['row'],self.jobs[r]['dispatched_at']) for r in d['jobs']])
                             for g,d in enumerate(self.devices)])

    def tick(self,tick):
        if not self.enabled:
            return super().tick(tick)
        while self.cursor<len(self.rows) and self.rows[self.cursor]['arrival_s']<=tick:
            row=self.rows[self.cursor]
            self.queues[row['job_type']].append(row['request_id'])
            self.emit('release',request_id=row['request_id'])
            self.cursor+=1
        obs=self.observation(tick)
        decision=self.guard.decide(obs)
        self.emit('decision',target=decision['target'],observation=obs,guard=decision)
        self.actuate(decision['target'])
        # Each reservation consumes one free slot; re-evaluate urgency afterward.
        while any(d['state']=='active' and len(d['jobs'])<4 for d in self.devices):
            kind=self.guard.dispatch_kind(self.observation(tick),decision['reserve'])
            if kind is None:
                break
            rid=self.queues[kind].popleft()
            gpu=min((g for g,d in enumerate(self.devices) if d['state']=='active' and len(d['jobs'])<4),
                    key=lambda g:(len(self.devices[g]['jobs']),g))
            self.devices[gpu]['jobs'][rid]=1.
            self.jobs[rid].update(dispatched_at=self.now,route=gpu)
            self.emit('dispatch',request_id=rid,route=gpu)
        queue=self.queues['online']
        while queue and len(self.api)<8 and self.now-self.jobs[queue[0]]['row']['arrival_s']>=10:
            rid=queue.popleft()
            self.jobs[rid].update(dispatched_at=self.now,route='synthetic_api')
            self.api[rid]=self.now+5
            self.api_ids.append(rid)
            self.emit('dispatch',request_id=rid,route='synthetic_api')

    def run(self):
        result=super().run()
        # Setup/cleanup are proxies, never fitted into window occupancy.
        result['summary']['outside_window']['evidence']='separate deployment proxy, not per-generation calibration target'
        return result
