"""Explicit GPU-site S5a entry: six restart probes or one frozen short canary.

CPU only imports are safe. `run` launches real workers and is not run on CPU.
"""
import argparse
import asyncio
import sys
import time
import statistics
from pathlib import Path

from .runtime_adapter import Controller, AcceptedController, supervised_run, local_model_check
from fidelityloop.legacy.maxopt_v2.engine import TargetPolicy
from fidelityloop.legacy.maxopt_v2.calibrated import service_seconds
from scripts.maxopt_n5.verify import prior, verify_generations, verify_projection
from .physical_ledger import reduce_lifecycle
from .policy import DeadlinePolicy
from .common import read,lines,save,digest
from .freeze import verify_lock


def prepare(locked,out,kind,environment,identities):
    verify_lock(locked)
    matrix=read(locked/'matrix.json')
    arm=next(a for a in matrix['arms'] if a['scale'] and a['reservation'])
    env=read(environment);expected=read(locked/'expected_environment.json')
    if ({r['name']:r['sha256'] for r in env['model_files']}!={r['name']:r['sha256'] for r in expected['model_files']}
            or env['packages']!=expected['packages']):
        raise ValueError('environment mismatch; HOLD')
    out.mkdir(parents=True,exist_ok=False)
    rows=[] if kind=='drift' else read(locked/'engineering_inputs'/(kind+'.json'))
    save(out/'expected_requests.json',rows);save(out/'environment_model_snapshot.json',env)
    plan=dict(read(locked/'runtime_template.json'),schema='maxopt-v3-engineering-v1',kind=kind,
         run_id='V3_S5a_'+kind,locked_root=str(locked.resolve()),arm=arm,safety=matrix['safety'],
         guard_model=matrix['guard_models']['E'],window_seconds=600,drain_seconds=300,
         python=sys.executable,model=env['model_path'],gpu_identity_expected=read(identities),
         prediction_lock_sha256=digest(locked/'prediction_lock.json'),expected_requests_sha256=digest(out/'expected_requests.json'))
    save(out/'run_plan.json',plan)


async def wait_until(controller,predicate,deadline):
    while not predicate():
        controller.check_health()
        if time.monotonic()>=deadline:
            raise TimeoutError('S5a engineering budget exceeded')
        await asyncio.sleep(.01)


async def execute(out):
    plan=read(out/'run_plan.json');locked=Path(plan['locked_root'])
    verify_lock(locked);local_model_check(read(out/'environment_model_snapshot.json'))
    if digest(out/'expected_requests.json')!=plan['expected_requests_sha256']:
        raise ValueError('engineering requests changed')
    matrix=read(locked/'matrix.json')
    if plan['guard_model']!=matrix['guard_models']['E'] or plan['safety']!=matrix['safety']:
        raise ValueError('engineering guard differs from frozen parameters')
    drift=plan['kind']=='drift'
    if not drift and read(out/'expected_requests.json')!=read(locked/'engineering_inputs'/(plan['kind']+'.json')):
        raise ValueError('canary input binding')
    if drift:
        c=AcceptedController(plan,out,policy=TargetPolicy('all2'),formal=False)
    else:
        c=Controller(plan,out,engineering=True)
    clean=False
    started=time.monotonic()
    try:
        await c.setup()
        c.start_window(seconds=3600 if drift else 900)
        if drift:
            deadline=started+3480  # Reserve the final two minutes for cleanup.
            for pause in (60,120,300):
                c.actuator.apply(0)
                await wait_until(c,lambda:all(d['state']=='off' for d in c.actuator.devices),deadline)
                for remaining in range(pause):
                    c.check_health()
                    if time.monotonic()+1>=deadline:raise TimeoutError('drift wall cap')
                    await asyncio.sleep(1)
                c.actuator.apply(2)
                await wait_until(c,lambda:all(d['state']=='active' for d in c.actuator.devices),deadline)
            c.actuator.apply(0)
            await wait_until(c,lambda:all(d['state']=='off' for d in c.actuator.devices),deadline)
            c.cutoff_ns=time.monotonic_ns()
            c.close_window()
        else:
            for tick in range(900):
                due=c.origin_ns+tick*10**9
                while time.monotonic_ns()<due:
                    await asyncio.sleep(min(.01,(due-time.monotonic_ns())/1e9))
                if time.monotonic_ns()>=due+10**9:raise RuntimeError('canary tick missed')
                c.tick(tick)
            await wait_until(c,lambda:time.monotonic_ns()>=c.cutoff_ns,started+2220)
            c.close_window()
    except BaseException as exc:
        c.fail(exc)
        if c.origin_ns is not None:c.close_window()
    finally:
        try:clean=await c.cleanup()
        except BaseException as exc:c.fail(exc)
        save(out/'engineering_result.json',dict(technical_valid=not c.errors and clean,errors=c.errors,
            cleanup_complete=clean,kind=plan['kind'],wall_seconds=time.monotonic()-started,
            origin_ns=c.origin_ns,cutoff_ns=c.cutoff_ns,plan_sha256=digest(out/'run_plan.json')))
        c.raw.close();c.ledger.close()


def verify(out,locked):
    verify_lock(locked)
    plan=read(out/'run_plan.json');result=read(out/'engineering_result.json')
    events=lines(out/'controller_events.jsonl');requests=lines(out/'request_events.jsonl');rows=read(out/'expected_requests.json')
    audit=prior.Audit()
    audit.check(result['technical_valid'] and result['cleanup_complete'],'engineering_technical_valid')
    audit.check(result['plan_sha256']==digest(out/'run_plan.json'),'engineering_plan_SHA')
    origin,cutoff=result['origin_ns'],result['cutoff_ns']
    hashes={}
    verify_generations(audit,out,plan,read(out/'environment_model_snapshot.json'),events,hashes,rows,cutoff)
    decision=dict(status='HOLD',kind=plan['kind'],checks=audit.result['checks'],raw_sha256=hashes)
    if plan['kind']=='drift':
        starts={(e['gpu'],e['generation']):e['controller_monotonic_ns'] for e in events if e['kind']=='launch_issued'}
        stops={(e['gpu'],e['generation']):e['controller_monotonic_ns'] for e in events if e['kind']=='shutdown_issued'}
        ready=[(e['controller_monotonic_ns']-starts[e['gpu'],e['generation']])/1e9 for e in events if e['kind']=='ready' and e['generation']>1]
        shutdown=[(e['controller_monotonic_ns']-stops[e['gpu'],e['generation']])/1e9 for e in events if e['kind']=='verified_release' and e['generation']>1]
        begun={};services=[]
        for e in events:
            if e['kind']!='worker_receipt' or e['generation']<=1:continue
            w=e['worker_event'];rid=w.get('request_id')
            if w['kind']=='submit_begin':begun[rid]=w['worker_monotonic_ns']
            if w['kind']=='finished' and rid in begun:services.append((w['worker_monotonic_ns']-begun[rid])/1e9)
        model=read(locked/'matrix.json')['guard_models']['E']
        audit.check(len(ready)==len(shutdown)==len(services)==6,'six_restart_cycles')
        if len(ready)==len(shutdown)==len(services)==6:
            expected=service_seconds(model,dict(input_tokens=128,max_output_tokens=32),1)
            decision.update(startup_median=statistics.median(ready),shutdown_median=statistics.median(shutdown),service_median=statistics.median(services))
            audit.check(abs(statistics.median(ready)/model['startup_seconds']-1)<=.2,'startup_drift_20pct')
            audit.check(abs(statistics.median(shutdown)/model['shutdown_seconds']-1)<=.2,'shutdown_drift_20pct')
            audit.check(abs(statistics.median(services)-expected)<=.5,'service_drift_half_second')
        audit.check(result['wall_seconds']<=3600,'drift_wall_cap')
    else:
        verify_projection(audit,rows,events,requests,origin,cutoff_ns=cutoff)
        guard=DeadlinePolicy(plan['arm']['policy'],plan['guard_model'],plan['safety'])
        ledger=reduce_lifecycle(rows,events,requests,origin_ns=origin,cutoff_ns=cutoff,guard=guard,require_ticks=False)
        audit.check(ledger['status']=='PASS','canary_V3_policy_ledger',ledger['technical_errors'])
        ticks=[e for e in events if e['kind']=='policy_tick']
        audit.check([e['tick'] for e in ticks]==list(range(900)) and cutoff-origin==900*10**9,'900_real_canary_ticks')
        decision.update(guard_ticks=sum(bool(e['guard']['risky_ids']) for e in ticks),
                        reservation_ticks=sum(e['guard']['reserve'] for e in ticks),
                        api_accepts=sum(e['kind']=='api_accept' for e in events))
    decision['status']='PASS' if all(r['passed'] for r in audit.result['checks']) else 'HOLD'
    return decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','check'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--locked',type=Path);p.add_argument('--kind',choices=('drift','canary_sparse','canary_recovery'))
    p.add_argument('--environment',type=Path);p.add_argument('--identities',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.locked,a.output,a.kind,a.environment,a.identities)
    elif a.action=='run':
        from types import SimpleNamespace
        asyncio.run(supervised_run(SimpleNamespace(run=lambda:execute(a.output))))
    else:
        report=verify(a.output,a.locked);save(a.output/'engineering_verification.json',report);print(report['status'])
