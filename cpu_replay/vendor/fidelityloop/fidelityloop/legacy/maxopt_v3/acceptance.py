"""Portable V3 physical RAW audit: shared generation checks plus V3 causal ledger."""
import argparse
from pathlib import Path

from . import runtime_adapter
from scripts.maxopt_n5.verify import prior, verify_generations, verify_projection
from fidelityloop.legacy.maxopt_v2.engine import TargetPolicy
from .physical_ledger import reduce_lifecycle, reduce_requests
from .policy import DeadlinePolicy
from .common import read,lines,digest,save
from .freeze import verify_lock


def verify(out,locked):
    out,locked=Path(out),Path(locked)
    audit=prior.Audit();check=audit.check
    report=audit.result
    report.update(schema='maxopt-v3-physical-verification-v1',status='TECHNICAL_INVALID',raw_sha256={})
    try:
        verify_lock(locked)
        plan=read(out/'run_plan.json');result=read(out/'run_result.json')
        matrix=read(locked/'matrix.json')
        cell=next(c for c in matrix['runs'] if c['run_id']==plan['run_id'])
        arm=next(a for a in matrix['arms'] if a['id']==cell['policy'])
        check(plan['schema']=='maxopt-v3-run-v1' and plan['arm']==arm,'V3_frozen_arm')
        check(plan['safety']==matrix['safety'] and plan['guard_model']==matrix['guard_models'][arm['guard_model']], 'frozen_guard_parameters')
        expected=read(out/'expected_requests.json');events=lines(out/'controller_events.jsonl');requests=lines(out/'request_events.jsonl')
        check(expected==lines(locked/'inputs'/(cell['window']+'.jsonl')),'frozen_request_identity')
        check(digest(out/'expected_requests.json')==plan['expected_requests_sha256'],'request_SHA')
        check(result['plan_sha256']==digest(out/'run_plan.json'),'executed_plan_SHA')
        check(result['technical_valid'] and result['cleanup_complete'] and not result['technical_errors'],'technical_result')
        check(not any(e['kind'] in ('technical_failure','exception_traceback','cleanup_failure') for e in events),'RAW_errors')
        lock=read(locked/'prediction_lock.json')
        for name,expected_hash in lock['code'].items():
            check(digest(out/'source_snapshot'/name)==expected_hash,'source_snapshot:'+name)
        model=read(out/'environment_model_snapshot.json')
        expected_model=read(locked/'expected_environment.json')
        check({r['name']:r['sha256'] for r in model['model_files']}=={r['name']:r['sha256'] for r in expected_model['model_files']},'same_weight_tokenizer_revision')
        check(model['packages']==expected_model['packages'],'runtime_packages_match')
        start=audit.one(events,'observation_start');end=audit.one(events,'observation_end')
        origin=start['origin_monotonic_ns'];cutoff=origin+2100*10**9
        check(result['origin_monotonic_ns']==origin and result['cutoff_monotonic_ns']==cutoff,'observation_clock')
        observed=(end['controller_monotonic_ns']-origin)/1e9
        check(observed>=2100,'full_observation')
        setup=audit.one(events,'setup_start');barrier=audit.one(events,'local_ready_barrier')
        cleanup=audit.one(events,'cleanup_start');cleaned=audit.one(events,'cleanup_complete')
        check(setup['controller_monotonic_ns']<=barrier['controller_monotonic_ns']<=origin
              and barrier['queues_empty'] and (barrier['controller_monotonic_ns']-setup['controller_monotonic_ns'])/1e9<=1200,'dual_ready_barrier')
        check(cleaned['complete'] and cutoff<=cleanup['controller_monotonic_ns']<=cleaned['controller_monotonic_ns']
              and (cleaned['controller_monotonic_ns']-cleanup['controller_monotonic_ns'])/1e9<=120,'owned_cleanup_budget')
        verify_projection(audit,expected,events,requests,origin)
        report['generations']=verify_generations(audit,out,plan,model,events,report['raw_sha256'],expected,cutoff)
        request_summary=reduce_requests(expected,requests,observed,run_errors=result['technical_errors'])
        check(request_summary['technical_valid'],'request_conservation')
        if arm['scale'] or arm['reservation']:
            policy=None;guard=DeadlinePolicy(arm['policy'],plan['guard_model'],plan['safety'],scale=arm['scale'],reservation=arm['reservation'])
        else:
            policy=TargetPolicy(arm['policy']);guard=None
        lifecycle=reduce_lifecycle(expected,events,requests,origin_ns=origin,cutoff_ns=cutoff,policy=policy,guard=guard)
        check(lifecycle['status']=='PASS','V3_RAW_policy_admission_lifecycle',lifecycle['technical_errors'])
        report.update(lifecycle=lifecycle,request_summary=request_summary,run_id=cell['run_id'])
        # Strong G-F and P-1 computed directly from immutable rows + terminals.
        terminal={r['request_id']:r for r in requests if r['kind']=='terminal'}
        counts={}
        for kind in ('online','offline'):
            group=[r for r in expected if r['job_type']==kind]
            ontime=sum(terminal[r['request_id']]['status']=='completed' and terminal[r['request_id']]['at_s']<=r['deadline_s'] for r in group)
            counts[kind]=dict(arrivals=len(group),on_time=ontime,
                unfinished=sum(terminal[r['request_id']]['status']!='completed' for r in group))
        report['populations']=counts
        report['G_F']=all(v['arrivals']>0 and v['on_time']==v['arrivals'] for v in counts.values())
        report['P_1']=(counts['offline']['arrivals']==counts['offline']['on_time']==120
                       and counts['online']['arrivals']>0 and counts['online']['on_time']>=.99*counts['online']['arrivals']
                       and all(v['unfinished']==0 for v in counts.values()))
        report['status']='PASS' if all(r['passed'] for r in report['checks']) else 'TECHNICAL_INVALID'
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name not in ('verification.json',):
                report['raw_sha256'][str(p.relative_to(out))]=digest(p)
    except Exception as exc:
        check(False,'malformed_or_incomplete',repr(exc))
        report['status']='TECHNICAL_INVALID'
    report['errors']=[r for r in report['checks'] if not r['passed']]
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--locked',type=Path,required=True)
    args=parser.parse_args();report=verify(args.output,args.locked)
    save(args.output/'verification.json',report)
    print(report['status'])
    raise SystemExit(0 if report['status']=='PASS' else 2)
