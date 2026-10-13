"""S3 common candidate pool on six old development windows; freeze selection."""
import copy
import statistics
import time

from fidelityloop.legacy.maxopt_v2.engine import fingerprint
from fidelityloop.legacy.maxopt_v2.reselect import configurations
from fidelityloop.legacy.maxopt_v2.calibrated import quantile
from .common import OUT, DEV, H, H_ID, read, lines, save, digest, verify_protection
from .simulator import Simulator
from .audit import audit


def candidates():
    base=[dict(**c,scale=False,reservation=False) for c in configurations()]
    return base+[dict(id=c['id']+'__guard',policy=c['policy'],scale=True,reservation=True)
                 for c in configurations() if c['policy']['name'] not in ('all1','all2')]


def feasible(summary):
    p=summary['populations']
    return (p['offline']['arrivals']==120 and p['offline']['on_time']==120
            and p['online']['arrivals']>0 and p['online']['on_time']>=.99*p['online']['arrivals']
            and summary['censored']==0)


def choose(records,specs):
    ranking=[]
    for spec in specs:
        rows=[r for r in records if r['configuration']==spec['id'] and r['window'].startswith('validation_')]
        if len(rows)!=3 or len({r['window'] for r in rows})!=3:
            raise ValueError('selection requires exactly three old validation windows')
        ranking.append(dict(**spec,feasible=all(feasible(r['summary']) for r in rows),
            mean_cost=statistics.mean(r['summary']['costs']['total'] for r in rows),
            worst_offline_slack=min(r['worst_offline_slack'] for r in rows),
            failed_windows=[r['window'] for r in rows if not feasible(r['summary'])]))
    ranking.sort(key=lambda r:(not r['feasible'],r['mean_cost'],-r['worst_offline_slack'],r['id']))
    valid=[r for r in ranking if r['feasible'] and r['policy']['name'] not in ('all1','all2')]
    return dict(selected=valid[0] if valid else None,ranking=ranking)


def run():
    verify_protection()
    start=time.monotonic()
    models=read(OUT/'s2/models.json');safety=read(OUT/'s1/startup_model_v3.json')
    contract=read(DEV/'inputs/contract.json')
    gate=read(OUT/'s2/gate.json')
    inputs={p.stem:lines(p) for p in sorted((DEV/'inputs/C2').glob('*.jsonl'))}
    specs=candidates()
    records,cache=[],{}
    count=0
    def execute(model_name,spec,w):
        nonlocal count
        model=models[model_name]
        key_model=copy.deepcopy(model)
        # Fixed policies never launch; all2 never stops within the observation.
        # Cache only window results. Outside-window proxies must not be reused.
        if spec['policy']['name'] in ('all1','all2'):
            key_model['startup_seconds']=0
            if spec['policy']['name']=='all2':
                key_model['shutdown_seconds']=0
        key=fingerprint([key_model,spec['policy'],spec['scale'],spec['reservation'],w])
        if key not in cache:
            if count>=1500 or time.monotonic()-start>8*3600:
                raise RuntimeError('S3 rollout/wall budget exceeded')
            result=Simulator(inputs[w],contract,model,spec['policy'],safety=safety,
                             scale=spec['scale'],reservation=spec['reservation']).run()
            verification=audit(inputs[w],result['events'],result['summary'],contract,model,spec['policy'],result['requests'],
                               safety=safety,scale=spec['scale'],reservation=spec['reservation'])
            summary=copy.deepcopy(result['summary'])
            summary.pop('outside_window')
            offline=[r for r in inputs[w] if r['job_type']=='offline']
            slack=min((r['deadline_s']-result['requests'][r['request_id']]['completed_at']
                       if result['requests'][r['request_id']]['completed_at'] is not None else -2100) for r in offline)
            online=[result['requests'][r['request_id']]['completed_at']-r['arrival_s'] for r in inputs[w]
                    if r['job_type']=='online' and result['requests'][r['request_id']]['completed_at'] is not None]
            cache[key]=dict(summary=summary,worst_offline_slack=slack,
               online_p95=quantile(online,.95),online_p99=quantile(online,.99),
               guard_ticks=sum(bool(e.get('guard',{}).get('risky_ids')) for e in result['events']),
               event_audit=verification['status'])
            count+=1
        return dict(model=model_name,configuration=spec['id'],window=w,**cache[key])
    selections={}
    for name in models:
        model_rows=[]
        for spec in specs:
            for w in inputs:
                model_rows.append(execute(name,spec,w))
        selections[name]=choose(model_rows,specs)
        records.extend(model_rows)
        print('S3 model',name,'selected',selections[name]['selected']['id'] if selections[name]['selected'] else None,
              'unique_rollouts',count,flush=True)
    # 2x2 on all six old development windows for the final model, shared H and
    # H+guard cells reuse search results; only two mechanism cells are new.
    final=gate['engineering_model']
    ablation=[]
    for scale,reservation in ((False,False),(True,False),(False,True),(True,True)):
        spec=dict(id=f'H_scale{int(scale)}_reserve{int(reservation)}',policy=H,scale=scale,reservation=reservation)
        ablation.extend(execute(final,spec,w) for w in inputs)
    picked=selections[gate['final_model']]['selected'] if gate['final_model'] else None
    policy_selected=None if picked is None else {k:picked[k] for k in ('id','policy','scale','reservation')}
    save(OUT/'s3/candidates.json',specs)
    save(OUT/'s3/search_results.json',records)
    save(OUT/'s3/rankings.json',selections)
    save(OUT/'s3/ablation.json',ablation)
    save(OUT/'s3/selection.json',dict(status='FROZEN_BEFORE_NEW_WINDOWS',model=gate['final_model'],
        engineering_model=final,P_selected=policy_selected,guard_safety=safety,
        fixed_guard_model='E',search_matrix_cells=len(records),unique_rollouts=count,
        reuse='identical window dynamics only; setup/cleanup proxies excluded from cache',
        elapsed_seconds=time.monotonic()-start,models_sha256=digest(OUT/'s2/models.json'),
        protected_files_verified=verify_protection()))
    print('S3 frozen',policy_selected,'unique rollouts',count)


if __name__=='__main__':
    run()
