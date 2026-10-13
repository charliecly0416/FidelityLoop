"""N5 post-acceptance CPU analysis: fixed behavior prices and paper source data."""
import argparse
from collections import Counter
import csv
import itertools
import math
from pathlib import Path
import statistics

from .calibrated import quantile
from .n5_acceptance import DEFAULT, BASE, close
from .reselect import ROOT, DOC, digest, read, lines, save

REGIMES=('steady','recovery','burst_offline')
DYNAMIC='hysteresis_u8_d0_c60'
POLICIES=('all1','all2',DYNAMIC)


def table(path,rows):
    if not rows:raise ValueError('empty source table')
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def reprice(row,gpu_multiplier,api_multiplier):
    if any(not math.isfinite(x) or x<=0 for x in (gpu_multiplier,api_multiplier)):
        raise ValueError('positive finite price multipliers required')
    transition=.002*row['startup_events_in_window']+.001*row['shutdown_events_in_window']
    operating=.001*row['gpu_occupied_seconds']*gpu_multiplier+transition+row['api_accept_cost']*api_multiplier
    return dict(operating=operating,total=operating+row['offline_deadline_miss_penalty'])


def break_even_api(dynamic,baseline,gpu_multiplier):
    """API-price multiplier where fixed observed behavior has equal total cost."""
    delta_api=dynamic['api_accept_cost']-baseline['api_accept_cost']
    fixed=(dynamic['gpu_occupied_seconds']-baseline['gpu_occupied_seconds'])*.001*gpu_multiplier
    fixed+=(dynamic['startup_events_in_window']-baseline['startup_events_in_window'])*.002
    fixed+=(dynamic['shutdown_events_in_window']-baseline['shutdown_events_in_window'])*.001
    fixed+=dynamic['offline_deadline_miss_penalty']-baseline['offline_deadline_miss_penalty']
    return -fixed/delta_api if delta_api else None


def reduce_group(rows,keys,metrics):
    result=[]
    groups={tuple(r[k] for k in keys) for r in rows}
    for key in sorted(groups):
        subset=[r for r in rows if tuple(r[k] for k in keys)==key]
        for metric in metrics:
            values=[r[metric] for r in subset if r[metric] is not None]
            result.append(dict(**dict(zip(keys,key)),metric=metric,n=len(values),
                               mean=statistics.mean(values) if values else None,
                               minimum=min(values) if values else None,maximum=max(values) if values else None))
    return result


def route(x):
    if isinstance(x,int) or x in ('gpu0','gpu1'):return 'local'
    return 'api' if x=='synthetic_api' else 'undispatched'


def compare_prediction(actual,prediction,pred_events):
    errors=[];confusion=Counter();actual_local=pred_local=0
    for rid,r in actual['requests'].items():
        p=prediction['requests'][rid];ar,pr=route(r['route']),route(p['route'])
        confusion[pr+'->'+ar]+=1
        actual_ok=ar=='local' and r['status']=='completed'
        pred_ok=pr=='local' and p['completed_at'] is not None
        actual_local+=actual_ok;pred_local+=pred_ok
        if actual_ok and pred_ok:
            errors.append(abs((r['completed_at']-r['dispatched_at'])-(p['completed_at']-p['dispatched_at'])))
    targets={int(k):v for k,v in actual['targets'].items()}
    predicted_targets={e['observation']['time']:e['target'] for e in pred_events if e['kind']=='decision'}
    if set(targets)!=set(predicted_targets):raise ValueError('policy tick identity mismatch')
    disagreements=sum(v!=predicted_targets[t] for t,v in targets.items())
    s=actual['summary'];n=len(actual['requests'])
    result=dict(shared_local_service_mae_seconds=statistics.mean(errors) if errors else None,
                shared_local_samples=len(errors),total_requests=n,shared_local_coverage_all_arrivals=len(errors)/n,
                actual_completed_local=actual_local,predicted_completed_local=pred_local,
                shared_local_coverage_actual_completed_local=len(errors)/actual_local if actual_local else None,
                local_api_route_disagreements=confusion['local->api']+confusion['api->local'],
                all_route_disagreements=sum(v for k,v in confusion.items() if len(set(k.split('->')))>1),
                actual_failed_or_censored=sum(r['status']!='completed' for r in actual['requests'].values()),
                predicted_censored=sum(p['completed_at'] is None for p in prediction['requests'].values()),
                target_disagreement_ticks=disagreements,target_disagreement_rate=disagreements/len(targets))
    for k in ('gpu_occupied_seconds','gpu_active_seconds'):
        result[k+'_actual']=s[k];result[k+'_predicted']=prediction['summary'][k]
        result[k+'_absolute_error']=abs(s[k]-prediction['summary'][k])
        result[k+'_relative_error']=result[k+'_absolute_error']/s[k] if s[k] else None
    return result,dict(confusion)


def analyze(acceptance,output,payload=DEFAULT/'payload'):
    acceptance,output,payload=map(Path,(acceptance,output,payload))
    output.mkdir(parents=True,exist_ok=False)
    accepted=read(acceptance/'acceptance.json')
    if accepted['status']!='PASS':raise ValueError('accepted N5 return required')
    raw=read(acceptance/'raw_recomputed.json');rows=[v['summary'] for v in raw.values()]
    if len(rows)!=27 or not all(r['feasible'] for r in rows):raise ValueError('unexpected accepted matrix')
    for r in rows:
        for key,expected in reprice(r,1,1).items():
            if not close(expected,r['window_'+('operating_cost' if key=='operating' else 'total_cost')]):raise ValueError('base-price identity')
    table(output/'run_metrics.csv',rows)
    request_rows=[dict(run_id=rid,request_id=req,**r) for rid,data in raw.items() for req,r in data['requests'].items()]
    table(output/'request_metrics.csv',request_rows)
    common=read(acceptance/'common_support_prediction_comparison.json')['rows'];table(output/'common_support_errors.csv',common)
    returned=read(payload/'n5_return_summary_20260920_r1/summary.json')
    registered={(r['run_id'],r['model']):r for r in returned['prediction_comparisons']}
    predroot=payload/'n5_prediction_lock_20260919_r1';prediction_rows=[];prediction_checks=0
    for rid,data in raw.items():
        r=data['summary']
        for model in ('original','calibrated'):
            folder=predroot/'predictions'/model/r['policy']/r['window']
            c,confusion=compare_prediction(data,read(folder/'result.json'),lines(folder/'events.jsonl'))
            published=registered[(rid,model)]
            for k,v in c.items():
                if k in published:
                    if not (close(v,published[k]) if isinstance(v,(int,float)) else v==published[k]):raise ValueError('returned comparison mismatch '+k)
                    prediction_checks+=1
            if published['route_confusion_predicted_to_actual']!=confusion:raise ValueError('route confusion differs')
            for metric in ('gpu_occupied_seconds','gpu_active_seconds'):
                for name in ('actual','predicted','absolute_error','relative_error'):
                    if not close(published[metric][name],c[metric+'_'+name]):raise ValueError('prediction resource error differs')
                    prediction_checks+=1
            prediction_rows.append(dict(run_id=rid,policy=r['policy'],window=r['window'],repeat=r['repeat'],model=model,**c))
    table(output/'prediction_errors.csv',prediction_rows)
    grouped=reduce_group(rows,('window','policy'),('window_operating_cost','window_total_cost','deployment_total_cost',
             'gpu_occupied_seconds','gpu_active_seconds','offline_on_time','censored','online_e2e_p95_seconds','online_e2e_p99_seconds'))
    table(output/'run_group_statistics.csv',grouped)
    table(output/'prediction_group_statistics.csv',reduce_group(prediction_rows,('window','policy','model'),
          ('shared_local_service_mae_seconds','shared_local_coverage_all_arrivals','gpu_occupied_seconds_absolute_error',
           'gpu_active_seconds_absolute_error','target_disagreement_rate','local_api_route_disagreements')))
    index={(r['window'],r['policy'],r['repeat']):r for r in rows}
    repriced=[];paired=[];boundaries=[]
    for r,g,a in itertools.product(rows,(.5,1.,2.),(.5,1.,2.)):
        v=reprice(r,g,a)
        repriced.append(dict(run_id=r['run_id'],window=r['window'],policy=r['policy'],repeat=r['repeat'],
                             gpu_multiplier=g,api_multiplier=a,operating_cost=v['operating'],total_cost=v['total'],feasible=r['feasible']))
    for window,repeat,baseline,g,a in itertools.product(sorted({r['window'] for r in rows}),(1,2,3),('all1','all2'),(.5,1.,2.),(.5,1.,2.)):
        d=index[(window,DYNAMIC,repeat)];b=index[(window,baseline,repeat)]
        dv,bv=reprice(d,g,a),reprice(b,g,a)
        paired.append(dict(window=window,repeat=repeat,baseline=baseline,gpu_multiplier=g,api_multiplier=a,
            dynamic_total=dv['total'],baseline_total=bv['total'],difference=dv['total']-bv['total'],
            savings_fraction=1-dv['total']/bv['total'],eligible=d['feasible'] and b['feasible']))
    for window,repeat,baseline,g in itertools.product(sorted({r['window'] for r in rows}),(1,2,3),('all1','all2'),(.5,1.,2.)):
        d=index[(window,DYNAMIC,repeat)];b=index[(window,baseline,repeat)]
        boundaries.append(dict(window=window,repeat=repeat,baseline=baseline,gpu_multiplier=g,
                               break_even_api_multiplier=break_even_api(d,b,g)))
    price_groups=[]
    for window,baseline,g,a in itertools.product(sorted({r['window'] for r in rows}),('all1','all2'),(.5,1.,2.),(.5,1.,2.)):
        points=[r for r in paired if (r['window'],r['baseline'],r['gpu_multiplier'],r['api_multiplier'])==(window,baseline,g,a)]
        avgd=statistics.mean(r['dynamic_total'] for r in points);avgb=statistics.mean(r['baseline_total'] for r in points)
        price_groups.append(dict(window=window,baseline=baseline,gpu_multiplier=g,api_multiplier=a,
            n=len(points),mean_difference=avgd-avgb,savings_ratio_of_means=1-avgd/avgb,
            min_difference=min(r['difference'] for r in points),max_difference=max(r['difference'] for r in points),
            eligible_repeats=sum(r['eligible'] for r in points),saving_repeats=sum(r['eligible'] and r['difference']<0 for r in points)))
    for name,data in (('repriced_runs',repriced),('price_pairs',paired),('price_group_statistics',price_groups),('price_break_even',boundaries)):
        table(output/(name+'.csv'),data)
    lifecycle=[]
    for rid,data in raw.items():
        s=data['summary']
        for g in data['generations']:
            lifecycle.append(dict(run_id=rid,window=s['window'],policy=s['policy'],repeat=s['repeat'],**g))
    table(output/'lifecycle_samples.csv',lifecycle)
    life_summary=[]
    for phase in ('setup','window'):
        subset=[g for g in lifecycle if g['launch_phase']==phase]
        x=[g['startup_seconds'] for g in subset]
        life_summary.append(dict(phase=phase,n=len(x),minimum=min(x),median=statistics.median(x),p90=quantile(x,.9),maximum=max(x),
                                 unit='seconds',scope='generation-weighted descriptive; correlated within runs'))
    table(output/'lifecycle_startup_summary.csv',life_summary)
    campaign=payload/'n5-frozen-heldout-campaign-20260919-r1';timeline=[];state_rows=[];miss_rows=[]
    for regime in REGIMES:
        window='locked_test_'+regime+'_v2';rid=DYNAMIC+'__'+window+'__r1';data=raw[rid]
        c=lines(campaign/(rid+'__attempt01')/'controller_events.jsonl');origin=data['origin_ns']
        arrivals=Counter(r['arrival_s'] for r in data['requests'].values())
        predicted=lines(predroot/'predictions/calibrated'/DYNAMIC/window/'events.jsonl')
        predicted_targets={e['observation']['time']:e['target'] for e in predicted if e['kind']=='decision'}
        api_counts=Counter(int(r['dispatched_at']) for r in data['requests'].values() if r['route']=='synthetic_api')
        for e in c:
            if e['kind']=='policy_tick':
                o=e['observation'];tick=e['tick']
                timeline.append(dict(window=window,tick=tick,arrivals=arrivals[tick],queue_online=o['queue_online'],queue_offline=o['queue_offline'],
                    target=e['target'],predicted_target=predicted_targets[tick],active=o['active'],starting=o['starting'],draining=o['draining'],off=o['off'],api_accepts=api_counts[tick]))
            if e['kind']=='lifecycle_state':
                state_rows.append(dict(window=window,gpu=e['gpu'],generation=e['generation'],time_s=(e['controller_monotonic_ns']-origin)/1e9,state=e['state']))
        for req,r in data['requests'].items():
            if r['job_type']=='offline' and not r['timely']:
                miss_rows.append(dict(window=window,request_id=req,arrival_s=r['arrival_s'],deadline_s=r['deadline_s'],status=r['status']))
    table(output/'timeline_repeat1.csv',timeline);table(output/'state_transitions_repeat1.csv',state_rows)
    table(output/'offline_misses_repeat1.csv',miss_rows)
    # Summary is deliberately per window, with both operating and penalty-inclusive costs.
    summary=[]
    for regime in REGIMES:
        window='locked_test_'+regime+'_v2'
        d=[r for r in rows if r['window']==window and r['policy']==DYNAMIC]
        b=[r for r in rows if r['window']==window and r['policy']=='all1']
        values={k:1-statistics.mean(r[k] for r in d)/statistics.mean(r[k] for r in b)
                for k in ('window_operating_cost','window_total_cost','deployment_total_cost','gpu_occupied_seconds')}
        summary.append(dict(window=window,online_on_time=sum(r['online_on_time'] for r in d),online_arrivals=sum(r['online_arrivals'] for r in d),
                            offline_on_time=[r['offline_on_time'] for r in d],savings_against_all1=values))
    result=dict(status='CPU_ANALYSIS_COMPLETE',summary=summary,registered_prediction_field_checks=prediction_checks,
                repriced_cells=len(repriced),price_pairs=len(paired),price_groups=price_groups,
                break_even_rows=boundaries,lifecycle_startup=life_summary,
                no_new_GPU_runs=True,no_policy_reselection=True,no_predictor_refitting=True,
                common_support_scope='post-hoc diagnostic on actual+original+calibrated completed-local intersection',
                additional_GPU_work_required=False)
    save(output/'analysis.json',result)
    bindings={str(p.relative_to(output)):digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    save(output/'analysis_manifest.json',dict(files=bindings,acceptance_sha256=digest(acceptance/'acceptance.json'),
         raw_recomputed_sha256=digest(acceptance/'raw_recomputed.json'),script_sha256=digest(Path(__file__)),
         registered_design_sha256=digest(ROOT/DOC/'N5_PAPER_EXPERIMENT_DESIGN_20260919.md'),
         figure_contract_sha256=digest(ROOT/DOC/'N5_CPU_FIGURE_CONTRACT_20260920.md')))
    print(__import__('json').dumps({k:v for k,v in result.items() if k not in ('price_groups','break_even_rows')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acceptance',type=Path,default=BASE/'n5_cpu_acceptance_20260920_r1')
    p.add_argument('--payload',type=Path,default=DEFAULT/'payload')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.acceptance,a.output,a.payload)
