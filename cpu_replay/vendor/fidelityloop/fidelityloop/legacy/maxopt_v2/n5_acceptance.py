"""CPU acceptance of the actual N5 return, followed by independent RAW tables.

Never starts a GPU, changes evidence, reselects a policy, or fits test data.
Returned portable validators are one check; request/occupation/price tables
below are recomputed separately from controller events and frozen inputs.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

from .calibrated import CalibratedSimulator, quantile
from .calibrated_audit import audit
from .prepare_n5 import derive, canonical
from .reselect import ROOT, DOC, digest, read, lines, save

BASE=ROOT/'artifacts/max_optimization_v2_20260916'
DEFAULT=BASE/'n5_gpu_return_20260920_r1'
ARCHIVE_SHA='20c6be6c66d17a6fbc10202957f8ebbcc227a0be89024c5c8f11dfc4bd01610f'
FREL=Path('artifacts/max_optimization_v2_20260916/n4b_cpu_reselection_20260919_r2')


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)


def integrate(intervals,left,right):
    return sum(max(0,min(stop,right)-max(start,left))/1e9 for start,stop in intervals)


def raw_metrics(expected,controller,projected,record):
    """Separate request and resource primitive computation, no GPU ledger import."""
    start=next(e for e in controller if e['kind']=='observation_start')
    origin=start['origin_monotonic_ns'];cutoff=origin+2100*10**9
    by_id={r['request_id']:r for r in expected}
    dispatch,terminal={},{}
    generations={}
    active_intervals=[];active_since={}
    targets={}
    for seq,e in enumerate(controller):
        assert e['seq']==seq
        t=e['controller_monotonic_ns'];k=e['kind']
        key=(e.get('gpu'),e.get('generation'))
        if k=='launch_issued':generations[key]=dict(launch=t,gpu=key[0],generation=key[1])
        elif k=='ready':generations[key]['ready']=t
        elif k=='shutdown_issued':generations[key]['shutdown']=t
        elif k=='verified_release':generations[key]['release']=t
        elif k=='lifecycle_state':
            if e['previous']=='active':active_intervals.append((active_since.pop(e['gpu']),t))
            if e['state']=='active':active_since[e['gpu']]=t
        elif k in ('local_dispatch','api_accept'):
            rid=e['request_id'];assert rid in by_id and rid not in dispatch
            dispatch[rid]=dict(at_s=(t-origin)/1e9,route='gpu'+str(e['gpu']) if k=='local_dispatch' else 'synthetic_api',raw_seq=seq)
        elif k=='worker_receipt':
            w=e['worker_event'];rid=w.get('request_id')
            if rid in by_id and w['kind'] in ('finished','failed','cancelled') and t<=cutoff:
                assert rid not in terminal
                terminal[rid]=dict(at_s=(t-origin)/1e9,status='completed' if w['kind']=='finished' else 'failed',raw_seq=seq)
        elif k=='api_completed' and t<=cutoff:
            rid=e['request_id'];assert rid not in terminal and rid in dispatch
            assert (t-origin)/1e9-dispatch[rid]['at_s']>=5
            terminal[rid]=dict(at_s=(t-origin)/1e9,status='completed',raw_seq=seq)
        elif k=='policy_tick':
            assert e['tick'] not in targets
            targets[e['tick']]=e['target']
    assert len(targets)==2100 and not active_since
    projected_term={e['request_id']:e for e in projected if e['kind']=='terminal'}
    assert len(projected_term)==len(expected)
    requests={}
    for rid,r in by_id.items():
        actual=terminal.get(rid);projection=projected_term[rid]
        status=actual['status'] if actual else 'censored'
        assert projection['status']==status
        if actual:assert close(projection['at_s'],actual['at_s']) and projection['raw_controller_seq']==actual['raw_seq']
        complete=actual['at_s'] if actual and status=='completed' else None
        d=dispatch.get(rid,{})
        timely=complete is not None and complete <= (r['arrival_s']+60 if r['job_type']=='online' else r['deadline_s'])
        requests[rid]=dict(job_type=r['job_type'],arrival_s=r['arrival_s'],deadline_s=r['deadline_s'],
                           route=d.get('route'),dispatched_at=d.get('at_s'),completed_at=complete,status=status,timely=timely,
                           input_tokens=r['input_tokens'],output_tokens=r['max_output_tokens'])
    phases={}
    boundaries=dict(setup=(controller[0]['controller_monotonic_ns'],origin),window=(origin,cutoff),
                    cleanup=(cutoff,controller[-1]['controller_monotonic_ns']))
    generation_rows=[]
    for g in generations.values():
        assert g['launch']<=g['ready']<=g['shutdown']<=g['release']
        generation_rows.append(dict(**g,startup_seconds=(g['ready']-g['launch'])/1e9,
                                    shutdown_seconds=(g['release']-g['shutdown'])/1e9,
                                    launch_phase='setup' if g['launch']<origin else 'window'))
    for phase,(left,right) in boundaries.items():
        occupied=integrate([(g['launch'],g['release']) for g in generations.values()],left,right)
        active=integrate(active_intervals,left,right)
        starts=sum(left<=g['launch']<right for g in generations.values())
        stops=sum(left<=g['shutdown']<right for g in generations.values())
        phases[phase]=dict(occupied_seconds=occupied,active_seconds=active,startups=starts,shutdowns=stops,
                           holding_transition_cost=.001*occupied+.002*starts+.001*stops)
    api=[r for r in requests.values() if r['route']=='synthetic_api']
    api_cost=sum(r['input_tokens']*1e-6+r['output_tokens']*3e-6 for r in api)
    group={k:[r for r in requests.values() if r['job_type']==k] for k in ('online','offline')}
    counts={k:dict(arrivals=len(g),on_time=sum(r['timely'] for r in g)) for k,g in group.items()}
    misses=counts['offline']['arrivals']-counts['offline']['on_time']
    online=[r['completed_at']-r['arrival_s'] for r in group['online'] if r['completed_at'] is not None]
    s=dict(run_id=record['run_id'],policy=record['policy'],window=record['window'],repeat=record['repeat'],attempt=record['attempt'],
           online_arrivals=counts['online']['arrivals'],online_on_time=counts['online']['on_time'],
           offline_arrivals=counts['offline']['arrivals'],offline_on_time=counts['offline']['on_time'],
           censored=sum(r['status']=='censored' for r in requests.values()),api_accepted=len(api),
           gpu_occupied_seconds=phases['window']['occupied_seconds'],gpu_active_seconds=phases['window']['active_seconds'],
           window_holding_cost=.001*phases['window']['occupied_seconds'],
           startup_events_in_window=phases['window']['startups'],shutdown_events_in_window=phases['window']['shutdowns'],
           api_accept_cost=api_cost,offline_deadline_miss_penalty=.01*misses,
           window_operating_cost=phases['window']['holding_transition_cost']+api_cost,
           setup_total_cost=phases['setup']['holding_transition_cost'],cleanup_total_cost=phases['cleanup']['holding_transition_cost'],
           online_e2e_p95_seconds=quantile(online,.95),online_e2e_p99_seconds=quantile(online,.99))
    s['window_total_cost']=s['window_operating_cost']+s['offline_deadline_miss_penalty']
    s['deployment_total_cost']=s['window_total_cost']+s['setup_total_cost']+s['cleanup_total_cost']
    s['feasible']=(counts['online']['on_time']>=.99*counts['online']['arrivals'] and counts['offline']['on_time']>=.95*counts['offline']['arrivals'])
    return dict(summary=s,requests=requests,phases=phases,generations=generation_rows,targets=targets,origin_ns=origin)


def review(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    payload=root/'payload';docs=payload/'docs/maxopt_n5_execution_20260919'
    output.mkdir(parents=True,exist_ok=False)
    checks=[]
    def ck(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise ValueError(name)
    archive=ROOT/'n5_gpu_return_20260920_r1.zip'
    ck('outer_SHA',digest(archive)==ARCHIVE_SHA)
    manifest=read(root/'MANIFEST.json')
    for n,m in manifest['files'].items():
        p=payload/n
        ck('payload:'+n,p.stat().st_size==m['bytes'] and digest(p)==m['sha256'])
    # The original locally accepted package, not the return's account of it.
    with zipfile.ZipFile(ROOT/'n5_preparation_outbound_20260919_r1.zip') as z:
        original=json.loads(z.read('N5_PACKAGE_MANIFEST.json'))
        for name,m in original['files'].items():
            ck('outbound_unchanged:'+name,digest(payload/'n5-preparation-20260919-r1'/name)==m['sha256'])
    frozen=payload/'n5-preparation-20260919-r1'/FREL
    local=ROOT/FREL
    for name in ('registration.json','models.json','n5_plan.json','inputs/contract.json'):
        ck('local_authority:'+name,digest(frozen/name)==digest(local/name))
    plan,models,contract=read(frozen/'n5_plan.json'),read(frozen/'models.json'),read(frozen/'inputs/contract.json')
    prediction=payload/'n5_prediction_lock_20260919_r1';lock=read(prediction/'prediction_lock.json')
    ck('prediction_lock',digest(prediction/'prediction_lock.json')=='79557ebbeda32ff04cbca4baeb95590feb1684a3118d897d9ea6e38759dcf942')
    for n,h in lock['files'].items():ck('prediction:'+n,digest(prediction/n)==h)
    from tokenizers import Tokenizer
    sources=BASE/'sources';parent=BASE/'n1_public_v4';pm=read(parent/'manifest.json')
    ck('parent_manifest',digest(parent/'manifest.json')==lock['parent_manifest_sha256'])
    for n in ('tokenizer.json','prompt_locked_test.rst'):
        ck('parent_source:'+n,digest(sources/n)==pm['sources']['records'][n]['sha256'])
    tokens=Tokenizer.from_file(str(sources/'tokenizer.json')).encode((sources/'prompt_locked_test.rst').read_text()).ids
    corpus=dict(split='locked_test',tokens=tokens,prompt_source_sha256=digest(sources/'prompt_locked_test.rst'),
                tokenizer_sha256=digest(sources/'tokenizer.json'),arrival_source_sha256=pm['sources']['records']['BurstGPT_without_fails_1.csv']['sha256'])
    recipe=read(frozen/'inputs/revision_registration.json');data={}
    for regime in ('steady','recovery','burst_offline'):
        window='locked_test_'+regime+'_v2';name=window+'.jsonl'
        ck('parent_input:'+window,digest(parent/name)==pm['files'][name]['sha256'])
        rows,mapping=derive(lines(parent/name),window,recipe,corpus,digest(parent/name))
        ck('C2_recipe_byte_reproduction:'+window,hashlib.sha256(b''.join(canonical(r)+b'\n' for r in rows)).hexdigest()==digest(prediction/name))
        ck('C2_full_mapping_and_statistics:'+window,mapping==read(prediction/(window+'_mapping_statistics.json')))
        data[window]=rows
    prediction_checks=0
    for model,policy,window in itertools.product(models,plan['policies'],data):
        p=prediction/'predictions'/model/policy['id']/window
        expected=read(p/'result.json');events=lines(p/'events.jsonl')
        replay=CalibratedSimulator(data[window],contract,models[model],policy['policy'],allow_locked_test=True).run()
        ck('prediction_exact_replay:'+str(p.relative_to(prediction)),replay.pop('events')==events and replay==expected)
        v=audit(data[window],events,expected['summary'],contract,models[model],policy['policy'],expected['requests'])
        prediction_checks+=v['checks']
    # Load only the returned CPU verifiers; never call GPU runner/backend.
    import scripts
    scripts.__path__.append(str(payload/'scripts'))
    from scripts.maxopt_n5.verify import verify
    from scripts.maxopt_n5.verify_canary import verify_canary
    cv=verify_canary(payload/'n5-lifecycle-actuator-canary-20260919-r1')
    save(output/'canary_portable_recheck.json',cv)
    ck('canary_portable_recheck',cv['status']=='PASS')
    campaign=payload/'n5-frozen-heldout-campaign-20260919-r1';campaign_result=read(campaign/'campaign_result.json')
    accepted=[a for a in campaign_result['attempts'] if a['status']!='TECHNICAL_INVALID']
    invalid=[a for a in campaign_result['attempts'] if a['status']=='TECHNICAL_INVALID']
    ck('matrix_attempts',len(accepted)==27 and len(invalid)==1 and campaign_result['technical_retries']==1)
    ck('frozen_order',[(a['run_id'],a['policy'],a['window'],a['repeat']) for a in accepted]==[(a['run_id'],a['policy'],a['window'],a['repeat']) for a in plan['matrix']])
    ck('only_failed_cell_retried',all(a['attempt']==(2 if a['run_id']==invalid[0]['run_id'] else 1) for a in accepted))
    receipt=read(docs/'prediction_freeze_receipt.json')
    summary=read(payload/'n5_return_summary_20260920_r1/summary.json')
    reported={r['run_id']:r for r in summary['runs']};actual={};verifiers=[]
    for a in accepted:
        folder=campaign/(a['run_id']+f'__attempt{a["attempt"]:02}')
        v=verify(folder)
        ck('portable_RAW_verifier:'+a['run_id'],v['status']=='PASS')
        ck('stored_verifier_binding:'+a['run_id'],digest(folder/'verification.json')==a['verification_sha256'])
        stored=read(folder/'verification.json')
        for key in ('status','request_summary','lifecycle','accounting','raw_sha256'):
            ck('verifier_recomputed:'+a['run_id']+':'+key,v[key]==stored[key])
        verifiers.append(dict(run_id=a['run_id'],status=v['status'],checks=len(v['checks']),raw_sha256=v['raw_sha256']))
        controller=lines(folder/'controller_events.jsonl')
        ck('preobservation_freeze:'+a['run_id'],controller[0]['controller_monotonic_ns']>receipt['frozen_monotonic_ns'])
        raw=raw_metrics(data[a['window']],controller,lines(folder/'request_events.jsonl'),a)
        for key,value in raw['summary'].items():
            if key in reported[a['run_id']]:
                published=reported[a['run_id']][key]
                ck('independent_metric:'+a['run_id']+':'+key,close(value,published) if isinstance(value,(int,float)) else value==published)
        actual[a['run_id']]=raw
    save(output/'portable_rechecks.json',dict(runs=verifiers,total_checks=sum(v['checks'] for v in verifiers)))
    save(output/'raw_recomputed.json',actual)
    # Evaluate new and original models on the SAME intersection as an additional
    # post-hoc diagnostic; never replace the registered shared-local metrics.
    comparison=[]
    for a in accepted:
        raw=actual[a['run_id']]
        preds={m:read(prediction/'predictions'/m/a['policy']/a['window']/'result.json') for m in models}
        common=[rid for rid,r in raw['requests'].items() if r['status']=='completed' and r['route'] in ('gpu0','gpu1')
                and all(isinstance(p['requests'][rid]['route'],int) and p['requests'][rid]['completed_at'] is not None for p in preds.values())]
        for m,p in preds.items():
            errors=[]
            for rid in common:
                r,q=raw['requests'][rid],p['requests'][rid]
                errors.append(abs(r['completed_at']-r['dispatched_at']-(q['completed_at']-q['dispatched_at'])))
            comparison.append(dict(run_id=a['run_id'],policy=a['policy'],window=a['window'],repeat=a['repeat'],model=m,
                common_samples=len(common),arrivals=len(raw['requests']),common_coverage=len(common)/len(raw['requests']),
                common_local_mae_seconds=statistics.mean(errors) if errors else None,
                occupied_prediction=p['summary']['gpu_occupied_seconds'],occupied_actual=raw['summary']['gpu_occupied_seconds'],
                occupied_absolute_error=abs(p['summary']['gpu_occupied_seconds']-raw['summary']['gpu_occupied_seconds'])))
    save(output/'common_support_prediction_comparison.json',dict(scope='post-hoc same-request-support diagnostic, no fitting',rows=comparison))
    release=read(docs/'final_release_snapshot.json')
    ck('final_release_identity',release['status']=='PASS' and not release['matching_live_processes'] and len(release['owned_generations'])==148
       and all(not g['original_identity_present'] for g in release['owned_generations']))
    ck('final_GPU_queries',all(q['returncode']==0 for q in release['gpu_queries']) and not release['gpu_queries'][1]['stdout'].strip())
    # All declared evidence remains byte-identical after CPU-only verification.
    ck('evidence_unchanged_after_review',all(digest(payload/n)==m['sha256'] for n,m in manifest['files'].items()))
    result=dict(status='PASS',scope='N5 acceptance; scientific limitations retained',archive_sha256=ARCHIVE_SHA,
        payload_files=len(manifest['files']),checks=checks,check_count=len(checks),canary_checks=len(cv['checks']),
        prediction_runs=18,prediction_ledger_checks=prediction_checks,accepted_real_runs=27,technical_invalid_attempts=1,
        real_RAW_verifier_checks=sum(v['checks'] for v in verifiers),independent_raw_requests=sum(len(r['requests']) for r in actual.values()),
        online_misses=sum(r['summary']['online_arrivals']-r['summary']['online_on_time'] for r in actual.values()),
        offline_misses=sum(r['summary']['offline_arrivals']-r['summary']['offline_on_time'] for r in actual.values()),
        all_runs_feasible=all(r['summary']['feasible'] for r in actual.values()),GPU_additional_work_required=False,
        limits=['synthetic API, scenario prices; no actual energy/bill claim','3 fixed held-out windows, 3 repeats; no reliability guarantee',
                '12 offline requests censored; SLO pass is not all tasks complete','local accuracy improvement does not imply closed-loop resource accuracy improvement'])
    save(output/'acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','limits')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--return-root',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();review(a.return_root,a.output)
