"""Recompute deadline, resource and target metrics from reduced records (stdlib)."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def csv_rows(name):
    with gzip.open(ROOT/name,'rt',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)

def recompute():
    for name,digest in read('FILES_SHA256.json').items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    cases=read('cases.json')['cases'];expected=read('expected.json')
    assert len(cases)==12 and sum(len(c['physical_ids']) for c in cases)==36
    samples={s:c for c in cases for s in [c['prediction_id']]+c['physical_ids']}
    assert len(samples)==48
    requests={s:{} for s in samples};intervals={s:{0:[],1:[]} for s in samples};targets={s:{} for s in samples}
    for row in csv_rows('requests.csv.gz'):
        sample,rid=row['sample_id'],row['request_id'];assert rid not in requests[sample]
        end=samples[sample]['horizon'];finish=float(row['completed_at']) if row['completed_at'] else None
        arrival,deadline=float(row['arrival_s']),float(row['deadline_s'])
        assert 0<=arrival<end and arrival<=deadline
        assert row['job_type'] in ('online','offline')
        if finish is not None:
            assert row['status']=='completed' and finish>=float(row['dispatched_at'])>=arrival
        row['on_time']=finish is not None and finish<=min(end,deadline)
        requests[sample][rid]=row
    for row in csv_rows('occupation_intervals.csv.gz'):
        sample=row['sample_id'];start,stop=float(row['start_s']),float(row['end_s'])
        assert 0<=start<=stop<=samples[sample]['horizon']
        intervals[sample][int(row['device'])].append((start,stop))
    for row in csv_rows('targets.csv.gz'):
        sample=row['sample_id'];tick,target=int(row['tick']),int(row['target'])
        assert tick not in targets[sample] and target in (0,1,2)
        targets[sample][tick]=target
    metrics={}
    for sample,c in samples.items():
        assert sorted(targets[sample])==list(range(c['horizon']))
        occupied=0.
        for device,spans in intervals[sample].items():
            last=0.
            for a,b in sorted(spans):assert a>=last;occupied+=b-a;last=b
        metrics[sample]=dict(occupied_seconds=occupied,
            api_requests=sum(r['route']=='synthetic_api' for r in requests[sample].values()),
            on_time={k:sum(r['job_type']==k and r['on_time'] for r in requests[sample].values()) for k in ('online','offline')})
    pairs=[];case_results=[]
    for c in cases:
        pid=c['prediction_id'];pred=metrics[pid]
        for real in c['physical_ids']:
            assert requests[real].keys()==requests[pid].keys()
            for rid in requests[pid]:
                for k in ('job_type','arrival_s','deadline_s'):assert requests[pid][rid][k]==requests[real][rid][k]
            actual=metrics[real]
            pair=dict(case_id=c['case_id'],run_id=real,
                occupied_signed_relative_error=pred['occupied_seconds']/actual['occupied_seconds']-1,
                target_tick_agreement=sum(targets[pid][t]==targets[real][t] for t in targets[pid])/c['horizon'],
                ontime_label_disagreements=sum(requests[pid][r]['on_time']!=requests[real][r]['on_time'] for r in requests[pid]))
            ref=next(p for p in expected['pairs'] if p['run_id']==real)
            for k in ['occupied_seconds','api_requests']:
                close(pred[k],ref['predicted'][k]);close(actual[k],ref['physical'][k])
            assert pred['on_time']==ref['predicted']['on_time'] and actual['on_time']==ref['physical']['on_time']
            for k in ['occupied_signed_relative_error','target_tick_agreement','ontime_label_disagreements']:close(pair[k],ref[k])
            pairs.append(pair)
        group=pairs[-3:];ref=next(r for r in expected['cases'] if r['case_id']==c['case_id'])
        ape=statistics.median(abs(p['occupied_signed_relative_error']) for p in group)
        close(ape,ref['median_occupied_APE'])
        assert [min(p['target_tick_agreement'] for p in group),max(p['target_tick_agreement'] for p in group)]==ref['target_agreement_range']
        assert {k:[metrics[s]['on_time'][k] for s in c['physical_ids']] for k in ['online','offline']}==ref['physical_on_time_counts']
        case_results.append(dict(case_id=c['case_id'],median_resource_APE=ape))
    return dict(status='PASS',predictions=12,physical_runs=36,request_records=sum(map(len,requests.values())),
        control_ticks=sum(map(len,targets.values())),ontime_label_disagreements=sum(p['ontime_label_disagreements'] for p in pairs),
        verified='Deadline counts, API routing counts, occupation integrals and target agreement from reduced evidence',
        scope='Does not rerun Vidur, fit profiles, reproduce GPU inference or revalidate raw machine journals',cases=case_results)

if __name__=='__main__':print(json.dumps(recompute(),indent=2))
