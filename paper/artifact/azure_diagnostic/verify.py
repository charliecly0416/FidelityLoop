"""Recompute reduced Azure diagnostic metrics, not raw-event evidence."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from collections import defaultdict

BASE=Path(__file__).resolve().parent
MODELS=('O','C','C30','E','V')
POLICIES=('all1','all2','hysteresis_u8_d0_c60','hysteresis_u8_d0_c60__guard')
WINDOWS=('azure_conv_back_1500s','azure_conv_front_1500s')

def require(value,message):
    if not value: raise ValueError(message)

def rows(name): return list(csv.DictReader(io.StringIO((BASE/name).read_text())))
def boolean(value):
    require(value in ('True','False'),'invalid Boolean')
    return value=='True'

def outcome(row):
    keys=('online_total','online_timely','offline_total','offline_timely','completed','unfinished')
    counts={k:int(row[k]) for k in keys}
    require(all(v>=0 for v in counts.values()),'negative count')
    total=counts['online_total']+counts['offline_total']
    require(counts['online_total']>0 and counts['offline_total']>0,'empty population')
    require(counts['online_timely']<=counts['online_total'] and counts['offline_timely']<=counts['offline_total'],'timely denominator')
    require(counts['completed']+counts['unfinished']==total,'completion conservation')
    strict=counts['completed']==total and counts['online_timely']==counts['online_total'] and counts['offline_timely']==counts['offline_total']
    require(strict==boolean(row['p1_plus']),'strict P1+ differs from counts')
    historical=counts['online_total']-counts['online_timely']<=.01*counts['online_total'] and counts['offline_timely']>=.95*counts['offline_total']
    require(historical==boolean(row['contract_feasible']),'historical qualification differs')
    for k in ('occupied_s','cost'):require(math.isfinite(float(row[k])) and float(row[k])>=0,'invalid resource/cost')
    return strict

def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9)

def main():
    bindings=json.loads((BASE/'SOURCE_BINDINGS.json').read_text())
    params=json.loads((BASE/'PARAMETERS.json').read_text())
    require(params['prediction_provenance']=='retrospective' and bindings['provenance']=='retrospective','prediction provenance changed')
    for name,digest in bindings['reduced_files_sha256'].items():
        require(Path(name).name==name,'unexpected binding path')
        require(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,'hash mismatch: '+name)
    physical,predictions,pairs=rows('PHYSICAL_ROWS.csv'),rows('PREDICTION_ROWS.csv'),rows('PAIRED_ROWS.csv')
    require(len(physical)==24 and len(predictions)==40 and len(pairs)==120,'grid size')
    actual={r['run_id']:r for r in physical};pred={r['prediction_id']:r for r in predictions}
    require(len(actual)==24 and len(pred)==40,'duplicate identity')
    require({(r['policy'],r['window'],int(r['repeat'])) for r in physical}=={(p,w,i) for p in POLICIES for w in WINDOWS for i in (1,2,3)},'physical grid identity')
    require({(r['model'],r['policy'],r['window']) for r in predictions}=={(m,p,w) for m in MODELS for p in POLICIES for w in WINDOWS},'prediction grid identity')
    require({c['run_id'] for c in bindings['cells']}==set(actual),'raw bindings run identities')
    for r in physical+predictions:outcome(r)
    recomputed=[]
    for p in predictions:
        for a in physical:
            if (p['policy'],p['window'])!=(a['policy'],a['window']):continue
            pp,aa=outcome(p),outcome(a)
            recomputed.append(dict(model=p['model'],policy=p['policy'],window=p['window'],prediction_id=p['prediction_id'],run_id=a['run_id'],repeat=int(a['repeat']),predicted_p1_plus=pp,physical_p1_plus=aa,agreement=pp==aa,false_accept=pp and not aa,false_reject=not pp and aa,occupied_abs_error=abs(float(p['occupied_s'])-float(a['occupied_s'])),cost_abs_error=abs(float(p['cost'])-float(a['cost']))))
    keyed={(r['prediction_id'],r['run_id']):r for r in pairs}
    require(len(keyed)==120,'duplicate pair identity')
    require(set(keyed)=={(r['prediction_id'],r['run_id']) for r in recomputed},'pair set differs')
    for r in recomputed:
        saved=keyed[(r['prediction_id'],r['run_id'])]
        for k,v in r.items():
            if isinstance(v,bool):require(boolean(saved[k])==v,'pair Boolean differs: '+k)
            elif isinstance(v,(int,float)):require(close(saved[k],v),'pair numeric differs: '+k)
            else:require(saved[k]==v,'pair identity differs: '+k)
    summary=[]
    for model in MODELS:
        for scope in ('all_4_policies','dynamic_H_and_guard'):
            subset=[r for r in recomputed if r['model']==model and (scope=='all_4_policies' or r['policy'].startswith('hysteresis'))]
            summary.append(dict(model=model,scope=scope,physical_cells=len(subset),unique_predictions=len({r['prediction_id'] for r in subset}),physical_positive=sum(r['physical_p1_plus'] for r in subset),physical_negative=sum(not r['physical_p1_plus'] for r in subset),agreement=sum(r['agreement'] for r in subset),false_accept=sum(r['false_accept'] for r in subset),false_reject=sum(r['false_reject'] for r in subset),occupied_mae_s=sum(r['occupied_abs_error'] for r in subset)/len(subset),cost_mae=sum(r['cost_abs_error'] for r in subset)/len(subset)))
    saved_summary={(r['model'],r['scope']):r for r in rows('SUMMARY.csv')}
    require(len(saved_summary)==10,'summary grid')
    for r in summary:
        saved=saved_summary[(r['model'],r['scope'])]
        for k,v in r.items():
            if isinstance(v,(int,float)):require(close(saved[k],v),'summary numeric differs: '+k)
            else:require(saved[k]==v,'summary identity differs')
    mixed=[]
    for w in WINDOWS:
        rs=sorted([r for r in physical if r['policy']=='hysteresis_u8_d0_c60' and r['window']==w],key=lambda r:int(r['repeat']))
        values=[outcome(r) for r in rs]
        require(set(values)=={False,True},'mixed-repeat fact differs')
        mixed.append(dict(window=w,physical_p1_plus_in_repeat_order=values,minimum_errors_for_one_constant_boolean_prediction=min(sum(values),len(values)-sum(values))))
    table=''.join(f"{r['model']} & {r['agreement']}/24 & {r['false_accept']} & {r['false_reject']} & {r['occupied_mae_s']:.2f} & {r['cost_mae']:.4f} " + chr(92)*2 + chr(10) for r in summary if r['scope']=='all_4_policies')
    require((BASE/'table_rows.tex').read_text()==table,'table rows differ')
    print(json.dumps(dict(status='PASS',scope='reduced-count and metric verification; not raw-event or replay verification',prediction_provenance='retrospective',physical_runs=24,unique_predictions=40,pairs=120,summary=summary,mixed_H_repeats=mixed),indent=2))

if __name__=='__main__':main()
