"""S2 bounded two-parameter calibration; fold lifecycle estimation has no heldout access."""
import copy
import itertools
import statistics
import time

from fidelityloop.legacy.maxopt_v2.calibrated import CalibratedSimulator
from fidelityloop.legacy.maxopt_v2.n5_analysis import compare_prediction
from .lifecycle_extract import parameters
from .common import OUT, DEV, PRED, H, H_ID, read, lines, save, verify_protection


class Replay(CalibratedSimulator):
    """Actual-target injection for development diagnostics only."""
    def __init__(self, *args, targets=None, **kwargs):
        super().__init__(*args, **kwargs)
        if targets is not None:
            if set(targets) != set(map(str, range(self.end))):
                raise ValueError('incomplete diagnostic target timeline')
            self.policy.decide = lambda observation: targets[str(observation['time'])]


def model_set(p):
    old = read(DEV / 'models.json')
    models = dict(O=old['original'], C=old['calibrated'])
    models['C30'] = dict(models['C'], startup_seconds=30.)
    models['E'] = dict(models['C'], startup_seconds=p['startup_seconds'], shutdown_seconds=p['shutdown_seconds'])
    # Window starts dual-ready: setup is disclosure only and not a fitted parameter.
    return copy.deepcopy(models)


def grid(p):
    return sorted(set((max(p['startup_min_seconds'],p['startup_seconds']*s),
                       max(p['shutdown_min_seconds'],p['shutdown_seconds']*d))
                      for s,d in itertools.product((.9,1.,1.1),(.8,1.,1.2))))


def loss(prediction, truth):
    p, t = prediction['summary'], truth['summary']
    n = t['online_arrivals']+t['offline_arrivals']
    occ = abs(p['gpu_occupied_seconds']-t['gpu_occupied_seconds'])
    active = abs(p['gpu_active_seconds']-t['gpu_active_seconds'])
    queue = abs(p['queue_seconds']-truth['queue_seconds'])
    return dict(L=.5*occ/4200+.25*active/4200+.25*queue/(2100*n),
                occupied_mae=occ, active_mae=active, queue_mae=queue)


def grouped_mean(records, key):
    windows = sorted({r['window'] for r in records})
    return statistics.mean(statistics.mean(r[key] for r in records if r['window']==w) for w in windows)


def evaluate(predictions, truth, run_ids):
    return [dict(run_id=rid,window=truth[rid]['summary']['window'],
                 **loss(predictions[truth[rid]['summary']['window']],truth[rid])) for rid in sorted(run_ids)]


def choose_grid(candidates, empirical):
    return min(candidates,key=lambda c:(round(c['L'],9),
        abs(c['startup_seconds']/empirical['startup_seconds']-1)+abs(c['shutdown_seconds']/empirical['shutdown_seconds']-1),
        c['startup_seconds'],c['shutdown_seconds']))


def gate(records):
    totals={m:grouped_mean([r for r in records if r['model']==m],'L') for m in ('O','C','C30','E','V')}
    windows=sorted({r['window'] for r in records})
    def safe(a,b):
        return all(statistics.mean(r['occupied_mae'] for r in records if r['model']==a and r['window']==w)
                   <=statistics.mean(r['occupied_mae'] for r in records if r['model']==b and r['window']==w)+21 for w in windows)
    incremental=totals['V']<=.9*totals['E'] and safe('V','E')
    selected='V' if incremental else 'E'
    passed=totals[selected]<=.8*totals['O'] and safe(selected,'O')
    return dict(status='PASS' if passed else 'FAIL', final_model=selected if passed else None,
                engineering_model=selected if passed else 'E', V_incremental_pass=incremental,
                mean_L=totals, policy_only=not passed)


def run():
    verify_protection()
    begun=time.monotonic()
    samples, truth = read(OUT/'s1/lifecycle_samples.json'), read(OUT/'s1/truth.json')
    contract=read(DEV/'inputs/contract.json')
    windows=sorted({v['summary']['window'] for v in truth.values()})
    inputs={w:lines(PRED/(w+'.jsonl')) for w in windows}
    dynamic={rid for rid in truth if rid.startswith(H_ID+'__')}
    count=0
    def predict(model, names, targets=None):
        nonlocal count
        result={}
        for w in names:
            if count>=500 or time.monotonic()-begun>8*3600:
                raise RuntimeError('S2 budget exhausted')
            sim=Replay(inputs[w],contract,model,H,allow_locked_test=True,targets=targets)
            result[w]=sim.run()
            count+=1
        return result
    def fit(train):
        support={r for r in truth if (truth[r]['summary']['window'],truth[r]['summary']['repeat'])
                 in {(truth[t]['summary']['window'],truth[t]['summary']['repeat']) for t in train}}
        p=parameters(samples,support)
        models=model_set(p)
        candidates=[]
        train_windows=sorted({truth[r]['summary']['window'] for r in train})
        for startup,shutdown in grid(p):
            model=dict(models['E'],startup_seconds=startup,shutdown_seconds=shutdown)
            pred=predict(model,train_windows)
            rows=evaluate(pred,truth,train)
            candidates.append(dict(startup_seconds=startup,shutdown_seconds=shutdown,L=grouped_mean(rows,'L')))
        selected=choose_grid(candidates,p)
        models['V']=dict(models['E'],**{k:selected[k] for k in ('startup_seconds','shutdown_seconds')})
        return models,p,candidates
    folds, records, repeats=[],[],[]
    for w in windows:
        test={r for r in dynamic if truth[r]['summary']['window']==w}
        models,p,candidates=fit(dynamic-test)
        fold=dict(heldout_window=w,train_run_ids=sorted(dynamic-test),test_run_ids=sorted(test),
                  parameters=p,models=models,grid=candidates)
        folds.append(fold)
        for m,model in models.items():
            pred=predict(model,[w])
            records.extend(dict(model=m,**r) for r in evaluate(pred,truth,test))
    for repeat in (1,2,3):
        test={r for r in dynamic if truth[r]['summary']['repeat']==repeat}
        models,p,candidates=fit(dynamic-test)
        for m,model in models.items():
            repeats.extend(dict(heldout_repeat=repeat,model=m,**r) for r in evaluate(predict(model,windows),truth,test))
    models,p,candidates=fit(dynamic)
    diagnostic=[]
    predictions={m:predict(model,windows) for m,model in models.items()}
    for rid in sorted(dynamic):
        w=truth[rid]['summary']['window']
        pred=predict(models['E'],[w],targets=truth[rid]['targets'])[w]
        diagnostic.append(dict(run_id=rid,mode='conditional_actual_target_development_only',**loss(pred,truth[rid])))
    comparisons=[]
    for rid in sorted(dynamic):
        w=truth[rid]['summary']['window']
        for m in models:
            prediction=predictions[m][w]
            actual=dict(truth[rid])
            metrics,_=compare_prediction(actual,prediction,prediction['events'])
            comparisons.append(dict(run_id=rid,model=m,**metrics))
    result=gate(records)
    save(OUT/'s2/models.json',models)
    save(OUT/'s2/folds.json',folds)
    save(OUT/'s2/lowo.json',records)
    save(OUT/'s2/repeat_sensitivity.json',repeats)
    save(OUT/'s2/refit_grid.json',candidates)
    save(OUT/'s2/conditional_diagnostic.json',diagnostic)
    save(OUT/'s2/refit_comparisons.json',comparisons)
    save(OUT/'s2/gate.json',dict(**result,rollouts=count,elapsed_seconds=time.monotonic()-begun,
         development_only=True, old_v2_unchanged=verify_protection()))
    print('S2',result,'rollouts',count,flush=True)


if __name__=='__main__':
    run()
