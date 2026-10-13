"""Registered development support, fixed-policy controls and startup stress checks."""
import statistics
from fidelityloop.legacy.maxopt_v2.calibrated import CalibratedSimulator
from .common import OUT, DEV, PRED, RAW, H, H_ID, read, lines, save
from .simulator import Simulator
from .audit import audit


def run():
    models=read(OUT/'s2/models.json');safety=read(OUT/'s1/startup_model_v3.json')
    contract=read(DEV/'inputs/contract.json');truth=read(RAW)
    windows=sorted({v['summary']['window'] for v in truth.values()})
    support=[];rollouts=0
    for w in windows:
        rows=lines(PRED/(w+'.jsonl'))
        for policy_id,policy in [('all1',{'name':'all1'}),('all2',{'name':'all2'}),(H_ID,H)]:
            predictions={}
            for name,model in models.items():
                result=CalibratedSimulator(rows,contract,model,policy,allow_locked_test=True).run();rollouts+=1
                audit(rows,result['events'],result['summary'],contract,model,policy,result['requests'])
                predictions[name]=result['requests']
            for repeat in (1,2,3):
                actual=truth[f'{policy_id}__{w}__r{repeat}']['requests']
                common=[rid for rid,r in actual.items() if r['status']=='completed' and str(r['route']).startswith('gpu')
                        and all(type(p[rid]['route']) is int and p[rid]['completed_at'] is not None for p in predictions.values())]
                for name,pred in predictions.items():
                    errors=[abs((pred[r]['completed_at']-pred[r]['dispatched_at'])-
                                (actual[r]['completed_at']-actual[r]['dispatched_at'])) for r in common]
                    support.append(dict(window=w,policy=policy_id,repeat=repeat,model=name,count=len(common),
                         shared_all_five_mae=statistics.mean(errors) if errors else None,
                         E_local_constraint_pass=len(common)>=30 and bool(errors) and statistics.mean(errors)<=.5))
    stress=[]
    for startup in (safety['startup']['generation']['p90'],safety['startup_max_seconds'],114.,285.):
        for path in sorted((DEV/'inputs/C2').glob('*.jsonl')):
            rows=lines(path);model=dict(models['E'],startup_seconds=startup)
            result=Simulator(rows,contract,model,H,safety=safety,guard_model=models['E'],scale=True,reservation=True).run()
            audit(rows,result['events'],result['summary'],contract,model,H,result['requests'],
                  safety=safety,guard_model=models['E'],scale=True,reservation=True)
            rollouts+=1
            stress.append(dict(window=path.stem,startup_seconds=startup,summary=result['summary']))
    save(OUT/'s2/development_support_and_stress.json',dict(common_support=support,startup_stress=stress,
         extra_rollouts=rollouts,total_s2_rollouts=read(OUT/'s2/gate.json')['rollouts']+rollouts,
         scope='old development only; fixed selection unchanged; no new heldout consumed'))
    print('Development support/stress complete:',rollouts,'additional CPU rollouts')


if __name__=='__main__':run()
