"""S4 preparation and final G-L freeze. No GPU access or performance selection."""
import argparse
from pathlib import Path

from .common import ROOT, OUT, DEV, DOC, H, H_ID, CAMPAIGN, read, lines, save, save_lines, digest, verify_protection
from .workload import build
from .simulator import Simulator
from .audit import audit


def arms_for(selection):
    arms=[dict(id=name,policy=dict(name=name),scale=False,reservation=False,guard_model='E') for name in ('all1','all2')]
    arms.extend([dict(id=H_ID,policy=H,scale=False,reservation=False,guard_model='E'),
                 dict(id=H_ID+'__guard',policy=H,scale=True,reservation=True,guard_model='E')])
    selected=selection['P_selected']
    if selected:
        spec=dict(selected,guard_model=selection['model'])
        if not any(all(a[k]==spec[k] for k in ('policy','scale','reservation','guard_model')) for a in arms):
            arms.append(spec)
    return arms


def prepare(output):
    verify_protection()
    selection=read(OUT/'s3/selection.json');models=read(OUT/'s2/models.json')
    contract=read(DEV/'inputs/contract.json');safety=selection['guard_safety']
    arms=arms_for(selection)
    inputs=build(output)
    runs=[]
    windows=list(inputs)
    # Feasibility first. Remaining arms balanced within window/repeat blocks.
    for w in windows:
        for repeat in (1,2,3):
            runs.append(dict(run_id=f'all2__{w}__r{repeat}',policy='all2',window=w,repeat=repeat,gate='G-F'))
    rest=[a for a in arms if a['id']!='all2']
    for wi,w in enumerate(windows):
        for repeat in (1,2,3):
            offset=(wi+repeat-1)%len(rest)
            for a in rest[offset:]+rest[:offset]:
                runs.append(dict(run_id=f'{a["id"]}__{w}__r{repeat}',policy=a['id'],window=w,repeat=repeat,gate='P'))
    matrix=dict(schema='maxopt-v3-matrix-v1',arms=arms,runs=runs,safety=safety,guard_models=models,
        all2_first_count=9,valid_runs=len(runs),max_technical_retries=3,max_retries_per_cell=1,
        wall_hours=36,canaries=2,canary_max_seconds=900,drift_restarts=6,drift_max_wall_seconds=3600,
        observation_seconds=2100,model_selection=selection['model'],gpu_authorized=False,
        P_selected_deduplicated=len(arms)==4 and selection['P_selected'] is not None)
    save(output/'matrix.json',matrix)
    save(output/'contract.json',contract)
    template=read(next(CAMPAIGN.glob('all1*attempt01/run_plan.json')))
    keep=('model','setup_probe','runtime','effective_runtime_expected','limits','synthetic_api')
    save(output/'runtime_template.json',{k:template[k] for k in keep})
    predictions=[]
    for m,model in models.items():
        for arm in arms:
            for w,rows in inputs.items():
                result=Simulator(rows,contract,model,arm['policy'],safety=safety,guard_model=models[arm['guard_model']],
                                 scale=arm['scale'],reservation=arm['reservation'],allow_locked_test=True).run()
                check=audit(rows,result['events'],result['summary'],contract,model,arm['policy'],result['requests'],
                            safety=safety,guard_model=models[arm['guard_model']],scale=arm['scale'],reservation=arm['reservation'])
                folder=output/'predictions'/m/arm['id']/w
                save(folder/'result.json',{k:v for k,v in result.items() if k!='events'})
                save_lines(folder/'events.jsonl',result['events'])
                save(folder/'verification.json',check)
                predictions.append(dict(model=m,policy=arm['id'],window=w,summary=result['summary']))
    save(output/'predictions.json',dict(cells=predictions,rollouts=len(predictions),physical_observations=0,
         selection_frozen_before_workload=True,not_used_for_reselection=True))
    print('S4 prepared:',len(runs),'GPU runs;',len(predictions),'CPU predictions; no GPU executed')


def code_paths():
    files=list((ROOT/'scripts/maxopt_v3').rglob('*.py'))
    files.extend(ROOT/'scripts/maxopt_v2'/name for name in ('__init__.py','engine.py','workload.py','acquire.py',
                 'calibrated.py','reselect.py','calibrated_audit.py','prepare_n5.py','n5_analysis.py','n5_acceptance.py'))
    return sorted(set(files))


def lock(output):
    verify_protection()
    if not (output/'predictions.json').is_file():
        raise ValueError('predictions missing')
    matrix=read(output/'matrix.json');predictions=read(output/'predictions.json')
    windows=read(output/'window_selection.json')['windows']
    if (len(matrix['runs']) not in (36,45) or len({r['run_id'] for r in matrix['runs']})!=len(matrix['runs'])
            or any(r['policy']!='all2' for r in matrix['runs'][:9])
            or predictions['rollouts']!=len(matrix['arms'])*3*5):
        raise ValueError('incomplete frozen matrix')
    for i,w in enumerate(windows):
        if any(abs(w['start']-x['start'])<3600 for x in windows[i+1:]):
            raise ValueError('new-window guard-band violation')
        rows=lines(output/'inputs'/(w['id']+'.jsonl'))
        offline=[r for r in rows if r['job_type']=='offline']
        if len(offline)!=120 or sorted(r['arrival_s'] for r in offline)!=list(range(0,1800,15)):
            raise ValueError('offline workload contract')
    for path in (output/'predictions').rglob('verification.json'):
        if read(path)['status']!='PASS':raise ValueError('prediction event audit failed')
    save(output/'expected_environment.json',read(next(CAMPAIGN.glob('all1*attempt01/environment_model_snapshot.json'))))
    for kind,window in [('canary_sparse','train_steady_v2'),('canary_recovery','train_recovery_v2')]:
        save(output/'engineering_inputs'/(kind+'.json'),[r for r in lines(DEV/'inputs/C2'/(window+'.jsonl')) if r['arrival_s']<600])
    save(output/'cpu_gate.json',dict(status='PASS_WITH_RECORDED_CPU_ENGINEERING_RERUN',
         model_gate=read(OUT/'s2/gate.json')['status'],selected_model='E',matrix_runs=len(matrix['runs']),
         prediction_rollouts=predictions['rollouts'],new_gpu_runs=0,
         scientific_selection_unchanged=True,engineering_rerun_upper_bound=888,
         final_s3_rollouts=read(OUT/'s3/selection.json')['unique_rollouts'],
         deviation_record='docs/max_optimization_v3_20260920/S3_EXECUTION_20260921.md'))
    save(output/'prediction_lock.json',dict(schema='maxopt-v3-freeze-v1',status='G-L_PASS_CPU_ONLY',
         preregistration_sha256=digest(DOC/'V3_PREREGISTRATION.md'),selection_sha256=digest(OUT/'s3/selection.json'),
         files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
         code={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},gpu_authorized=False,
         remaining='Claude G-L procedural release and GPU scheduling; S5a real drift/canary gate'))
    print(verify_lock(output))


def verify_lock(output):
    output=Path(output)
    frozen=read(output/'prediction_lock.json')
    for name,expected in frozen['files'].items():
        path=output/name
        if '..' in Path(name).parts or path.is_symlink() or digest(path)!=expected:
            raise ValueError('frozen input changed: '+name)
    for name,expected in frozen['code'].items():
        if '..' in Path(name).parts or digest(ROOT/name)!=expected:
            raise ValueError('frozen code changed: '+name)
    return dict(status='PASS',files=len(frozen['files']),code=len(frozen['code']),lock_sha256=digest(output/'prediction_lock.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','lock','check'))
    parser.add_argument('--output',type=Path,default=OUT/'s4')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.output)
    elif args.action=='lock':lock(args.output)
    else:print(verify_lock(args.output))
