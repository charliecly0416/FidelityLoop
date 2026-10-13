"""Prepare one frozen physical cell at the GPU site; does not launch it."""
import argparse
import shutil
import sys
from pathlib import Path

from .common import ROOT,read,lines,digest,save
from .freeze import verify_lock


def prepare(locked,run_id,out,environment,identities,attempt=1):
    locked,out=Path(locked).resolve(),Path(out).resolve()
    verify_lock(locked)
    matrix=read(locked/'matrix.json')
    cell=next(c for c in matrix['runs'] if c['run_id']==run_id)
    arm=next(a for a in matrix['arms'] if a['id']==cell['policy'])
    if attempt not in (1,2):
        raise ValueError('at most one technical retry per cell')
    env=read(environment);expected=read(locked/'expected_environment.json')
    if ({r['name']:r['sha256'] for r in env['model_files']}!={r['name']:r['sha256'] for r in expected['model_files']}
            or env['packages']!=expected['packages']):
        raise ValueError('weight/tokenizer/runtime differs; return to CPU before new test')
    out.mkdir(parents=True,exist_ok=False)
    rows=lines(locked/'inputs'/(cell['window']+'.jsonl'))
    save(out/'expected_requests.json',rows)
    save(out/'environment_model_snapshot.json',env)
    lock=read(locked/'prediction_lock.json')
    for name in lock['code']:
        target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    template=read(locked/'runtime_template.json')
    plan=dict(template,schema='maxopt-v3-run-v1',run_id=run_id,attempt=attempt,arm=arm,
        window_id=cell['window'],repeat=cell['repeat'],window_seconds=1800,drain_seconds=300,
        locked_root=str(locked),python=sys.executable,model=env['model_path'],
        gpus=[0,1],gpu_identity_expected=read(identities),safety=matrix['safety'],
        guard_model=matrix['guard_models'][arm['guard_model']],
        expected_requests_sha256=digest(out/'expected_requests.json'),prediction_lock_sha256=digest(locked/'prediction_lock.json'))
    save(out/'run_plan.json',plan)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--identities',type=Path,required=True);p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();prepare(a.locked,a.run_id,a.output,a.environment,a.identities,a.attempt)
    print('PREPARED; no GPU launched')
