"""Read-only campaign progression: physical feasibility before cost experiments."""
import argparse
from pathlib import Path
from datetime import datetime,timezone

from .common import read,digest
from .freeze import verify_lock
from .acceptance import verify
from .gpu_engineering import verify as verify_engineering


def next_cell(locked,campaign,engineering):
    verify_lock(locked)
    matrix=read(locked/'matrix.json')
    probes={k:verify_engineering(engineering/k,locked) for k in ('drift','canary_sparse','canary_recovery')}
    if any(r['status']!='PASS' for r in probes.values()):
        return dict(status='HOLD_G_R',engineering=probes)
    if (sum(probes[k]['guard_ticks'] for k in ('canary_sparse','canary_recovery'))==0
            or sum(probes[k]['api_accepts'] for k in ('canary_sparse','canary_recovery'))==0):
        return dict(status='HOLD_CANARY_COVERAGE',reason='two canaries must exercise real deadline guard and API route')
    elapsed=sum(read(engineering/k/'engineering_result.json')['wall_seconds'] for k in probes)
    accepted=[];retry_count=0;next_run=None
    for cell in matrix['runs']:
        attempts=sorted(campaign.glob(cell['run_id']+'__attempt*'))
        if len(attempts)>2:
            return dict(status='HOLD_BUDGET',reason='more than one retry in cell')
        retry_count+=max(0,len(attempts)-1)
        success=None
        for path in attempts:
            report=verify(path,locked)
            events_path=path/'controller_events.jsonl'
            if events_path.is_file():
                from .common import lines
                events=lines(events_path)
                if events:elapsed+=(events[-1]['controller_monotonic_ns']-events[0]['controller_monotonic_ns'])/1e9
            if report['status']=='PASS':
                if success is not None:return dict(status='HOLD_DUPLICATE_VALID_ATTEMPT',run_id=cell['run_id'])
                success=report
        if success:
            accepted.append(cell['run_id'])
            if cell['gate']=='G-F' and not success['G_F']:
                return dict(status='STOP_G_F_SCIENTIFIC_FAILURE',run_id=cell['run_id'],populations=success['populations'],
                            reason='Do not retry scientific failure, replace window, or continue cost claim.')
        elif next_run is None:
            if len(attempts)>=2:
                return dict(status='HOLD_TECHNICAL_RETRY_EXHAUSTED',run_id=cell['run_id'])
            next_run=dict(cell,attempt=len(attempts)+1)
    if retry_count>3 or (next_run and next_run['attempt']==2 and retry_count>=3):
        return dict(status='HOLD_BUDGET',reason='campaign retry cap')
    from .common import lines
    started=min(datetime.fromisoformat(lines(engineering/k/'controller_events.jsonl')[0]['utc']) for k in probes)
    wall=(datetime.now(timezone.utc)-started).total_seconds()
    if next_run and max(elapsed,wall)+3420>36*3600:
        return dict(status='PARTIAL_WALL_BUDGET',elapsed_seconds=elapsed,wall_seconds=wall,accepted=accepted)
    return dict(status='READY_NEXT_CELL' if next_run else 'COMPLETE_RETURN_TO_CPU',next=next_run,
                accepted=accepted,technical_retries=retry_count,accounted_execution_seconds=elapsed,wall_seconds=wall)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--engineering',type=Path,required=True)
    a=p.parse_args()
    import json
    print(json.dumps(next_cell(a.locked,a.campaign,a.engineering),indent=2))
