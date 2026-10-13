"""CPU engineering checks and predeclared historical input runs. No outcome access."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import traceback
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'analysis'))
from controlled_vidur import ControlledVidur, native_reference
from verify_ledger import audit

def dump(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def lines(path,rows): path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
def rows(bursts, output=128):
    return [dict(request_id=f'r{n:03}',arrival_s=at,deadline_s=at+15,
        job_type='offline' if n%3==0 else 'online',input_tokens=2048 if n%2 else 199,max_output_tokens=output)
        for n,at in enumerate(at for at,count in bursts for _ in range(count))]

def execute(name,spec, *, targets=None, reference=False):
    out=ROOT/'runtime'/name
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    dump(out/'config.json',spec)
    run=None
    try:
        run=ControlledVidur(out=out,targets=targets,**spec)
        summary=run.run()
        if reference:
            native=native_reference(run.rows,out/'reference',run.admissions)
            errors=[abs(native[a['request_id']]['completed_at']-run.records[a['request_id']]['completed_at']) for a in run.admissions]
            assert errors and max(errors)<1e-12
            summary['conditional_admission_max_error']=max(errors)
            dump(out/'conditional_reference.json',native)
        dump(out/'summary.json',summary)
    except Exception:
        (out/'FAILURE.txt').write_text(traceback.format_exc())
        raise
    finally:
        if run is not None:
            dump(out/'records.json',run.records)
            lines(out/'events.jsonl',run.events); lines(out/'decisions.jsonl',run.decisions)
            lines(out/'admissions.jsonl',run.admissions)
            with gzip.open(out/'native_events.jsonl.gz','wt') as f:
                for e in run.native_events: f.write(json.dumps(e)+'\n')
            dump(out/'native_requests.json',{rid:r.to_dict() for rid,r in run.requests.items()})
    receipt=audit(out)
    dump(out/'LEDGER_AUDIT.json',receipt)
    print(json.dumps(dict(case=name,status='PASS',completed=summary['metrics']['completed'],
        occupied_seconds=summary['occupied_seconds'],domain=summary['domain_audit'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['engineering','historical'],required=True)
    p.add_argument('--case');a=p.parse_args()
    if a.phase=='engineering':
        fixed=dict(rows=rows([(0,8),(1,4),(2,4),(8,8)],256),horizon=30,initial='active',policy='all2')
        execute('static_equivalence',fixed,reference=True)
        execute('lifecycle_fixture',dict(fixed,horizon=50),targets=lambda t: 2 if t<1 or t>=10 else 0,reference=True)
        execute('api_fixture',dict(rows=rows([(0,24)]),horizon=18,policy='all2'),targets=lambda t:0)
        execute('cutoff_fixture',dict(rows=rows([(0,8)],256),horizon=1,initial='active',policy='all2'))
        execute('mixed_fixture',dict(rows=rows([(0,2),(1,4),(2,2)],256),horizon=20,initial='active',policy='all2'),reference=True)
        base=rows([(0,20),(65,8)],256)
        execute('prefix_base',dict(rows=base,horizon=120,policy='guard'))
        execute('prefix_future',dict(rows=base+[dict(rows([(100,1)])[0],request_id='future')],horizon=120,policy='guard'))
    else:
        cases=json.loads((ROOT/'inputs/cases.json').read_text())['cases']
        assert a.case == 'all' or a.case in [c['case_id'] for c in cases]
        def timeout(*_): raise TimeoutError('predeclared 180-second case limit')
        signal.signal(signal.SIGALRM,timeout)
        for c in cases:
            if a.case != 'all' and c['case_id'] != a.case: continue
            signal.alarm(180)
            try: execute(c['case_id'],c['spec'])
            finally: signal.alarm(0)

if __name__=='__main__':main()
