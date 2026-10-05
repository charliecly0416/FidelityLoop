"""Extract the CPU bundle away from the repository and exercise its public commands."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def check(archive, report):
    archive,report=Path(archive).resolve(),Path(report).resolve()
    work=Path(tempfile.mkdtemp(prefix='n5_cpu_clean_'))
    root=work/'bundle';root.mkdir()
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('archive CRC')
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe archive member')
        z.extractall(root)
    def run(arguments,expected=0):
        r=subprocess.run([sys.executable,'-I',*arguments],cwd=root,capture_output=True,text=True,timeout=90)
        if r.returncode!=expected:raise ValueError(r.stdout+r.stderr)
        return dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    reproduce=run(['source/n5_reproduce.py','--root','.','--render-to',str(work/'rendered')])
    workload=run(['source/n5_workload_disclosure.py','--lock','evidence/frozen_workload','--output',str(work/'workload')])
    doc=root/'docs/maxopt_n5_cpu_analysis_20260920'
    images={}
    for name in ('cost_slo','prediction_error','lifecycle_repeat1','price_sensitivity'):
        expected=(doc/'figures'/(name+'.png')).read_bytes()
        observed=(work/'rendered'/(name+'.png')).read_bytes()
        if observed!=expected:raise ValueError('PNG differs: '+name)
        images[name]='byte-identical PNG'
    if (work/'workload/raw_derived_burstiness.csv').read_bytes()!=(doc/'workload_disclosure/raw_derived_burstiness.csv').read_bytes():
        raise ValueError('workload disclosure changed')
    table=doc/'data/repriced_runs.csv';original=table.read_bytes()
    table.write_bytes(original+b'\n')
    tamper=run(['source/n5_reproduce.py','--root','.'],expected=1)
    if 'SHA mismatch' not in tamper['stderr']:raise ValueError('wrong tamper rejection')
    table.write_bytes(original)
    # A changed table plus a self-consistent updated SHA must still fail arithmetic.
    manifest_path=root/'bundle_manifest.json';manifest_bytes=manifest_path.read_bytes()
    manifest=json.loads(manifest_bytes)
    rows=list(csv.DictReader(io.StringIO(original.decode())))
    rows[0]['total_cost']=str(float(rows[0]['total_cost'])+1)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);table.write_text(stream.getvalue())
    manifest['files'][str(table.relative_to(root))]=hashlib.sha256(table.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    arithmetic=run(['source/n5_reproduce.py','--root','.'],expected=1)
    if ' != ' not in arithmetic['stderr']:raise ValueError('wrong arithmetic rejection')
    table.write_bytes(original);manifest_path.write_bytes(manifest_bytes)
    restored=run(['source/n5_reproduce.py','--root','.'])
    result=dict(status='PASS',archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        clean_directory=str(work),isolated_python=True,reproduction=reproduce,workload=workload,
        PNG_comparisons=images,negative_checks=['SHA tamper rejected','repriced cost spoof with updated SHA rejected'],
        restored_reproduction=restored,
        scope='fresh directory and isolated Python; not a new full repository checkout or independent human review')
    with report.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reproduction','workload','restored_reproduction')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();check(a.archive,a.report)
