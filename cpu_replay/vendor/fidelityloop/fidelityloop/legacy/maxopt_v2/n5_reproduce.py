"""CPU-only small reproduction of an N5 result bundle, using the standard library.

This checks accepted per-request/lifecycle reductions and fixed-behavior prices.
It is not a replacement for the full RAW verifier or for a new physical replay.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def verify(root):
    root=Path(root).resolve()
    data_root=root/'docs/maxopt_n5_cpu_analysis_20260920/data'
    manifest=json.loads((root/'bundle_manifest.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
            and p.name!='bundle_manifest.json'}
    if actual!=set(manifest['files']):raise ValueError('bundle inventory mismatch')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('SHA mismatch: '+name)
    def read(name):
        with (data_root/name).open() as f:return list(csv.DictReader(f))
    checks=0
    def equal(actual,expected):
        nonlocal checks
        if not math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-7):
            raise ValueError(f'{actual} != {expected}')
        checks+=1
    raw=json.loads((root/'evidence/raw_recomputed.json').read_text())
    runs=read('run_metrics.csv');index={r['run_id']:r for r in runs}
    if len(raw)!=27 or set(index)!=set(raw):raise ValueError('27 runs required')
    for rid,data in raw.items():
        row=index[rid];requests=data['requests'];phases=data['phases']
        origin=data['origin_ns'];end=origin+2100*10**9
        occupied=sum(max(0,min(g['release'],end)-max(g['launch'],origin))/1e9
                     for g in data['generations'])
        starts=sum(origin<=g['launch']<end for g in data['generations'])
        stops=sum(origin<=g['shutdown']<end for g in data['generations'])
        api=[r for r in requests.values() if r['route']=='synthetic_api']
        api_cost=sum(r['input_tokens']*1e-6+r['output_tokens']*3e-6 for r in api)
        misses=sum(r['job_type']=='offline' and not r['timely'] for r in requests.values())
        for field,value in dict(gpu_occupied_seconds=occupied,startup_events_in_window=starts,
                                shutdown_events_in_window=stops,api_accept_cost=api_cost,
                                api_accepted=len(api),offline_deadline_miss_penalty=.01*misses).items():
            equal(row[field],value)
        for job in ('online','offline'):
            rs=[r for r in requests.values() if r['job_type']==job]
            ontime=sum(r['status']=='completed' and r['completed_at'] is not None
                       and r['completed_at']<=(r['arrival_s']+60 if job=='online' else r['deadline_s']) for r in rs)
            equal(row[job+'_arrivals'],len(rs));equal(row[job+'_on_time'],ontime)
        total=.001*occupied+.002*starts+.001*stops+api_cost+.01*misses
        equal(row['window_total_cost'],total)
        equal(row['deployment_total_cost'],total+phases['setup']['holding_transition_cost']+
              phases['cleanup']['holding_transition_cost'])
    repriced=read('repriced_runs.csv');price_index={}
    for r in repriced:
        s=index[r['run_id']];g=float(r['gpu_multiplier']);a=float(r['api_multiplier'])
        total=float(s['gpu_occupied_seconds'])*.001*g+float(s['api_accept_cost'])*a
        total+=.002*int(s['startup_events_in_window'])+.001*int(s['shutdown_events_in_window'])
        total+=float(s['offline_deadline_miss_penalty'])
        equal(r['total_cost'],total)
        price_index[(r['window'],r['policy'],r['repeat'],g,a)]=total
    groups=read('price_group_statistics.csv')
    for r in groups:
        w=r['window'];g=float(r['gpu_multiplier']);a=float(r['api_multiplier'])
        d=[price_index[(w,'hysteresis_u8_d0_c60',str(i),g,a)] for i in (1,2,3)]
        b=[price_index[(w,r['baseline'],str(i),g,a)] for i in (1,2,3)]
        equal(r['savings_ratio_of_means'],1-statistics.mean(d)/statistics.mean(b))
    if len(repriced)!=243 or len(groups)!=54:raise ValueError('incomplete price grid')
    timelines=read('timeline_repeat1.csv')
    for w in {r['window'] for r in timelines}:
        ticks=[int(r['tick']) for r in timelines if r['window']==w]
        if sorted(ticks)!=list(range(2100)):raise ValueError('incomplete timeline')
    report=dict(status='PASS',files=len(actual),numeric_checks=checks,runs=len(runs),
                requests=sum(len(d['requests']) for d in raw.values()),repriced_cells=len(repriced),
                scope='small accepted-data reproduction; original GPU RAW is an external input',
                no_GPU=True)
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--render-to',type=Path)
    args=parser.parse_args();verify(args.root)
    if args.render_to:
        subprocess.run([sys.executable,str(args.root/'source/n5_figures.py'),
                        '--data',str(args.root/'docs/maxopt_n5_cpu_analysis_20260920/data'),
                        '--output',str(args.render_to)],check=True)
