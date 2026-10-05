"""Build manuscript assets/tables from accepted N5 data, preserving the v1 paper."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[2]
ANALYSIS=ROOT/'docs/maxopt_n5_cpu_analysis_20260920'
DOC=ROOT/'docs/maxopt_n9_paper_20260920'
REGIMES=('steady','recovery','burst_offline')
NAMES=('Steady','Recovery','Burst')
POLICIES=('all1','all2','hysteresis_u8_d0_c60')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(name):
    with (ANALYSIS/'data'/name).open() as f:return list(csv.DictReader(f))


def selected(rows,**keys):return [r for r in rows if all(r[k]==str(v) for k,v in keys.items())]


def mean(rows,key):return statistics.mean(float(r[key]) for r in rows)


def write(path,text):
    with path.open('x') as f:f.write(text)


def build(paper):
    paper=Path(paper);paper.mkdir(parents=True,exist_ok=True)
    (paper/'figures').mkdir(exist_ok=False);(paper/'generated').mkdir(exist_ok=False)
    v1={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'paper-overleaf').rglob('*')) if p.is_file()}
    for p in (ANALYSIS/'figures').glob('*.pdf'):shutil.copyfile(p,paper/'figures'/p.name)
    for name in ('mlsys2025.sty','mlsys2025.bst'):shutil.copyfile(ROOT/'paper-overleaf'/name,paper/name)
    rows=read_csv('run_metrics.csv');errors=read_csv('prediction_errors.csv')
    result=[];macros=[];comparisons=[]
    for regime,label in zip(REGIMES,NAMES):
        window='locked_test_'+regime+'_v2'
        dynamic=selected(rows,window=window,policy=POLICIES[2]);baseline=selected(rows,window=window,policy='all1')
        for key,macro in [('window_total_cost','Savings'),('gpu_occupied_seconds','ResourceSavings'),('deployment_total_cost','DeploymentSavings')]:
            val=100*(1-mean(dynamic,key)/mean(baseline,key))
            macros.append('\\newcommand{\\'+label+macro+'}{'+f'{val:.1f}'+r'\%}')
        for policy,policy_name in zip(POLICIES,('All1','All2','Dynamic')):
            rs=sorted(selected(rows,window=window,policy=policy),key=lambda r:int(r['repeat']))
            total=mean(rs,'window_total_cost');mn=min(float(r['window_total_cost']) for r in rs);mx=max(float(r['window_total_cost']) for r in rs)
            offline='/'.join(r['offline_on_time'] for r in rs)
            result.append(f'{label} & {policy_name} & {total:.3f} [{mn:.3f}, {mx:.3f}] & {mean(rs,"gpu_occupied_seconds"):.1f} & {offline} '+r'\\')
        for model in ('original','calibrated'):
            rs=selected(errors,window=window,policy=POLICIES[2],model=model)
            comparisons.append(f'{label} & {model.capitalize()} & {mean(rs,"shared_local_service_mae_seconds"):.3f} & '+
               f'{100*mean(rs,"shared_local_coverage_all_arrivals"):.1f} & {mean(rs,"gpu_occupied_seconds_absolute_error"):.1f} & '+
               f'{mean(rs,"gpu_active_seconds_absolute_error"):.1f} & {100*mean(rs,"target_disagreement_rate"):.1f} '+r'\\')
    write(paper/'generated/macros.tex','% Generated from accepted N5 CSVs.\n'+'\n'.join(macros)+'\n')
    write(paper/'generated/cost_rows.tex','\n'.join(result)+'\n')
    write(paper/'generated/error_rows.tex','\n'.join(comparisons)+'\n')
    runlines=[]
    for regime,label in zip(REGIMES,NAMES):
        for policy,policy_name in zip(POLICIES,('All1','All2','Dynamic')):
            for r in sorted(selected(rows,window='locked_test_'+regime+'_v2',policy=policy),key=lambda r:int(r['repeat'])):
                runlines.append(f'{label} & {policy_name} & {r["repeat"]} & {r["online_on_time"]}/{r["online_arrivals"]} & '+
                   f'{r["offline_on_time"]}/120 & {int(r["api_accepted"])} & {float(r["online_e2e_p95_seconds"]):.2f} & '+
                   f'{float(r["online_e2e_p99_seconds"]):.2f} & {float(r["window_total_cost"]):.6f} & '+
                   f'{float(r["deployment_total_cost"]):.6f} '+r'\\')
    write(paper/'generated/run_rows.tex','\n'.join(runlines)+'\n')
    # Public primary metadata supplies all authors; preprints remain preprints.
    keys=('vidur2024','charon2026','kwon2023vllm','wang2024burstgpt','qiu2025hygen','sun2024llumnix','chen2023frugalgpt','gujarati2020clockwork_arxiv')
    bibliography=[];bindings={}
    for key in keys:
        source=DOC/('literature_fallback' if key.endswith('_arxiv') else 'literature_sources')/(key+'.json')
        r=json.loads(source.read_text());m=r['metadata'];bindings[str(source.relative_to(ROOT))]=sha(source)
        title=m['citation_title'][0];authors=' and '.join(m['citation_author'])
        if key in ('vidur2024','charon2026'):
            year=m['citation_publication_date'][0][:4]
            entry='@inproceedings{'+key+',\n'
            fields=dict(title='{'+title+'}',author=authors,year=year,booktitle='Proceedings of Machine Learning and Systems',
                        volume=m['citation_volume'][0],pages=m['citation_firstpage'][0]+'--'+m['citation_lastpage'][0],url=r['url'])
        else:
            entry='@misc{'+key+',\n'
            fields=dict(title='{'+title+'}',author=authors,year=m['citation_date'][0][:4],
                        eprint=m['citation_arxiv_id'][0],archivePrefix='arXiv',url=r['url'])
        bibliography.append(entry+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items())+'\n}\n')
    write(paper/'references.bib','\n'.join(bibliography))
    record=dict(status='ASSETS_PREPARED',v1_files=v1,input_analysis_manifest=sha(ANALYSIS/'data/analysis_manifest.json'),
                input_figures_manifest=sha(ANALYSIS/'figures/figure_manifest.json'),literature=bindings,
                generated={str(p.relative_to(paper)):sha(p) for p in sorted(paper.rglob('*')) if p.is_file()},
                source_sha256=sha(__file__),no_new_experiments=True)
    write(DOC/'material_preparation.json',json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],protected_v1_files=len(v1),references=len(keys),run_table_rows=len(runlines)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper',type=Path,default=ROOT/'paper-maxopt-v2')
    a=p.parse_args();build(a.paper)
