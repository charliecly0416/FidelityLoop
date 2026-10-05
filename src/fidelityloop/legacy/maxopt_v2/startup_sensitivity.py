"""Post-selection validation diagnostic; never selects or edits N5 policies."""
import argparse
import copy
import json
import time
from pathlib import Path

from .reselect import check_bindings, digest, execute_cell, lines, read, save, ROOT, DOC


def register(frozen, output):
    frozen,output=Path(frozen),Path(output)
    registry=check_bindings(frozen)
    plan=read(frozen/'n5_plan.json')
    lifecycle=read(frozen/'lifecycle_estimates.json')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='maxopt-postselection-startup-sensitivity-v1',
                scope='validation only; explanatory diagnostic, prohibited for re-selection',
                main_registration_sha256=digest(frozen/'registration.json'),
                n5_plan_sha256=digest(frozen/'n5_plan.json'),
                source_sha256=digest(Path(__file__)),
                analysis_design_sha256=digest(ROOT/DOC/'N5_PAPER_EXPERIMENT_DESIGN_20260919.md'),
                policies=plan['policies'],windows=registry['windows'],
                scenarios=dict(development_p90=lifecycle['startup']['p90'],historical_slow_approximation=285.56),
                reused_main_median_cells=9,new_cpu_runs=18,policy_ticks=37800,
                price_or_service_changes=False,locked_test_opened=False)
    save(output/'registration.json',result)
    return result


def run(frozen, output):
    frozen,output=Path(frozen),Path(output)
    check_bindings(frozen)
    reg=read(output/'registration.json')
    for path,expected in ((frozen/'registration.json',reg['main_registration_sha256']),
                          (frozen/'n5_plan.json',reg['n5_plan_sha256']),
                          (Path(__file__),reg['source_sha256']),
                          (ROOT/DOC/'N5_PAPER_EXPERIMENT_DESIGN_20260919.md',reg['analysis_design_sha256'])):
        if digest(path)!=expected:
            raise ValueError('diagnostic preregistration binding changed')
    if len(reg['policies'])!=3:
        raise ValueError('this diagnostic requires the frozen three-policy branch')
    base=read(frozen/'models.json')['calibrated']
    contract=read(frozen/'inputs/contract.json')
    main=read(frozen/'search_results.json')
    ids={p['id'] for p in reg['policies']}
    cells=[dict(scenario='main_median_reused',policy=c['configuration'],window=c['window'],summary=c['summary'])
           for c in main['cells'] if c['model']=='calibrated' and c['configuration'] in ids]
    begin=time.perf_counter()
    for scenario,startup in reg['scenarios'].items():
        model=copy.deepcopy(base); model['startup_seconds']=startup
        for policy in reg['policies']:
            for window in reg['windows']:
                rows=lines(frozen/f'inputs/C2/{window}.jsonl')
                cell=execute_cell(output,f'{scenario}/{policy["id"]}/{window}',rows,contract,model,policy['policy'])
                cell.update(scenario=scenario,policy=policy['id'],window=window)
                cells.append(cell)
    result=dict(registration_sha256=digest(output/'registration.json'),cells=cells,
                new_runs=18,reused_runs=9,wall_seconds=time.perf_counter()-begin,
                changes_to_main_selection=False,changes_to_n5_plan=False)
    save(output/'results.json',result)
    return dict(status='COMPLETE',new_runs=18,reused_runs=9,
                dynamic_cells=[dict(scenario=c['scenario'],window=c['window'],feasible=c['summary']['feasible'],
                                    offline_on_time=c['summary']['populations']['offline']['on_time'],
                                    cost=c['summary']['costs']['total']) for c in cells if c['policy'] not in ('all1','all2')])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('register','run'))
    p.add_argument('--frozen',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(register(a.frozen,a.output) if a.command=='register' else run(a.frozen,a.output),indent=2))
