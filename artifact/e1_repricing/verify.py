"""Check a fixed package inventory and its numerically reproduced reference."""
import hashlib
import json
import math
from pathlib import Path
import analysis

HOME = Path(__file__).resolve().parent
FILES = {'LICENSE','README.md','analysis.py','verify.py','test_analysis.py','reference_results.json','data/ledger.csv','data/prices.json','data/requests.json','data/runs.json'}


def check_manifest(directory=HOME):
    manifest = json.loads((directory/'MANIFEST.json').read_text())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if set(manifest['files']) != FILES or actual != FILES|{'MANIFEST.json'}:
        raise ValueError('package members differ from the fixed inventory')
    if any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('symbolic links are not artifact members')
    for name,record in manifest['files'].items():
        value = (directory/name).read_bytes()
        if hashlib.sha256(value).hexdigest()!=record['sha256'] or len(value)!=record['bytes']:
            raise ValueError('package bytes changed: '+name)
    return manifest


def compare(actual, expected):
    errors = []
    numbers = 0
    def walk(left,right,location):
        nonlocal numbers
        if isinstance(right,dict):
            if not isinstance(left,dict) or set(left)!=set(right):
                raise ValueError('result field mismatch at '+location)
            for key in right:walk(left[key],right[key],location+'/'+key)
        elif isinstance(right,list):
            if not isinstance(left,list) or len(left)!=len(right):
                raise ValueError('result list mismatch at '+location)
            for i,(a,b) in enumerate(zip(left,right)):walk(a,b,location+'/'+str(i))
        elif isinstance(right,(int,float)) and not isinstance(right,bool):
            if not isinstance(left,(int,float)) or not math.isfinite(left) or not math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('numerical disagreement at '+location)
            numbers+=1;errors.append(abs(left-right))
        elif left!=right:
            raise ValueError('result label mismatch at '+location)
    walk(actual,expected,'result')
    return {'numeric_values_checked':numbers,'maximum_absolute_error':max(errors,default=0),'relative_tolerance':1e-12,'absolute_tolerance':1e-12}


def main():
    manifest = check_manifest()
    result = analysis.calculate()
    agreement = compare(result,json.loads((HOME/'reference_results.json').read_text()))
    print(json.dumps({'status':'PASS_REDUCED_LEDGER_REPRICING','manifest_files':len(manifest['files']),'formal_ledgers':result['formal_count'],'comparison_ledgers':result['comparison_count'],'window_phase_results':len(result['results']),'reference_comparison':agreement}))


if __name__ == '__main__':
    main()
