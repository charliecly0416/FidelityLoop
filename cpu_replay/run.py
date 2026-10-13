"""Portable entry point for the fixed, previously reported CPU replay cases."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def equal(actual, expected, where='root'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for k in expected:
            equal(actual[k], expected[k], where+'.'+k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, where+f'[{i}]')
    elif type(expected) is float:
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), (where, actual, expected)
    else:
        assert actual == expected, (where, actual, expected)

def verify_files():
    manifest = json.loads((ROOT/'FILES_SHA256.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name

def check():
    with gzip.open(ROOT/'expected/predictions.json.gz', 'rt') as f:
        expected = json.load(f)
    count = 0
    for case, old in expected.items():
        folder = ROOT/'runtime'/case
        for name in ['records', 'summary']:
            equal(json.loads((folder/f'{name}.json').read_text()), old[name], case+'.'+name)
        decisions = [json.loads(line)['decision'] for line in (folder/'decisions.jsonl').read_text().splitlines()]
        equal(decisions, old['decisions'], case+'.decisions')
        count += len(decisions)
    print(json.dumps({'status': 'PASS_PORTABLE_NATIVE_REPLAY', 'cases': len(expected), 'decisions': count}))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['verify', 'engineering', 'historical', 'check', 'test'])
    args = p.parse_args()
    verify_files()
    if args.stage == 'verify':
        print('PASS_RELEASE_FILE_HASHES')
        return
    (ROOT/'runtime').mkdir(exist_ok=True)
    if args.stage in ('engineering', 'historical'):
        command = [sys.executable, '-I', '-B', str(ROOT/'scripts/run_precheck.py'), '--phase', args.stage]
        if args.stage == 'historical':
            command += ['--case', 'all']
        subprocess.run(command, cwd=ROOT, check=True)
    elif args.stage == 'check':
        check()
    else:
        for folder in ['tests', 'analysis']:
            subprocess.run([sys.executable, '-I', '-B', '-m', 'unittest', 'discover', '-s', str(ROOT/folder), '-v'], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
