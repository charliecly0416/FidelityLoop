"""Check internal V2 paper claims/assets and render PDFs; never run a GPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/maxopt_n9_paper_20260920'
PAPER = ROOT / 'paper-maxopt-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(output, render=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads((DOC / 'material_preparation.json').read_text())
    for name, digest in preparation['v1_files'].items():
        assert sha(ROOT / name) == digest, ('v1 changed', name)
    revision_path = DOC / 'revision_materials_r2.json'
    revision = json.loads(revision_path.read_text()) if revision_path.exists() else None
    for name, digest in preparation['generated'].items():
        if revision and name == 'references.bib':
            digest = revision['bibliography_sha256']
        assert sha(PAPER / name) == digest, ('generated asset changed', name)
    if revision:
        for name, digest in revision['generated'].items():
            assert sha(PAPER / name) == digest, name
        for name, digest in revision['source_records'].items():
            assert sha(ROOT / name) == digest, name
        ablation = DOC / 'ablation_r2'
        assert sha(ablation / 'manifest.json') == revision['ablation_manifest_sha256']
        with (ablation / 'groups.csv').open() as f:
            groups = list(csv.DictReader(f))
        assert len(groups) == 36
        display = (PAPER / 'generated/ablation_full_rows.tex').read_text().splitlines()
        assert len(display) == 36
        for row, line in zip(groups, display):
            # Table ordering is policy/window/cell, independent of CSV ordering.
            fields = [v.strip() for v in line.removesuffix('\\\\').split('&')]
            policy = {'All1': 'all1', 'All2': 'all2', 'Dynamic': 'hysteresis_u8_d0_c60'}[fields[0]]
            regime = {'Steady': 'steady', 'Recovery': 'recovery', 'Burst': 'burst_offline'}[fields[1]]
            cell = {'O/30': 'original', 'C/30*': 'service_only', 'O/69*': 'startup_only', 'C/69': 'calibrated'}[fields[2]]
            row = next(r for r in groups if (r['policy'], r['window'], r['cell']) ==
                       (policy, 'locked_test_' + regime + '_v2', cell))
            keys = ('occupied_predicted', 'gpu_occupied_seconds_absolute_error',
                    'shared_local_service_mae_seconds', 'shared_local_coverage_all_arrivals', 'target_disagreement_rate')
            for index, (key, scale, digits) in enumerate(zip(keys, (1, 1, 1, 100, 100), (1, 1, 3, 1, 1)), 3):
                assert fields[index] == f'{float(row[key])*scale:.{digits}f}', (line, key)
    raw_path = ROOT / 'artifacts/max_optimization_v2_20260916/n5_cpu_acceptance_20260920_r1/raw_recomputed.json'
    raw = json.loads(raw_path.read_text())
    with (ROOT / 'docs/maxopt_n5_cpu_analysis_20260920/data/run_metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    # Check generated display values against independently accepted RAW reductions.
    run_table = (PAPER / 'generated/run_rows.tex').read_text().splitlines()
    assert len(run_table) == 27
    policies = {'All1': 'all1', 'All2': 'all2', 'Dynamic': 'hysteresis_u8_d0_c60'}
    windows = {'Steady': 'steady', 'Recovery': 'recovery', 'Burst': 'burst_offline'}
    for line in run_table:
        fields = [v.strip() for v in line.removesuffix('\\\\').split('&')]
        rid = policies[fields[1]] + '__locked_test_' + windows[fields[0]] + '_v2__r' + fields[2]
        rs = list(raw[rid]['requests'].values())
        for job, index in [('online', 3), ('offline', 4)]:
            js = [r for r in rs if r['job_type'] == job]
            assert fields[index] == f'{sum(r["timely"] for r in js)}/{len(js)}'
        assert int(fields[5]) == sum(r['route'] == 'synthetic_api' for r in rs)
        row = next(r for r in rows if r['run_id'] == rid)
        assert fields[8] == f'{float(row["window_total_cost"]):.6f}'
        assert fields[9] == f'{float(row["deployment_total_cost"]):.6f}'
    misses = []
    tails = {}
    for rid, run in raw.items():
        censored = [(k, r) for k, r in run['requests'].items() if not r['timely']]
        if not censored:
            continue
        tail = [run['targets'][str(t)] for t in range(1800, 2100)]
        assert tail == [0] * 300
        tails[rid] = {'last_300_targets_all_zero': True, 'remaining_offline': len(censored)}
        for request_id, r in censored:
            assert r['job_type'] == 'offline' and r['status'] == 'censored'
            assert r['route'] is None and r['dispatched_at'] is None
            assert r['deadline_s'] <= 2100
            misses.append(dict(run_id=rid, request_id=request_id, **r))
    assert len(misses) == 12 and len(tails) == 5
    (output / 'censor_mechanism.json').write_text(json.dumps(dict(
        scope='Post-hoc diagnostic of frozen physical observations; no policy changes',
        raw_sha256=sha(raw_path), rule='P > 8 * max(1, old_target)',
        affected_runs=tails, requests=misses), indent=2) + '\n')
    body = (PAPER / 'body.tex').read_text()
    sources = body + (PAPER / 'abstract.tex').read_text()
    cited = set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', sources) for k in group.split(','))
    bibkeys = set(re.findall(r'@\w+\{([^,]+),', (PAPER / 'references.bib').read_text()))
    citation_count = revision['citation_count'] if revision else 8
    assert cited == bibkeys and len(cited) == citation_count
    labels = re.findall(r'\\label\{([^}]+)\}', body)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r'\\ref\{([^}]+)\}', body)) <= set(labels)
    pdfs = {}
    for variant, stem in [('mlsys', 'main_mlsys'), ('standard', 'main'), ('supplement', 'supplement')]:
        pdf = PAPER / 'build' / variant / (stem + '.pdf')
        log = pdf.with_suffix('.log').read_text()
        assert not any(s in log for s in ['undefined references', 'undefined citations', 'duplicate ignored', 'Misplaced \\noalign'])
        text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
        pages = text.split('\f')
        if not pages[-1].strip():
            pages.pop()
        (output / (stem + '.txt')).write_text(text)
        reference_page = next((i + 1 for i, p in enumerate(pages) if 'R EFERENCES' in p or 'References' in p), None)
        if variant == 'mlsys':
            assert reference_page is not None and reference_page <= 10
        pdfs[variant] = dict(pages=len(pages), references_begin=reference_page,
                            pdf_sha256=sha(pdf), log_sha256=sha(pdf.with_suffix('.log')),
                            overfull=re.findall(r'Overfull[^\n]+', log))
        if render:
            subprocess.run(['pdftoppm', '-r', '95', '-png', str(pdf), str(output / stem)], check=True)
    result = dict(status='PASS', protected_v1_files=len(preparation['v1_files']),
                  unchanged_generated_assets=len(preparation['generated']) - bool(revision),
                  revised_bibliography=bool(revision), ablation_rows_checked=36 if revision else 0,
                  physical_rows_checked=27, citations=citation_count, censored=12, affected_runs=5,
                  pdfs=pdfs, no_gpu=True, external_review='NOT_PERFORMED')
    (output / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    check(args.output, args.render)
