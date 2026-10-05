"""Generate R1 bibliography and R2 display tables from verified source records."""
import csv
import json
from pathlib import Path
import re
import shutil

from .n9_ablation import BASE, DEFAULT, DOC, REGIMES
from .n9_revision_sources import ARXIV
from .reselect import ROOT, digest, lines, read

PAPER = ROOT / 'paper-maxopt-v2'
LITERATURE = DOC / 'literature_revision_r2'
ABLATION = BASE / 'n9_posthoc_ablation_20260920_r1'


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def latex(text):
    for a, b in [('Í', r"{\'I}"), ('ñ', r'{\~n}'), ('é', r"{\'e}"), ('å', r'{\aa}'), ('&', r'\&')]:
        text = text.replace(a, b)
    return text


def build():
    metadata_files = []
    # Preserve the eight original entries; regenerate only this revision's additions.
    bib = (PAPER / 'references.bib').read_text().split('\n@misc{miao2023spotserve', 1)[0].rstrip() + '\n'
    entries = []
    for key in ARXIV:
        path = LITERATURE / (key + ('_retry' if key == 'cui2026opscale' else '') + '.json')
        record = read(path)
        m = record['metadata']
        metadata_files.append(path)
        authors = m['citation_author']
        if key == 'wan2025bros':
            # arXiv citation meta reverses family/given names. The PDF title page
            # independently supplies "Borui Wan, Juntao Zhao, ...".
            titlepage = (LITERATURE / 'bros_titlepage.txt').read_text()
            for person in ('Borui Wan', 'Juntao Zhao', 'Chenyu Jiang', 'Chuanxiong Guo', 'Chuan Wu'):
                assert person in titlepage
            authors = ['Wan, Borui', 'Zhao, Juntao', 'Jiang, Chenyu', 'Guo, Chuanxiong', 'Wu, Chuan']
            metadata_files.append(LITERATURE / 'bros_firstparty.pdf')
        fields = dict(title='{' + m['citation_title'][0] + '}', author=' and '.join(authors),
                      year=m['citation_date'][0][:4], eprint=m['citation_arxiv_id'][0],
                      archivePrefix='arXiv', url=record['url'])
        entries.append('@misc{' + key + ',\n' + ',\n'.join('  '+k+' = {'+latex(v)+'}' for k, v in fields.items()) + '\n}\n')
    for key in ('forssell1999closedloop', 'sargent2013validation'):
        path = LITERATURE / (key + '.json')
        record = read(path)
        m = record['metadata'] if key.startswith('forssell') else record['data']['message']
        metadata_files.append(path)
        fields = dict(title='{' + m['title'][0] + '}',
                      author=' and '.join(a['family'] + ', ' + a['given'] for a in m['author']),
                      journal=m['container-title'][0], year=str(m['published']['date-parts'][0][0]),
                      volume=m['volume'], pages=m['page'].replace('-', '--'), doi=m['DOI'], url=m['URL'])
        if 'issue' in m:
            fields['number'] = m['issue']
        entries.append('@article{' + key + ',\n' + ',\n'.join('  '+k+' = {'+latex(v)+'}' for k, v in fields.items()) + '\n}\n')
    (PAPER / 'references.bib').write_text(bib + '\n' + '\n'.join(entries))
    assert len(re.findall(r'@\w+\{', (PAPER / 'references.bib').read_text())) == 25
    with (ABLATION / 'groups.csv').open() as f:
        groups = list(csv.DictReader(f))
    cells = ('original', 'service_only', 'startup_only', 'calibrated')
    labels = ('Original & 30', r'Calibrated$^*$ & 30', r'Original$^*$ & 69.22', 'Calibrated & 69.22')
    rows = []
    for cell, label in zip(cells, labels):
        values = [next(r for r in groups if r['window'] == 'locked_test_' + regime + '_v2'
                       and r['policy'] == 'hysteresis_u8_d0_c60' and r['cell'] == cell) for regime in REGIMES]
        rows.append(label + ' & ' + ' & '.join(f'{float(r["gpu_occupied_seconds_absolute_error"]):.1f}' for r in values) + r' \\')
    (PAPER / 'generated/ablation_rows.tex').write_text('\n'.join(rows) + '\n')
    supplement = []
    for policy, short in [('all1', 'All1'), ('all2', 'All2'), ('hysteresis_u8_d0_c60', 'Dynamic')]:
        for regime, window_short in zip(REGIMES, ('Steady', 'Recovery', 'Burst')):
            for cell, label in zip(cells, ('O/30', 'C/30*', 'O/69*', 'C/69')):
                r = next(r for r in groups if (r['policy'], r['window'], r['cell']) == (policy, 'locked_test_' + regime + '_v2', cell))
                keys = ('occupied_predicted', 'gpu_occupied_seconds_absolute_error',
                        'shared_local_service_mae_seconds', 'shared_local_coverage_all_arrivals', 'target_disagreement_rate')
                values = [float(r[k]) for k in keys]
                supplement.append(f'{short} & {window_short} & {label} & {values[0]:.1f} & {values[1]:.1f} & {values[2]:.3f} & {100*values[3]:.1f} & {100*values[4]:.1f} '+r'\\')
    (PAPER / 'generated/ablation_full_rows.tex').write_text('\n'.join(supplement)+'\n')
    window = 'locked_test_steady_v2'
    paths = dict(service_only=ABLATION / 'predictions/service_only/hysteresis_u8_d0_c60' / window / 'events.jsonl',
                 calibrated=DEFAULT / 'payload/n5_prediction_lock_20260919_r1/predictions/calibrated/hysteresis_u8_d0_c60' / window / 'events.jsonl')
    event_evidence = {}
    for cell, path in paths.items():
        events = lines(path)
        decisions = [e for e in events if e['kind'] == 'decision' and e['time'] in (195, 225, 255, 265)]
        states = [e for e in events if e['kind'] == 'state' and 195 <= e['time'] <= 325]
        event_evidence[cell] = dict(events_sha256=digest(path), decisions=decisions, states=states)
    save(DOC / 'R2_STARTUP_COOLDOWN_TRACE.json', dict(scope='post-hoc illustrative simulated trace; not physical intervention',
                                                    window=window, evidence=event_evidence))
    target = DOC / 'ablation_r2'
    target.mkdir(exist_ok=True)
    for name in ('comparisons.csv', 'groups.csv', 'effects.csv', 'common_support.csv', 'design.json',
                 'result.json', 'event_audits.json', 'manifest.json'):
        shutil.copyfile(ABLATION / name, target / name)
    save(DOC / 'revision_materials_r2.json', dict(
         bibliography_sha256=digest(PAPER / 'references.bib'), citation_count=25,
         source_records={str(p.relative_to(ROOT)): digest(p) for p in metadata_files},
         ablation_manifest_sha256=digest(ABLATION / 'manifest.json'),
         generated={str(p.relative_to(PAPER)): digest(p) for p in
                    [PAPER / 'generated/ablation_rows.tex', PAPER / 'generated/ablation_full_rows.tex']},
         scope='R1/R2 review revision; original registered results unchanged'))
    print('Generated 25 references and 4/36-row post-hoc ablation tables')


if __name__ == '__main__':
    build()
