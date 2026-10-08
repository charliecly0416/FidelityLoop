"""Source-to-EOF, deterministic public-arrival / constructed-text packages."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import importlib.metadata
from collections import Counter, defaultdict
from pathlib import Path

from .acquire import ROOT, STORE, EXPECTED, digest

WINDOW = 1800
REGIMES = ('steady_v2', 'recovery_v2', 'burst_offline_v2')
FIELDS = {'request_id', 'source_sha256', 'source_row_ordinal', 'source_occurrence',
          'raw_row_hash', 'raw_timestamp_s', 'arrival_s', 'sim_tick', 'wall_monotonic_s',
          'job_type', 'prompt_token_ids', 'input_tokens', 'max_output_tokens', 'deadline_s',
          'split', 'window_id', 'evidence', 'prompt_source_sha256', 'prompt_offset'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def hash_object(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def raw_rows(path):
    """Require one physical CSV line per record; preserve raw bytes and ordinal."""
    with Path(path).open('rb') as stream:
        header = next(csv.reader([stream.readline().decode('utf-8-sig')]))
        expected = ['Timestamp', 'Model', 'Request tokens', 'Response tokens', 'Total tokens', 'Log Type']
        if header != expected:
            raise ValueError('unexpected source header')
        previous = -1
        at_time = Counter()
        for ordinal, line in enumerate(stream):
            fields = next(csv.reader([line.decode('utf-8')]))
            if len(fields) != len(header):
                raise ValueError(f'invalid row at {ordinal}')
            row = dict(zip(header, fields))
            timestamp = int(row['Timestamp'])
            if timestamp < previous:
                raise ValueError(f'unsorted source at {ordinal}; no silent reorder')
            if timestamp != previous:
                at_time.clear()
            raw_hash = hashlib.sha256(line).hexdigest()
            occurrence = at_time[raw_hash]
            at_time[raw_hash] += 1
            previous = timestamp
            for field in ('Request tokens', 'Response tokens', 'Total tokens'):
                if int(row[field]) < 0:
                    raise ValueError(f'negative token count: {ordinal}')
            yield ordinal, timestamp, row, raw_hash, occurrence


def scan(path):
    bins = defaultdict(lambda: [0] * 30)
    first = last = None
    count = duplicates = 0
    for ordinal, timestamp, row, row_hash, occurrence in raw_rows(path):
        if first is None:
            first = {'ordinal': ordinal, 'timestamp': timestamp, 'row_hash': row_hash}
        last = {'ordinal': ordinal, 'timestamp': timestamp, 'row_hash': row_hash}
        bins[timestamp // WINDOW][timestamp % WINDOW // 60] += 1
        count += 1
        duplicates += occurrence > 0
    if first is None:
        raise ValueError('empty public source')
    return {'rows': count, 'first': first, 'last': last, 'duplicate_occurrences': duplicates,
            'source_sha256': digest(path), 'complete_eof': True}, bins


def select_windows(audit, bins):
    """Arrival-only, time split, nonoverlapping windows; no policy results read."""
    lo = math.ceil(audit['first']['timestamp'] / WINDOW)
    hi = audit['last']['timestamp'] // WINDOW
    width = hi - lo
    cuts = [lo, lo + int(width * .6), lo + int(width * .8), hi]
    windows = []
    for i, split in enumerate(('train', 'validation', 'locked_test')):
        candidates = [b for b in sorted(bins) if cuts[i] <= b < cuts[i + 1]]
        if len(candidates) < 3:
            raise ValueError('insufficient split coverage')
        density = sorted(sum(bins[b]) for b in candidates)
        median = density[len(density) // 2]
        steady = min(candidates, key=lambda b: (abs(sum(bins[b]) - median), b))
        candidates.remove(steady)
        recovery = max(candidates, key=lambda b: (sum(bins[b][:15]) - sum(bins[b][15:]), -b))
        candidates.remove(recovery)
        burst = max(candidates, key=lambda b: (max(bins[b]) - sum(bins[b]) / 30, -b))
        for regime, b in zip(REGIMES, (steady, recovery, burst)):
            windows.append({'id': f'{split}_{regime}', 'split': split, 'regime': regime,
                            'start': b * WINDOW, 'end': (b + 1) * WINDOW,
                            'online_rows': sum(bins[b]), 'minute_counts': bins[b]})
    return windows, {s: [cuts[i] * WINDOW, cuts[i + 1] * WINDOW]
                     for i, s in enumerate(('train', 'validation', 'locked_test'))}


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b'\n')


def build(source_dir, output):
    from tokenizers import Tokenizer
    source_dir, output = Path(source_dir), Path(output)
    if output.exists():
        raise FileExistsError('immutable workload package already exists')
    source_manifest = json.loads((source_dir / 'manifest.json').read_text())
    for name, record in source_manifest['records'].items():
        if digest(source_dir / name) != record['sha256']:
            raise ValueError(f'source changed: {name}')
    source = source_dir / 'BurstGPT_without_fails_1.csv'
    audit, bins = scan(source)
    windows, splits = select_windows(audit, bins)
    tokenizer = Tokenizer.from_file(str(source_dir / 'tokenizer.json'))
    corpora = {s: tokenizer.encode((source_dir / f'prompt_{s}.rst').read_text()).ids for s in splits}
    rows_by_window = {w['id']: [] for w in windows}
    window_by_bin = {w['start'] // WINDOW: w for w in windows}
    for ordinal, timestamp, raw, row_hash, occurrence in raw_rows(source):
        window = window_by_bin.get(timestamp // WINDOW)
        if window is None:
            continue
        request_id = hash_object([audit['source_sha256'], ordinal, occurrence])
        tokens = corpora[window['split']]
        n = min(2048, max(128, int(raw['Request tokens'])))
        start = ordinal * 997 % len(tokens)
        prompt = (tokens + tokens)[start:start + n]
        if len(prompt) != n:
            raise ValueError('prompt corpus too short')
        rows_by_window[window['id']].append({
            'request_id': request_id, 'source_sha256': audit['source_sha256'],
            'source_row_ordinal': ordinal, 'source_occurrence': occurrence, 'raw_row_hash': row_hash,
            'raw_timestamp_s': timestamp, 'arrival_s': timestamp - window['start'],
            'sim_tick': timestamp - window['start'], 'wall_monotonic_s': None,
            'job_type': 'online', 'prompt_token_ids': prompt, 'input_tokens': len(prompt),
            'max_output_tokens': 128, 'deadline_s': timestamp - window['start'] + 60,
            'split': window['split'], 'window_id': window['id'],
            'prompt_source_sha256': digest(source_dir / f'prompt_{window["split"]}.rst'),
            'prompt_offset': start,
            'evidence': {'arrival': 'observed', 'prompt': 'derived_from_PSF_text',
                         'input_length': 'transformed_clamp_128_2048', 'output_budget': 'constructed_fixed_128'},
        })
    output.mkdir(parents=True)
    index = {}
    for window in windows:
        rows = rows_by_window[window['id']]
        if window['regime'] == 'burst_offline_v2':
            for i in range(30):
                rows.append({'request_id': hash_object(['offline-v2', audit['source_sha256'], window['id'], i]),
                             'source_sha256': None, 'source_row_ordinal': None, 'source_occurrence': None,
                             'raw_row_hash': None, 'raw_timestamp_s': None,
                             'arrival_s': i * 60, 'sim_tick': i * 60, 'wall_monotonic_s': None,
                             'job_type': 'offline', 'prompt_token_ids': corpora[window['split']][3000 + i:3256 + i],
                             'input_tokens': 256, 'max_output_tokens': 256, 'deadline_s': i * 60 + 300,
                             'split': window['split'], 'window_id': window['id'],
                             'prompt_source_sha256': digest(source_dir / f'prompt_{window["split"]}.rst'),
                             'prompt_offset': 3000 + i,
                             'evidence': {'arrival': 'constructed', 'prompt': 'derived_from_PSF_text',
                                          'input_length': 'constructed_256', 'output_budget': 'constructed_fixed_256'}})
        rows.sort(key=lambda r: (r['arrival_s'], r['request_id']))
        path = output / (window['id'] + '.jsonl')
        with path.open('wb') as stream:
            for row in rows:
                stream.write(canonical(row) + b'\n')
        index[path.name] = {'sha256': digest(path), 'rows': len(rows), 'window': window,
                           'request_ids_sha256': hash_object([r['request_id'] for r in rows]),
                           'row_hashes_sha256': hash_object([r['raw_row_hash'] for r in rows]),
                           'eof': {'first': rows[0]['request_id'], 'last': rows[-1]['request_id']}}
    manifest = {'schema': 'maxopt-workload-v2', 'identity': 'burstgpt_public_arrival_constructed_prompt_v2',
                'source_audit': audit, 'sources': source_manifest, 'splits': splits, 'files': index,
                'generator_sha256': digest(__file__), 'code_commit': subprocess.check_output(
                    ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'code_files': {p: digest(ROOT / p) for p in ['scripts/maxopt_v2/workload.py', 'scripts/maxopt_v2/acquire.py']},
                'tokenizers_version': importlib.metadata.version('tokenizers'),
                'tokenizer': {'model': 'Qwen/Qwen2.5-7B-Instruct',
                              'revision': source_manifest['tokenizer_revision'],
                              'sha256': digest(source_dir / 'tokenizer.json'), 'add_chat_template': False},
                'selection': '30-minute disjoint full bins; time 60/20/20; median-count steady; max first-half-minus-second recovery; max minute-minus-mean burst; ties earliest',
                'limits': 'source-success-only subset filtered by original authors on Response tokens; no Session ID so real-session isolation unverified; disjoint document groups and exact prompt dedup only, shared generic language may remain; selected arrival cases not random population sample; no production prompt/output reproduction; no policy test outcomes opened; fixed output budget synthetic scenario',
                'license': {'arrival': 'CC-BY-4.0; cite BurstGPT authors as in saved README',
                            'prompt': 'PSF license v3.13.0; retain python_LICENSE', 'tokenizer': 'Apache-2.0; retain qwen_LICENSE'}}
    write_json(output / 'manifest.json', manifest)
    return validate(output)


def validate(package):
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest.get('schema') != 'maxopt-workload-v2':
        raise ValueError('unsupported package schema')
    expected_names = {f'{s}_{r}.jsonl' for s in ('train', 'validation', 'locked_test') for r in REGIMES}
    if set(manifest['files']) != expected_names:
        raise ValueError('incomplete window matrix')
    bounds = [manifest['splits'][s] for s in ('train', 'validation', 'locked_test')]
    if any(lo >= hi for lo, hi in bounds) or any(bounds[i][1] != bounds[i+1][0] for i in range(2)):
        raise ValueError('noncontiguous/overlapping split')
    windows = sorted((v['window']['start'], v['window']['end']) for v in manifest['files'].values())
    if any(a[1] > b[0] for a, b in zip(windows, windows[1:])):
        raise ValueError('overlapping windows')
    ids, prompt_splits = set(), {}
    totals = {}
    for name, meta in manifest['files'].items():
        path = package / name
        if Path(name).name != name or path.is_symlink() or digest(path) != meta['sha256']:
            raise ValueError('unsafe path or workload hash mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != meta['rows'] or not rows:
            raise ValueError('row count / EOF mismatch')
        if [rows[0]['request_id'], rows[-1]['request_id']] != [meta['eof']['first'], meta['eof']['last']]:
            raise ValueError('EOF identity mismatch')
        window = meta['window']
        low, high = manifest['splits'][window['split']]
        if not low <= window['start'] < window['end'] <= high:
            raise ValueError('window crosses split')
        last_time = -1
        for row in rows:
            if set(row) != FIELDS:
                raise ValueError('unknown/missing fields, including future outcomes, rejected')
            if row['request_id'] in ids:
                raise ValueError('duplicate request ID')
            ids.add(row['request_id'])
            if row['split'] != window['split'] or row['window_id'] != window['id']:
                raise ValueError('split membership mismatch')
            t = row['arrival_s']
            if not last_time <= t < window['end'] - window['start']:
                raise ValueError('arrival order / half-open boundary')
            last_time = t
            if row['input_tokens'] != len(row['prompt_token_ids']) or row['deadline_s'] < t:
                raise ValueError('token / deadline mismatch')
            if row['job_type'] not in {'online', 'offline'} or not row['prompt_token_ids'] or any(type(x) is not int or x < 0 for x in row['prompt_token_ids']):
                raise ValueError('job type / token ID mismatch')
            if row['max_output_tokens'] != (128 if row['job_type'] == 'online' else 256):
                raise ValueError('output budget mismatch')
            if row['sim_tick'] != t or row['wall_monotonic_s'] is not None:
                raise ValueError('clock-domain mismatch')
            if row['job_type'] == 'online':
                expected_id = hash_object([row['source_sha256'], row['source_row_ordinal'], row['source_occurrence']])
                if expected_id != row['request_id'] or row['raw_timestamp_s'] != window['start'] + t:
                    raise ValueError('source identity or raw-to-normalized mismatch')
            prompt_hash = hash_object(row['prompt_token_ids'])
            if prompt_hash in prompt_splits and prompt_splits[prompt_hash] != row['split']:
                raise ValueError('exact prompt duplicate across split')
            prompt_splits[prompt_hash] = row['split']
        if hash_object([r['request_id'] for r in rows]) != meta['request_ids_sha256']:
            raise ValueError('split ID digest mismatch')
        if hash_object([r['raw_row_hash'] for r in rows]) != meta['row_hashes_sha256']:
            raise ValueError('row mapping digest mismatch')
        totals[name] = len(rows)
    return {'valid': True, 'rows': len(ids), 'windows': totals, 'manifest_sha256': digest(package / 'manifest.json')}


def validate_sources(package, source_dir):
    """Independently re-read raw file to physical EOF and reconstruct every mapping."""
    from tokenizers import Tokenizer
    package, source_dir = Path(package), Path(source_dir)
    result = validate(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    if set(manifest['sources']['records']) != set(EXPECTED):
        raise ValueError('source/license inventory incomplete')
    expected_code = {p: digest(ROOT / p) for p in ['scripts/maxopt_v2/workload.py', 'scripts/maxopt_v2/acquire.py']}
    if manifest['code_files'] != expected_code or manifest['generator_sha256'] != expected_code['scripts/maxopt_v2/workload.py'] or manifest['tokenizers_version'] != importlib.metadata.version('tokenizers'):
        raise ValueError('generator/dependency identity mismatch; use matching source snapshot')
    for name, record in manifest['sources']['records'].items():
        if digest(source_dir / name) != record['sha256'] or record['sha256'] != EXPECTED[name]:
            raise ValueError('source artifact changed')
    source_sha = digest(source_dir / 'BurstGPT_without_fails_1.csv')
    if manifest['source_audit']['source_sha256'] != source_sha or manifest['source_audit']['complete_eof'] is not True:
        raise ValueError('source identity / EOF flag mismatch')
    tokenizer = Tokenizer.from_file(str(source_dir / 'tokenizer.json'))
    corpora = {s: tokenizer.encode((source_dir / f'prompt_{s}.rst').read_text()).ids for s in manifest['splits']}
    wanted = {}
    for name in manifest['files']:
        offline_arrivals = []
        for line in (package / name).read_text().splitlines():
            row = json.loads(line)
            document = source_dir / f'prompt_{row["split"]}.rst'
            tokens = corpora[row['split']]
            start, length = row['prompt_offset'], row['input_tokens']
            if digest(document) != row['prompt_source_sha256'] or row['prompt_token_ids'] != (tokens + tokens)[start:start + length]:
                raise ValueError('prompt source/offset mapping mismatch')
            if row['job_type'] == 'online':
                if row['source_sha256'] != source_sha:
                    raise ValueError('row source identity mismatch')
                wanted[row['source_row_ordinal']] = row
            else:
                offline_arrivals.append(row['arrival_s'])
                i = row['arrival_s'] // 60
                if row['arrival_s'] != i * 60 or not 0 <= i < 30 or row['deadline_s'] != i * 60 + 300 or row['input_tokens'] != 256 or row['prompt_offset'] != 3000 + i or row['request_id'] != hash_object(['offline-v2', source_sha, row['window_id'], i]):
                    raise ValueError('offline constructed mapping mismatch')
        expected_offline = list(range(0, 1800, 60)) if manifest['files'][name]['window']['regime'] == 'burst_offline_v2' else []
        if offline_arrivals != expected_offline:
            raise ValueError('missing/extra constructed offline arrivals')
    expected_bins = {v['window']['start'] // WINDOW for v in manifest['files'].values()}
    bins = defaultdict(lambda: [0] * 30)
    count, duplicates, first, last = 0, 0, None, None
    for ordinal, timestamp, raw, row_hash, occurrence in raw_rows(source_dir / 'BurstGPT_without_fails_1.csv'):
        marker = {'ordinal': ordinal, 'timestamp': timestamp, 'row_hash': row_hash}
        first = marker if first is None else first
        last = marker
        count += 1
        duplicates += occurrence > 0
        bins[timestamp // WINDOW][timestamp % WINDOW // 60] += 1
        if timestamp // WINDOW in expected_bins and ordinal not in wanted:
            raise ValueError('missing selected raw source row')
        if ordinal in wanted:
            row = wanted.pop(ordinal)
            if row_hash != row['raw_row_hash'] or occurrence != row['source_occurrence'] or timestamp != row['raw_timestamp_s']:
                raise ValueError('raw row mapping mismatch')
            if row['input_tokens'] != min(2048, max(128, int(raw['Request tokens']))) or row['max_output_tokens'] != 128:
                raise ValueError('token demand mapping mismatch')
    audit = manifest['source_audit']
    if wanted or count != audit['rows'] or first != audit['first'] or last != audit['last'] or duplicates != audit['duplicate_occurrences']:
        raise ValueError('source-to-EOF completeness mismatch')
    selected, split_bounds = select_windows(audit, bins)
    if split_bounds != manifest['splits'] or {w['id']: w for w in selected} != {v['window']['id']: v['window'] for v in manifest['files'].values()}:
        raise ValueError('window selection / source counts mismatch')
    return {**result, 'source_rows_rechecked': count, 'full_source_eof': True}


def load_window(package, window_id, *, allowed_split):
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    meta = manifest['files'][window_id + '.jsonl']
    if meta['window']['split'] != allowed_split:
        raise ValueError('split access denied')
    validate(package)
    path = package / (window_id + '.jsonl')
    if digest(path) != meta['sha256']:
        raise ValueError('window changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if any(set(row) != FIELDS for row in rows):
        raise ValueError('unknown/missing fields rejected')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['build', 'validate', 'validate-sources'])
    parser.add_argument('--source', type=Path, default=STORE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'build':
        result = build(args.source, args.output)
    elif args.operation == 'validate-sources':
        result = validate_sources(args.output, args.source)
    else:
        result = validate(args.output)
    print(json.dumps(result, indent=2))
