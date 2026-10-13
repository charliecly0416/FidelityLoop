"""Inspect known development inputs. This module never generates final inputs."""
import hashlib
import json
from pathlib import Path

from fidelityloop.bridge.protocol import ROOT, INPUT, DOC, read, sha256
from fidelityloop.legacy.maxopt_v2.calibrated import validate_rows


def inspect_partitions(partitions):
    seen_ids, seen_source, windows, prompt_sets = {}, {}, {}, {}
    for name, rows in sorted(partitions.items()):
        validate_rows(rows, 1800, False)
        split = 'train' if name.startswith('train_') else 'validation'
        if not name.startswith(('train_', 'validation_')) or any(r['split'] != split or r['window_id'] != name for r in rows):
            raise ValueError('incorrect window or partition identity')
        online = [r for r in rows if r['job_type'] == 'online']
        offline = [r for r in rows if r['job_type'] == 'offline']
        if not online or len(offline) != 120 or sorted(r['arrival_s'] for r in offline) != list(range(0, 1800, 15)):
            raise ValueError('registered population or offline arrival pattern changed')
        starts, sources = set(), set()
        prompt_sets[name] = set()
        for row in rows:
            rid = row['request_id']
            if rid in seen_ids:
                raise ValueError('request ID crosses windows: ' + rid)
            seen_ids[rid] = name
            kind = row['job_type']
            if row['deadline_s'] != row['arrival_s'] + (60 if kind == 'online' else 300):
                raise ValueError('deadline changed')
            if row['max_output_tokens'] != (128 if kind == 'online' else 256):
                raise ValueError('output token budget changed')
            if kind == 'online':
                if not 128 <= row['input_tokens'] <= 2048:
                    raise ValueError('online token clamp changed')
                if not row['source_sha256'] or type(row['source_row_ordinal']) is not int or row['source_row_ordinal'] < 1:
                    raise ValueError('missing source-row identity')
                key = (row['source_sha256'], row['source_row_ordinal'])
                if key in seen_source:
                    raise ValueError('source row crosses windows')
                seen_source[key] = name
                starts.add(row['raw_timestamp_s'] - row['arrival_s'])
                sources.add(row['source_sha256'])
            elif row['input_tokens'] != 256:
                raise ValueError('offline input budget changed')
            prompt_sets[name].add(hashlib.sha256(json.dumps(row['prompt_token_ids']).encode()).hexdigest())
        if len(starts) != 1 or len(sources) != 1:
            raise ValueError('mixed source window')
        start = starts.pop()
        windows[name] = dict(split=split, start=start, end=start + 1800, online=len(online),
                             offline=len(offline), source_sha256=sources.pop())
    ordered = sorted(windows.values(), key=lambda w: w['start'])
    if any(b['start'] - a['end'] < 1800 for a, b in zip(ordered, ordered[1:])):
        raise ValueError('source windows violate 1800-second guard gap')
    overlap = {a + '|' + b: len(prompt_sets[a] & prompt_sets[b])
               for a in prompt_sets for b in prompt_sets if a < b}
    return dict(windows=windows, requests=len(seen_ids), source_rows=len(seen_source),
                request_id_overlap=0, source_row_overlap=0, exact_prompt_overlap=overlap,
                independence='source-time and source-row separation only; no user/session independence claim; prompt reuse disclosed')


def main():
    files = sorted((ROOT / INPUT / 'development').glob('*.jsonl'))
    if len(files) != 6:
        raise ValueError('exactly six registered development windows required')
    result = inspect_partitions({p.stem: [json.loads(s) for s in p.read_text().splitlines()] for p in files})
    result.update(schema='maxopt-bridge-data-split-v1', status='SELF_CHECKED',
                  files={str(p.relative_to(ROOT)): sha256(p) for p in files},
                  final_inputs_generated=False, final_test_status='BLOCKED_UNTIL_ENVIRONMENT_AND_POLICY_FREEZE')
    (ROOT / DOC / 'DATA_SPLIT_MANIFEST.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'requests', 'source_rows', 'final_inputs_generated')}))


if __name__ == '__main__':
    main()
