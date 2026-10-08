"""Registered final-input preparation. Real source access requires a reviewed gate.

Importing this module does not read data, choose windows, or load a tokenizer.
Pure selection/construction helpers support synthetic engineering tests only;
the sole production entry is generate() / the explicit CLI below.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from scripts.maxopt_v2 import workload as v2
from scripts.maxopt_v3.workload import classify
from scripts.maxopt_v2.calibrated import validate_rows
from .protocol import ROOT, DOC, INPUT, F4_IDS

RULE_SHA256 = '300c4daa15128d9de76b031b45678056612f6eb7c941ae6f5a4d890157043ac5'
PARENT_SHA256 = '4973bfe1738e4b5aac0eec0386571c7d97263e0e592f7ba4879bd2e25e60d4a9'
SOURCE_ZIP_SHA256 = '48356a1cb36a3d15b5084aecf90cc5e714f046c24dd1bd7e1ccdc44988437651'
DELIVERY_MANIFEST_SHA256 = '791bf8591c9c777b4ec64af37263360848b9f906ca61b7ca7c96b69a92dfc53e'
REF_NAMES = {'authorization', 'r0_acceptance', 'environment_acceptance',
             'route_freeze', 'policy_freeze', 'known_input_closure'}
GATE_FIELDS = {'schema', 'scope', 'decision', 'source_directory', 'output_directory', 'attempt_id',
               'preparer', 'rule_sha256', 'parent_manifest_sha256', 'generator_sha256',
               'source_version_sha256', 'provenance_sha256', 'gpu_uuids', 'bindings', 'independent_review'}


class NoWindowError(ValueError):
    """Registered role cannot be filled; candidates must not be resampled."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('a bound file requires exactly path and sha256')
    path = Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('bound file must be an absolute regular file')
    if digest(path) != ref['sha256']:
        raise ValueError('bound file hash mismatch: ' + str(path))
    return path


def registration(root):
    root = Path(root).resolve()
    delivery = root / 'SOURCE_DELIVERY_MANIFEST.json'
    if digest(delivery) != DELIVERY_MANIFEST_SHA256:
        raise ValueError('accepted original-source inventory changed')
    original = read(delivery)['files']
    if len(original) != 335:
        raise ValueError('incomplete original-source closure')
    for item in original:
        path = root / item['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or
                path.stat().st_size != item['bytes'] or digest(path) != item['sha256']):
            raise ValueError('original source changed: ' + item['path'])
    rule_path = root / DOC / 'TEST_GENERATION_RULE.json'
    parent_path = root / INPUT / 'known_windows/v2_manifest.json'
    if digest(rule_path) != RULE_SHA256 or digest(parent_path) != PARENT_SHA256:
        raise ValueError('registered rule or v2 parent changed')
    rule, parent = read(rule_path), read(parent_path)
    for rel, sha in [('scripts/maxopt_v2/workload.py', rule['candidate_enumeration']['scanner_source_sha256']),
                     ('scripts/maxopt_v3/workload.py', rule['shape_classifier_sha256'])]:
        if digest(root / rel) != sha:
            raise ValueError('registered scanner/classifier source changed')
    # Also check the modules actually imported, not only an alternate --root.
    if (digest(v2.__file__) != rule['candidate_enumeration']['scanner_source_sha256'] or
            digest(Path(__import__('sys').modules[classify.__module__].__file__)) != rule['shape_classifier_sha256']):
        raise ValueError('loaded scanner/classifier differs from registration')
    if (len(rule['known_windows']) != 12 or len({w['id'] for w in rule['known_windows']}) != 12 or
            rule['roles_in_order'] != ['recovery', 'steady'] or rule['window_seconds'] != 1800 or
            rule['guard_gap_seconds'] != 1800 or rule['status'] != 'REGISTERED_RULE_ONLY_NO_FINAL_INPUTS'):
        raise ValueError('unsupported registered generation semantics')
    for rel, item in rule['known_window_sources'].items():
        if digest(root / rel) != item['sha256']:
            raise ValueError('known-window registration changed')
    return rule, parent


def verify_gate(gate_path, expected_sha256, *, root, source_dir, output):
    """Verify all permits before opening the raw trace or creating output.

The externally supplied gate digest is the coordinator trust anchor. JSON
receipts are auditable file bindings, not cryptographic identity signatures.
"""
    gate_file = bound({'path': str(Path(gate_path).absolute()), 'sha256': expected_sha256})
    gate = read(gate_file)
    root, source_dir, output = map(lambda p: Path(p).resolve(), (root, source_dir, output))
    if (set(gate) != GATE_FIELDS or gate.get('schema') != 'maxopt-bridge-final-generation-gate-v1' or
            gate.get('scope') != 'FINAL_INPUT_GENERATION_ONLY' or
            gate.get('decision') != 'AUTHORIZED_AFTER_INDEPENDENT_REVIEW' or
            gate.get('source_directory') != str(source_dir) or gate.get('output_directory') != str(output) or
            not gate.get('attempt_id') or not gate.get('preparer')):
        raise ValueError('missing or incompatible final-generation permit')
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(source_dir) or source_dir.is_relative_to(output):
        raise ValueError('output must be new and outside source/workspace')
    rule, parent = registration(root)
    expected_code = digest(Path(__file__))
    bindings = gate.get('bindings', {})
    if (set(bindings) != REF_NAMES or gate.get('rule_sha256') != RULE_SHA256 or
            gate.get('parent_manifest_sha256') != PARENT_SHA256 or
            gate.get('generator_sha256') != expected_code or
            gate.get('source_version_sha256') != digest(root / 'SOURCE_VERSION.json') or
            gate.get('provenance_sha256') != digest(root / 'COMMIT_PROVENANCE.json')):
        raise ValueError('generation code/rule/provenance binding incomplete')
    docs = {name: read(bound(ref)) for name, ref in bindings.items()}
    auth, r0, env = (docs[k] for k in ('authorization', 'r0_acceptance', 'environment_acceptance'))
    if (auth.get('schema') != 'maxopt-bridge-current-authorization-v1' or
            r0.get('schema') != 'maxopt-bridge-r0-coordinator-acceptance-v1' or
            r0.get('decision') != 'ACCEPTED' or r0.get('source_zip_sha256') != SOURCE_ZIP_SHA256 or
            r0.get('authorization_sha256') != bindings['authorization']['sha256'] or
            env.get('environment_gate') != 'ACCEPTED_PASS_DEVELOPMENT_ENVIRONMENT_ONLY' or
            auth.get('development_environment_acceptance') != bindings['environment_acceptance'] or
            gate.get('gpu_uuids') != auth.get('accepted_current_gpu_uuids') or
            len(set(gate.get('gpu_uuids', []))) != 2 or
            auth.get('source_head') != read(root / 'SOURCE_VERSION.json').get('head')):
        raise ValueError('current source/environment acceptance missing')
    route, policy = docs['route_freeze'], docs['policy_freeze']
    for phase, doc in [('S06', route), ('S07', policy)]:
        if (doc.get('schema') != 'maxopt-bridge-pre-generation-stage-freeze-v1' or
                doc.get('phase') != phase or doc.get('decision') != 'ACCEPTED' or
                doc.get('authorization_sha256') != bindings['authorization']['sha256'] or
                not doc.get('frozen_artifacts')):
            raise ValueError('accepted ' + phase + ' freeze is missing')
        for ref in doc['frozen_artifacts']:
            bound(ref)
    if route.get('route') not in ('F4', 'F4_plus_two_raw_PPO') or policy.get('route') != route['route']:
        raise ValueError('route and policy freeze differ')
    if (policy.get('route_freeze_sha256') != bindings['route_freeze']['sha256'] or
            policy.get('f4_policy_ids') != list(F4_IDS)):
        raise ValueError('final policies not bound to the fixed F4 route')
    planned = route.get('planned_policy_ids')
    effective = policy.get('effective_policy_ids')
    if (not isinstance(planned, list) or len(set(planned)) != len(planned) or
            planned[:4] != list(F4_IDS) or policy.get('planned_policy_ids') != planned or
            not isinstance(effective, list) or len(set(effective)) != len(effective)):
        raise ValueError('planned/effective policy identity ledger missing')
    checkpoints = policy.get('checkpoints')
    if route['route'] == 'F4':
        if (planned != list(F4_IDS) or effective != list(F4_IDS) or policy.get('effective_route') != 'F4' or
                policy.get('disposition') != 'LAWFUL_SKIP_F4_ONLY' or checkpoints != [] or not policy.get('skip_reason')):
            raise ValueError('F4 route requires explicit lawful S07 skip')
    elif policy.get('disposition') == 'SYMMETRIC_PPO_EXIT':
        if (len(planned) != 6 or effective != list(F4_IDS) or policy.get('effective_route') != 'F4' or
                checkpoints != [] or policy.get('not_run_policy_ids') != planned[4:] or not policy.get('skip_reason')):
            raise ValueError('PPO exit must preserve and exclude both planned arms')
    else:
        if (policy.get('disposition') != 'FROZEN_RAW_PPO' or not isinstance(checkpoints, list) or
                len(checkpoints) != 2 or {c.get('training_model') for c in checkpoints} != {'C', 'C30'} or
                len(planned) != 6 or effective != planned or policy.get('effective_route') != route['route'] or
                {c.get('policy_id') for c in checkpoints} != set(planned[4:])):
            raise ValueError('both raw PPO checkpoints must be frozen')
        for ckpt in checkpoints:
            bound(ckpt['file'])
    review = read(bound(gate['independent_review']))
    reviewed = {k: v for k, v in gate.items() if k != 'independent_review'}
    if (review.get('schema') != 'maxopt-bridge-final-generation-review-v1' or
            review.get('decision') != 'PASS' or review.get('scope') != 'FINAL_INPUT_GENERATION_ONLY' or
            review.get('reviewed_gate_payload_sha256') != v2.hash_object(reviewed) or
            not review.get('reviewer') or review['reviewer'] == gate['preparer']):
        raise ValueError('independent generation review does not bind this permit')
    return gate, rule, parent, docs['known_input_closure']


def select_windows(audit, bins, rule):
    """Pure deterministic arrival-only selection; never change roles on failure."""
    window, gap = rule['window_seconds'], rule['guard_gap_seconds']
    if (audit.get('complete_eof') is not True or audit['rows'] != rule['source_rows'] or
            audit['source_sha256'] != rule['source_sha256'] or
            audit['first']['ordinal'] != 0 or audit['last']['ordinal'] != audit['rows'] - 1):
        raise ValueError('source SHA/rows/complete EOF mismatch')
    lower = math.ceil(audit['first']['timestamp'] / window)
    upper = audit['last']['timestamp'] // window
    candidates = []
    for b in sorted(bins):
        counts = bins[b]
        if type(b) is not int or len(counts) != 30 or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError('malformed arrival bin')
        start, end = b * window, (b + 1) * window
        if not lower <= b < upper or start < rule['earliest_start'] or not sum(counts):
            continue
        if any(not (end + gap <= old['start'] or start >= old['end'] + gap)
               for old in rule['known_windows']):
            continue
        candidates.append(dict(start=start, end=end, counts=counts, classes=classify(counts),
                               hash=hashlib.sha256((rule['window_salt'] + str(start)).encode()).hexdigest()))
    chosen = []
    for role, name in zip(rule['roles_in_order'], rule['new_window_ids']):
        ordered = sorted((c for c in candidates if c['classes'][role]), key=lambda c: (c['hash'], c['start']))
        match = next((c for c in ordered if all(abs(c['start'] - w['start']) >= window + gap for w in chosen)), None)
        if match is None:
            raise NoWindowError('HOLD_NO_WINDOW: ' + role + '; no resampling')
        chosen.append(dict(match, role=role, id=name))
    return chosen, candidates


def keep(window_id, request_id, rule):
    key = rule['thinning_salt'] + '\n' + window_id + '\n' + request_id
    value = hashlib.sha256(key.encode()).hexdigest()
    return int(value, 16) < int(rule['thinning_threshold']), value


def known_inputs(closure, rule, parent, root):
    """Read only gate-bound historical inputs; require all 12 registered windows."""
    if closure.get('schema') != 'maxopt-bridge-known-input-closure-v1' or not closure.get('files'):
        raise ValueError('complete known-input closure required')
    windows = {w['id']: w for w in rule['known_windows']}
    covered, parent_covered, actual_paths, ids, source_rows, prompts = set(), set(), set(), set(), {}, set()
    for item in closure['files']:
        path = bound(item['file']); actual_paths.add(path.resolve())
        if item['window_id'] not in windows:
            raise ValueError('unregistered historical input window')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if not rows or len(rows) != item['rows']:
            raise ValueError('historical rows/EOF mismatch')
        name = item['window_id']; w = windows[name]; covered.add(name)
        # Original v2 inputs can be independently bound to the copied parent.
        if item.get('kind') == 'v2_parent':
            record = parent['files'].get(name + '.jsonl')
            if not record or record['sha256'] != item['file']['sha256'] or record['rows'] != len(rows):
                raise ValueError('v2 history differs from copied parent')
            parent_covered.add(name)
        for row in rows:
            if row['window_id'] != name or not 0 <= row['arrival_s'] < 1800:
                raise ValueError('historical window membership mismatch')
            ids.add(row['request_id']); prompts.add(v2.hash_object(row['prompt_token_ids']))
            if row['job_type'] == 'online':
                key = (row['source_sha256'], row['source_row_ordinal'])
                expected_id = v2.hash_object([key[0], key[1], row['source_occurrence']])
                if (key[0] != rule['source_sha256'] or type(key[1]) is not int or key[1] < 0 or
                        row['raw_timestamp_s'] != w['start'] + row['arrival_s'] or row['request_id'] != expected_id):
                    raise ValueError('historical source/request identity mismatch')
                marker = (row['raw_timestamp_s'], row['raw_row_hash'], row['source_occurrence'])
                if key[1] in source_rows and source_rows[key[1]] != marker:
                    raise ValueError('inconsistent historical source row')
                source_rows[key[1]] = marker
    if covered != set(windows):
        raise ValueError('all twelve known windows must be covered')
    if parent_covered != {record['window']['id'] for record in parent['files'].values()}:
        raise ValueError('all original v2 parent inputs must be covered, beyond development copies')
    development = set((Path(root) / INPUT / 'development').glob('*.jsonl'))
    if len(development) != 6 or not {p.resolve() for p in development} <= actual_paths:
        raise ValueError('all six copied development inputs must be bound')
    for p in development:
        original = read(Path(root) / DOC / 'SOURCE_MANIFEST.json')['files'][str(p.relative_to(root))]['sha256']
        if digest(p) != original:
            raise ValueError('copied development input changed')
    return ids, source_rows, prompts


def construct(raw, windows, audit, rule, tokens, prompt_sha, history):
    """Consume a complete source iterator; no source path or tokenizer loading."""
    if len(tokens) < 3375 or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('prompt corpus is too short or has invalid token IDs')
    known_ids, known_rows, old_prompts = history
    pending = dict(known_rows)
    by_bin = {w['start'] // 1800: w for w in windows}
    rows = {w['id']: [] for w in windows}; mappings = {w['id']: [] for w in windows}
    count = duplicates = 0; first = last = None
    for ordinal, timestamp, source, raw_hash, occurrence in raw:
        marker = dict(ordinal=ordinal, timestamp=timestamp, row_hash=raw_hash)
        first = marker if first is None else first; last = marker
        count += 1; duplicates += occurrence > 0
        if ordinal in pending:
            if pending.pop(ordinal) != (timestamp, raw_hash, occurrence):
                raise ValueError('historical source mapping failed')
        w = by_bin.get(timestamp // 1800)
        if w is None:
            continue
        rid = v2.hash_object([audit['source_sha256'], ordinal, occurrence])
        if rid in known_ids or ordinal in known_rows:
            raise ValueError('final source row/request crosses a known partition')
        retained, thin_hash = keep(w['id'], rid, rule)
        mappings[w['id']].append(dict(ordinal=ordinal, occurrence=occurrence, raw_row_hash=raw_hash,
                                     raw_timestamp_s=timestamp, request_id=rid, thinning_sha256=thin_hash, kept=retained))
        if not retained:
            continue
        n = min(2048, max(128, int(source['Request tokens'])))
        offset = ordinal * 997 % len(tokens); prompt = (tokens + tokens)[offset:offset+n]
        if len(prompt) != n:
            raise ValueError('online prompt construction is incomplete')
        arrival = timestamp - w['start']
        rows[w['id']].append(dict(request_id=rid, source_sha256=audit['source_sha256'], source_row_ordinal=ordinal,
            source_occurrence=occurrence, raw_row_hash=raw_hash, raw_timestamp_s=timestamp, arrival_s=arrival,
            sim_tick=arrival, wall_monotonic_s=None, job_type='online', prompt_token_ids=prompt, input_tokens=n,
            max_output_tokens=128, deadline_s=arrival+60, split='locked_test', window_id=w['id'],
            prompt_source_sha256=prompt_sha, prompt_offset=offset, evidence=dict(arrival='observed',
            prompt='derived_from_PSF_text', input_length='transformed_clamp_128_2048', output_budget='constructed_fixed_128')))
    if (pending or count != audit['rows'] or first != audit['first'] or last != audit['last'] or
            duplicates != audit['duplicate_occurrences']):
        raise ValueError('second source-to-EOF pass differs from first scan/history')
    all_ids = set(); overlap = {}
    for w in windows:
        name = w['id']
        if len(mappings[name]) != sum(w['counts']):
            raise ValueError('selected source count differs between passes')
        for k in range(120):
            rid = v2.hash_object(['offline-bridge-v1', audit['source_sha256'], name, k])
            if rid in known_ids:
                raise ValueError('constructed request crosses a known partition')
            rows[name].append(dict(request_id=rid, source_sha256=None, source_row_ordinal=None,
                source_occurrence=None, raw_row_hash=None, raw_timestamp_s=None, arrival_s=15*k, sim_tick=15*k,
                wall_monotonic_s=None, job_type='offline', prompt_token_ids=tokens[3000+k:3256+k], input_tokens=256,
                max_output_tokens=256, deadline_s=15*k+300, split='locked_test', window_id=name,
                prompt_source_sha256=prompt_sha, prompt_offset=3000+k, evidence=dict(arrival='constructed',
                prompt='derived_from_PSF_text', input_length='constructed_256', output_budget='constructed_fixed_256')))
        rows[name].sort(key=lambda r: (r['arrival_s'], r['request_id']))
        validate_rows(rows[name], 1800, True)
        for row in rows[name]:
            if row['request_id'] in all_ids or len(row['prompt_token_ids']) != row['input_tokens']:
                raise ValueError('duplicate final ID or incomplete prompt')
            all_ids.add(row['request_id'])
        overlap[name] = sum(v2.hash_object(r['prompt_token_ids']) in old_prompts for r in rows[name])
    return rows, mappings, overlap


def save_new(path, obj):
    with Path(path).open('xb') as stream:
        stream.write(v2.canonical(obj) + b'\n')


def generate(*, gate_path, gate_sha256, source_dir, output, root=ROOT):
    """No valid production gate means no raw source reads and no output writes."""
    root, source_dir, output = map(lambda p: Path(p).resolve(), (root, source_dir, output))
    gate, rule, parent, closure = verify_gate(gate_path, gate_sha256, root=root, source_dir=source_dir, output=output)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / 'ATTEMPT.json', dict(schema='maxopt-bridge-final-input-attempt-v1',
             attempt_id=gate['attempt_id'], gate_sha256=gate_sha256, status='STARTED', final_experiment_lock=False))
    try:
        source_version = read(root / 'SOURCE_VERSION.json')
        registered_payload_sha256 = v2.hash_object([rule, parent])
        code_files = {p: digest(root/p) for p in
            ['scripts/maxopt_v2/workload.py', 'scripts/maxopt_v3/workload.py', 'scripts/maxopt_v2/acquire.py']}
        sources = parent['sources']['records']
        for name, meta in sources.items():
            p = source_dir / name
            if Path(name).name != name or p.is_symlink() or p.stat().st_size != meta['bytes'] or digest(p) != meta['sha256']:
                raise ValueError('pinned source bytes/hash mismatch: ' + name)
        source = source_dir / 'BurstGPT_without_fails_1.csv'
        if sources[source.name]['sha256'] != rule['source_sha256']:
            raise ValueError('raw source parent/rule mismatch')
        history = known_inputs(closure, rule, parent, root)
        audit, bins = v2.scan(source)
        save_new(output / 'SOURCE_AUDIT.json', audit)
        windows, candidates = select_windows(audit, bins, rule)
        from tokenizers import Tokenizer
        tokens = Tokenizer.from_file(str(source_dir / 'tokenizer.json')).encode(
            (source_dir / 'prompt_locked_test.rst').read_text(encoding='utf-8')).ids
        prompt_sha = sources['prompt_locked_test.rst']['sha256']
        rows, mappings, overlap = construct(v2.raw_rows(source), windows, audit, rule, tokens, prompt_sha, history)
        save_new(output / 'GENERATION_GATE.json', gate)
        save_new(output / 'WINDOW_SELECTION.json', dict(audit=audit, windows=windows, candidates=candidates,
                 old_windows=rule['known_windows'], performance_used=False, resampling=False))
        (output / 'inputs').mkdir(); (output / 'mappings').mkdir()
        statistics = {}
        for w in windows:
            name = w['id']
            with (output / 'inputs' / (name + '.jsonl')).open('xb') as stream:
                for row in rows[name]:
                    stream.write(v2.canonical(row) + b'\n')
            save_new(output / 'mappings' / (name + '.json'), mappings[name])
            statistics[name] = dict(raw_rows=len(mappings[name]), online=sum(r['job_type']=='online' for r in rows[name]),
                offline=120, rows=len(rows[name]), request_ids_sha256=v2.hash_object([r['request_id'] for r in rows[name]]),
                raw_row_hashes_sha256=v2.hash_object([r['raw_row_hash'] for r in mappings[name]]),
                exact_prompt_overlap_with_history=overlap[name])
        files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
        # Recheck the entire frozen closure at the publication boundary. Do not
        # scan or select again: drift invalidates this retained attempt.
        for name, meta in sources.items():
            p = source_dir / name
            if p.is_symlink() or p.stat().st_size != meta['bytes'] or digest(p) != meta['sha256']:
                raise ValueError('source dependency changed during generation')
        bound({'path': str(Path(gate_path).absolute()), 'sha256': gate_sha256})
        for ref in list(gate['bindings'].values()) + [gate['independent_review']]:
            bound(ref)
        for item in closure['files']:
            bound(item['file'])
        for phase in ('route_freeze', 'policy_freeze'):
            for ref in read(bound(gate['bindings'][phase]))['frozen_artifacts']:
                bound(ref)
        for checkpoint in read(bound(gate['bindings']['policy_freeze']))['checkpoints']:
            bound(checkpoint['file'])
        for path, sha in [(root/'SOURCE_VERSION.json', gate['source_version_sha256']),
                          (root/'COMMIT_PROVENANCE.json', gate['provenance_sha256']),
                          (Path(__file__).resolve(), gate['generator_sha256'])] + [
                              (root/name, sha) for name, sha in code_files.items()]:
            bound({'path': str(path), 'sha256': sha})
        if v2.hash_object(list(registration(root))) != registered_payload_sha256:
            raise ValueError('registered rule or parent changed during generation')
        manifest = dict(schema='maxopt-bridge-final-input-manifest-v1', status='GENERATED_PENDING_INDEPENDENT_AUDIT',
            gate_sha256=gate_sha256, rule_sha256=RULE_SHA256, parent_manifest_sha256=PARENT_SHA256,
            generator_sha256=gate['generator_sha256'], source_audit=audit, sources=parent['sources'],
            source_directory=str(source_dir), code_files=code_files,
            source_version=source_version, provenance_sha256=gate['provenance_sha256'],
            original_source_manifest_sha256=DELIVERY_MANIFEST_SHA256,
            tokenizer_version=importlib.metadata.version('tokenizers'), statistics=statistics, files=files,
            known_input_closure_sha256=gate['bindings']['known_input_closure']['sha256'],
            separation='source-time and source-row only; PSF corpus reused; no session independence claim',
            final_experiment_lock=False, physical_authorized=False)
        save_new(output / 'FINAL_INPUT_MANIFEST.json', manifest)
        save_new(output / 'STATUS.json', dict(status='GENERATED_PENDING_INDEPENDENT_AUDIT',
                 manifest_sha256=digest(output/'FINAL_INPUT_MANIFEST.json'), final_experiment_lock=False))
        return manifest
    except BaseException as exc:
        save_new(output / 'FAILURE.json', dict(status='HOLD_NO_WINDOW' if isinstance(exc, NoWindowError) else 'HOLD',
                 error_type=type(exc).__name__, error=str(exc),
                 original_attempt_retained=True, automatic_retry=False, final_experiment_lock=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-final-generation', action='store_true')
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--gate-sha256', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute_final_generation:
        parser.error('explicit --execute-final-generation is required; engineering readiness is not permission')
    result = generate(gate_path=args.gate, gate_sha256=args.gate_sha256, source_dir=args.source, output=args.output)
    print(json.dumps({'status': result['status'], 'final_experiment_lock': False}))


if __name__ == '__main__':
    main()
