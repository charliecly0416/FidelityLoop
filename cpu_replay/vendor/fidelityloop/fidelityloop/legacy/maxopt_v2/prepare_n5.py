"""Explicit, one-shot held-out derivation and pre-observation predictions.

This command is prepared on CPU but must NOT be invoked during N4b selection.
It consumes the existing C2 recipe without resetting its revision allowance.
It never launches real inference. The GPU work order owns the lifecycle gate.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from .calibrated import validate_rows
from .reselect import REGIMES, digest, read, lines, save, save_lines, check_bindings, execute_cell, verify

PARENT_SHA = '4973bfe1738e4b5aac0eec0386571c7d97263e0e592f7ba4879bd2e25e60d4a9'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def statistics(rows):
    result = {}
    for width in (1,60):
        counts = [0]*(1800//width)
        for row in rows:
            counts[row['arrival_s']//width] += 1
        mean = sum(counts)/len(counts)
        centered = [x-mean for x in counts]
        ss = sum(x*x for x in centered)
        result[str(width)] = dict(horizon_seconds=1800, arrival_count=sum(counts),
            mean_count_per_bin=mean, cv=math.sqrt(ss/len(counts))/mean if mean else None,
            peak_to_mean=max(counts)/mean if mean else None,
            lag1_autocorrelation=sum(centered[i]*centered[i-1] for i in range(1,len(counts)))/ss if ss else None,
            counts_including_empty_bins=counts,
            undefined_reasons=dict(**({'cv':'zero_mean','peak_to_mean':'zero_mean'} if not mean else {}),
                                   **({'lag1_autocorrelation':'zero_variance'} if not ss else {})))
    return result


def derive(rows, window, registration, corpus, source_sha):
    """Same Stage 2 thinning/offline recipe, generalized to a named split.

    Development fixtures can test equivalence without opening held-out files.
    """
    split = 'locked_test' if window.startswith('locked_test_') else window.split('_',1)[0]
    validate_rows(rows,1800,allow_locked_test=split=='locked_test')
    if split != corpus['split'] or any(r['window_id'] != window or r['split'] != split for r in rows):
        raise ValueError('split/window mismatch')
    if registration['salt'] != 'maxopt-stage2-thin-v1' or registration['capacity_revision_number'] != 1:
        raise ValueError('accepted recipe changed')
    candidate = next(c for c in registration['candidates'] if c['candidate_id']=='C2')
    threshold = int(candidate['sha256_keep_threshold_decimal'])
    if candidate['threshold_denominator_decimal'] != str(2**256):
        raise ValueError('threshold denominator')
    tokens = corpus['tokens']
    if len(tokens) < 3375:
        raise ValueError('prompt corpus too short')
    derived, mapping = [], []
    for line, row in enumerate(rows,1):
        offset = row['prompt_offset']
        if row['prompt_source_sha256'] != corpus['prompt_source_sha256'] or row['prompt_token_ids'] != (tokens+tokens)[offset:offset+row['input_tokens']]:
            raise ValueError('parent tokenizer/prompt mismatch')
        h = hashlib.sha256((registration['salt']+'\n'+window+'\n'+row['request_id']).encode()).hexdigest() if row['job_type']=='online' else None
        keep = h is not None and int(h,16)<threshold
        if keep:
            derived.append(copy.deepcopy(row))
        mapping.append(dict(source_file_sha256=source_sha, source_jsonl_line_1_based=line,
                            source_request_id=row['request_id'], raw_timestamp_s=row['raw_timestamp_s'],
                            source_row_ordinal=row['source_row_ordinal'], raw_row_hash=row['raw_row_hash'],
                            source_arrival_s=row['arrival_s'], source_deadline_s=row['deadline_s'],
                            hash_digest_hex=h, kept=keep,
                            derived_request_id_or_null=row['request_id'] if keep else None,
                            derived_arrival_s_or_null=row['arrival_s'] if keep else None,
                            derived_deadline_s_or_null=row['deadline_s'] if keep else None))
    constructed = []
    for k in range(120):
        rid = hashlib.sha256(canonical(['offline-stage2-v1',corpus['arrival_source_sha256'],window,k])).hexdigest()
        offset = 3000+k
        row = dict(request_id=rid,source_sha256=None,source_row_ordinal=None,source_occurrence=None,
                   raw_row_hash=None,raw_timestamp_s=None,arrival_s=15*k,sim_tick=15*k,wall_monotonic_s=None,
                   job_type='offline',prompt_token_ids=tokens[offset:offset+256],input_tokens=256,max_output_tokens=256,
                   deadline_s=15*k+300,split=split,window_id=window,prompt_source_sha256=corpus['prompt_source_sha256'],
                   prompt_offset=offset,evidence=dict(arrival='constructed',prompt='derived_from_PSF_text',
                      input_length='constructed_256',output_budget='constructed_fixed_256'))
        derived.append(row)
        constructed.append(dict(constructed_index=k,request_id=rid,arrival_s=15*k,deadline_s=15*k+300,
                                prompt_offset=offset,prompt_source_sha256=corpus['prompt_source_sha256'],
                                tokenizer_sha256=corpus['tokenizer_sha256']))
    derived.sort(key=lambda r:(r['arrival_s'],r['request_id']))
    validate_rows(derived,1800,allow_locked_test=split=='locked_test')
    stats = {label:{kind:statistics(population if kind=='all' else [r for r in population if r['job_type']==kind])
                   for kind in ('all','online','offline')} for label,population in (('raw',rows),('derived',derived))}
    return derived, dict(candidate='C2',raw_row_mapping=mapping,constructed_offline_mapping=constructed,
                         statistics=stats,raw_horizon_seconds=1800,derived_horizon_seconds=1800)


def prepare(frozen, parent, sources, output):
    frozen,parent,sources,output = map(Path,(frozen,parent,sources,output))
    check_bindings(frozen)
    verify(frozen)
    if digest(parent/'manifest.json') != PARENT_SHA:
        raise ValueError('parent manifest changed')
    manifest = read(parent/'manifest.json')
    records = manifest['sources']['records']
    for name in ('tokenizer.json','prompt_locked_test.rst'):
        if digest(sources/name) != records[name]['sha256']:
            raise ValueError('held-out source/tokenizer binding changed')
    for regime in REGIMES:
        name = f'locked_test_{regime}_v2.jsonl'
        if digest(parent/name) != manifest['files'][name]['sha256']:
            raise ValueError('parent held-out input changed')
    # Exclusive persistent lock belongs to frozen selection, not a new output
    # directory. Failures preserve the partial run and cannot silently resample.
    save(frozen/'heldout_generation_lock.json', dict(output=str(output.resolve()),
         n5_plan_sha256=digest(frozen/'n5_plan.json'), recipe_sha256=digest(frozen/'inputs/revision_registration.json'),
         capacity_revisions_used=1, generation='one deterministic C2 application to held-out; no new recipe'))
    output.mkdir(parents=True,exist_ok=False)
    from tokenizers import Tokenizer
    import importlib.metadata
    tokenizer = Tokenizer.from_file(str(sources/'tokenizer.json'))
    tokens = tokenizer.encode((sources/'prompt_locked_test.rst').read_text(encoding='utf-8')).ids
    corpus = dict(split='locked_test',tokens=tokens,prompt_source_sha256=records['prompt_locked_test.rst']['sha256'],
                  tokenizer_sha256=records['tokenizer.json']['sha256'],
                  arrival_source_sha256=records['BurstGPT_without_fails_1.csv']['sha256'])
    recipe = read(frozen/'inputs/revision_registration.json')
    plan, models, contract = read(frozen/'n5_plan.json'),read(frozen/'models.json'),read(frozen/'inputs/contract.json')
    data = {}
    for regime in REGIMES:
        window = f'locked_test_{regime}_v2'
        rows, mapping = derive(lines(parent/(window+'.jsonl')),window,recipe,corpus,digest(parent/(window+'.jsonl')))
        save_lines(output/(window+'.jsonl'),rows)
        save(output/(window+'_mapping_statistics.json'),mapping)
        data[window] = rows
    # Generate ALL windows before any predictions; retain infeasible windows.
    predictions = []
    for model,parameters in models.items():
        for policy in plan['policies']:
            for window,rows in data.items():
                cell = execute_cell(output,f'predictions/{model}/{policy["id"]}/{window}',rows,contract,
                                    parameters,policy['policy'],allow_locked_test=True)
                cell.update(model=model,policy=policy['id'],window=window)
                predictions.append(cell)
    save(output/'predictions.json',dict(cells=predictions,n5_plan_sha256=digest(frozen/'n5_plan.json'),
                                      models_sha256=digest(frozen/'models.json'),kind='independent_closed_loop',
                                      physical_completions_consumed=False))
    bindings = {str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save(output/'prediction_lock.json',dict(status='FROZEN_BEFORE_REAL_OBSERVATION',files=bindings,
         n5_plan_sha256=digest(frozen/'n5_plan.json'),tokenizers_version=importlib.metadata.version('tokenizers'),
         parent_manifest_sha256=PARENT_SHA,capacity_revisions_used=1))
    return dict(status='PREDICTIONS_FROZEN',prediction_lock_sha256=digest(output/'prediction_lock.json'),
                real_runs_planned=plan['runs'],cpu_prediction_runs=len(predictions))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen','parent','sources','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.frozen,args.parent,args.sources,args.output),indent=2))
