"""Single deterministic forward-domain construction after strategy selection."""
import hashlib
import math
from pathlib import Path

from fidelityloop.legacy.maxopt_v2.workload import scan, raw_rows, hash_object, canonical
from fidelityloop.legacy.maxopt_v2.prepare_n5 import statistics as arrival_statistics
from fidelityloop.legacy.maxopt_v2.calibrated import validate_rows
from .common import OLD, OUT, read, lines, digest, save, save_lines

KEEP=31011205654333198806677436545472758028941581568065260776719251554600780562432
THIN_SALT='maxopt-v3-thin-v1'
WINDOW_SALT='maxopt-v3-window-v1\n'


def classify(counts):
    n=sum(counts);left=sum(counts[:15]);right=sum(counts[15:])
    mean=n/30
    cv=math.sqrt(sum((v-mean)**2 for v in counts)/30)/mean if mean else 0
    peak=max(counts)/mean if mean else 0
    return dict(steady=n>=120 and cv<=1 and right>0 and .5<=left/right<=2 and sum(v>0 for v in counts)>=24,
                recovery=n>=120 and right>0 and left>=2*right,
                burst_offline=n>=120 and cv>=1 and peak>=3)


def select(audit,bins,old_windows):
    lo=max(w['end'] for w in old_windows)+1800
    hi=audit['last']['timestamp']//1800
    candidates=[]
    for b in sorted(bins):
        if not lo<=b*1800 or b>=hi:
            continue
        if any(abs(b-w['start']//1800)<=1 for w in old_windows):
            continue
        candidates.append(dict(start=b*1800,end=(b+1)*1800,classes=classify(bins[b]),
            counts=bins[b],hash=hashlib.sha256((WINDOW_SALT+str(b*1800)).encode()).hexdigest()))
    chosen=[]
    for role in ('steady','recovery','burst_offline'):
        available=sorted((c for c in candidates if c['classes'][role]),key=lambda c:(c['hash'],c['start']))
        candidate=next((c for c in available if all(abs(c['start']-w['start'])>=3600 for w in chosen)),None)
        if candidate is None:
            raise ValueError('forward greedy disjoint selection HOLD; no resampling')
        chosen.append(dict(candidate,role=role,id='locked_test_'+role+'_v3'))
    return chosen,candidates


def keep(window,rid,threshold=KEEP,salt=THIN_SALT):
    h=hashlib.sha256((salt+'\n'+window+'\n'+rid).encode()).hexdigest()
    return int(h,16)<threshold,h


def build(output):
    selection=read(OUT/'s3/selection.json')
    if selection['status']!='FROZEN_BEFORE_NEW_WINDOWS':
        raise ValueError('model/policy must precede new requests')
    save(OUT/'s4_generation_lock.json',dict(selection_sha256=digest(OUT/'s3/selection.json'),output=str(output),
         no_resampling=True,window_salt=WINDOW_SALT,thinning_salt=THIN_SALT,threshold=str(KEEP)))
    (output/'inputs').mkdir(parents=True,exist_ok=False)
    source_dir=OLD/'sources'
    source=source_dir/'BurstGPT_without_fails_1.csv'
    parent=read(OLD/'n1_public_v4/manifest.json')
    audit,bins=scan(source)
    if audit['rows']!=1404294 or audit['source_sha256']!=parent['source_audit']['source_sha256']:
        raise ValueError('full EOF source mismatch')
    old_windows=[r['window'] for r in parent['files'].values()]
    windows,candidates=select(audit,bins,old_windows)
    save(output/'window_selection.json',dict(audit=audit,windows=windows,candidates=candidates,old_windows=old_windows,
         selection_sha256=digest(OUT/'s3/selection.json'),performance_used=False,
         arrival_statistics_previously_inspected=True))
    from tokenizers import Tokenizer
    tokpath=source_dir/'tokenizer.json';promptpath=source_dir/'prompt_locked_test.rst'
    for path in (tokpath,promptpath):
        if digest(path)!=parent['sources']['records'][path.name]['sha256']:
            raise ValueError('prompt/tokenizer parent binding')
    tokens=Tokenizer.from_file(str(tokpath)).encode(promptpath.read_text(encoding='utf-8')).ids
    promptsha=digest(promptpath)
    by_bin={w['start']//1800:w for w in windows}
    rows={w['id']:[] for w in windows};mapping={w['id']:[] for w in windows};rawmeta={w['id']:[] for w in windows}
    for ordinal,t,raw,rawhash,occurrence in raw_rows(source):
        w=by_bin.get(t//1800)
        if w is None:
            continue
        rid=hash_object([audit['source_sha256'],ordinal,occurrence])
        retained,h=keep(w['id'],rid)
        mapping[w['id']].append(dict(ordinal=ordinal,occurrence=occurrence,raw_row_hash=rawhash,
            raw_timestamp_s=t,request_id=rid,thinning_sha256=h,kept=retained))
        rawmeta[w['id']].append(dict(arrival_s=t-w['start']))
        if not retained:
            continue
        n=min(2048,max(128,int(raw['Request tokens'])))
        offset=ordinal*997%len(tokens)
        rows[w['id']].append(dict(request_id=rid,source_sha256=audit['source_sha256'],source_row_ordinal=ordinal,
            source_occurrence=occurrence,raw_row_hash=rawhash,raw_timestamp_s=t,arrival_s=t-w['start'],sim_tick=t-w['start'],
            wall_monotonic_s=None,job_type='online',prompt_token_ids=(tokens+tokens)[offset:offset+n],input_tokens=n,
            max_output_tokens=128,deadline_s=t-w['start']+60,split='locked_test',window_id=w['id'],
            prompt_source_sha256=promptsha,prompt_offset=offset,evidence=dict(arrival='observed',prompt='derived_from_PSF_text',
                 input_length='transformed_clamp_128_2048',output_budget='constructed_fixed_128')))
    # Exact prompt overlap is disclosure; it never changes window selection.
    old_prompts={hash_object(r['prompt_token_ids']) for name in parent['files']
                 for r in lines(OLD/'n1_public_v4'/name)}
    statistics={}
    for w in windows:
        name=w['id']
        for k in range(120):
            rows[name].append(dict(request_id=hash_object(['offline-v3-v1',audit['source_sha256'],name,k]),
                source_sha256=None,source_row_ordinal=None,source_occurrence=None,raw_row_hash=None,raw_timestamp_s=None,
                arrival_s=15*k,sim_tick=15*k,wall_monotonic_s=None,job_type='offline',prompt_token_ids=tokens[3000+k:3256+k],
                input_tokens=256,max_output_tokens=256,deadline_s=15*k+300,split='locked_test',window_id=name,
                prompt_source_sha256=promptsha,prompt_offset=3000+k,evidence=dict(arrival='constructed',
                    prompt='derived_from_PSF_text',input_length='constructed_256',output_budget='constructed_fixed_256')))
        rows[name].sort(key=lambda r:(r['arrival_s'],r['request_id']))
        validate_rows(rows[name],1800,True)
        save_lines(output/'inputs'/(name+'.jsonl'),rows[name])
        save(output/'mappings'/(name+'.json'),mapping[name])
        statistics[name]=dict(raw=arrival_statistics(rawmeta[name]),
            online=arrival_statistics([r for r in rows[name] if r['job_type']=='online']),
            offline=arrival_statistics([r for r in rows[name] if r['job_type']=='offline']),
            exact_prompt_overlap_with_old_parent=sum(hash_object(r['prompt_token_ids']) in old_prompts for r in rows[name]),
            rows=len(rows[name]))
    save(output/'workload_manifest.json',dict(schema='maxopt-v3-workload-v1',source_sha256=audit['source_sha256'],
        parent_manifest_sha256=digest(OLD/'n1_public_v4/manifest.json'),tokenizer_sha256=digest(tokpath),prompt_sha256=promptsha,
        threshold=str(KEEP),denominator=str(2**256),thinning_salt=THIN_SALT,statistics=statistics,
        generalization='arrival-time holdout only; prompt corpus reused; no session independence claim',
        files={w['id']+'.jsonl':digest(output/'inputs'/(w['id']+'.jsonl')) for w in windows}))
    return rows
