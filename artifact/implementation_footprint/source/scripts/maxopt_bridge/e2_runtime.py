"""E2 exact 32-cell execution contract over the accepted physical engine.
File verification and preparation are CPU-only. GPU execution is explicit.
"""
import argparse
import copy
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from .campaign import Campaign, verify_seal
from .final_inputs import bound, digest, read, save_new
from scripts.maxopt_v2.workload import hash_object

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT.parent
FROZEN = PACKAGE / 'frozen_design'
RECEIPTS = {
    'preflight': '98947aba0dcfd600da9933fe084a4ba8102bb920c276d753c33f6a148ffeb036',
    'checkpoint_tensor': '25a377b7f0fd95594b0649d75eb63ba80d14a01e9ce5c02bab6d76f5ba008d54',
    'adapter_smoke': '4347aa556bc71a954dfdcd1e696e7495a8ff6e2a08062d3afd380e75a98a3cdd',
    'site_candidate': '7db7de22113f1f34d6b412bf97c5e4728ceaf7756d65faaedf59c1bf8617ce34',
}
MODE = 'GPU_FORMAL_BRIDGE'
MOCK = 'CPU_MOCK_ONLY'
ALIAS = {'ppo_C30_selected': 'ppo_C30_raw', 'all2_anchor': 'all2'}
CANDIDATE_SCHEMA = 'maxopt-v7-e2-executable-candidate-v1'
LOCK_SCHEMA = 'maxopt-v7-e2-reviewed-execution-envelope-v1'


def require(ok, message):
    if not ok: raise ValueError(message)


def reference(path):
    p = Path(path).absolute()
    for part in (p, *p.parents):
        require(not part.is_symlink(), 'symlink paths are forbidden')
    require(p.is_file(), 'file missing: ' + str(p))
    return {'path': str(p), 'sha256': digest(p)}


def checked(ref):
    require(set(ref) == {'path', 'sha256'}, 'exact path/SHA reference required')
    p = Path(ref['path'])
    require(p.is_absolute() and '..' not in p.parts, 'absolute canonical path required')
    for part in (p, *p.parents):
        require(not part.is_symlink(), 'symlink input is forbidden')
    return bound(ref)


def frozen_arms():
    selection = read(FROZEN / 'E2_EXPERIMENT_LOCK.json')['new_checkpoint_selection']
    return {ALIAS.get(a['arm_id'], a['arm_id']): a
            for a in [*selection['arms'], selection['original_selected_arm']]}


def build_matrix():
    plan = read(FROZEN / 'E2_CELL_PLAN.json')
    cells = []
    for cell in plan['cells']:
        policy = ALIAS.get(cell['arm_id'], cell['arm_id'])
        require(policy == cell['policy_id'], 'frozen arm/policy alias mismatch')
        cells.append(dict(run_id=f"{policy}__{cell['window_id']}__r{cell['repeat']}",
            policy_id=policy, window_id=cell['window_id'], repeat=cell['repeat'],
            kind='CAPACITY' if policy == 'all2' else 'COMPARISON', initial_status='READY'))
    require(len(cells) == 32 and len({c['run_id'] for c in cells}) == 32, 'exact 32 cells required')
    return dict(schema='maxopt-v7-e2-campaign-matrix-v1',
        windows=['bridge_locked_test_recovery_v1', 'bridge_locked_test_steady_v1'],
        planned_policy_ids=list(frozen_arms()), effective_policy_ids=list(frozen_arms()),
        planned_count=32, effective_count=32, observation_seconds=2100,
        max_technical_retries=3, max_retries_per_cell=1,
        cells=cells, run_order=[c['run_id'] for c in cells])


def validate_matrix(matrix):
    require(matrix == build_matrix(), 'E2 population/order/arm identity drift')
    return matrix


def verify_sources(manifest_ref):
    manifest = read(checked(manifest_ref))
    require(manifest['schema'] == 'maxopt-v7-e2-execution-source-v1', 'source schema')
    require(Path(manifest_ref['path']) == PACKAGE / 'SOURCE_MANIFEST.json', 'source manifest path differs')
    for relative, item in manifest['files'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'source path escape')
        actual = checked(dict(path=str(PACKAGE / path), sha256=item['sha256']))
        require(actual.stat().st_size == item['bytes'], 'source size drift')
    required = {'workspace/scripts/maxopt_bridge/e2_runtime.py',
        'workspace/scripts/maxopt_bridge/e2_deployment.py',
        'workspace/scripts/maxopt_bridge/e2_launcher.py', 'e2_entry.py'}
    require(required <= set(manifest['files']), 'new entrypoint closure missing')
    # Every imported source under the delivery must be inventoried; reject shadow imports.
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name == 'scripts' or name.startswith('scripts.'):
            if not path: continue  # namespace package
            p = Path(path).resolve()
            require(p.is_relative_to(ROOT), 'foreign scripts module: ' + name)
            relative = p.relative_to(PACKAGE).as_posix()
            require(relative in manifest['files'] and digest(p) == manifest['files'][relative]['sha256'],
                    'loaded module not bound: ' + name)
    return manifest


def remaining(budget, clock=time.time):
    require(set(budget) == {'origin_epoch', 'deadline_epoch', 'limit_seconds', 'origin_receipt'}, 'budget schema')
    require(budget['limit_seconds'] == 172800 and
        budget['deadline_epoch'] == budget['origin_epoch'] + 172800, '48h non-resettable budget required')
    now = clock()
    require(math.isfinite(now) and math.isfinite(budget['origin_epoch']) and now >= budget['origin_epoch'], 'invalid/regressed budget clock')
    return max(0., budget['deadline_epoch'] - now)


class E2Campaign(Campaign):
    matrix_validator = staticmethod(validate_matrix)

    def __init__(self, out, matrix, lock_sha256, *, budget, resume=False, clock=time.time, execution_evidence=MODE):
        self.budget = copy.deepcopy(budget)
        remaining(budget, clock)
        super().__init__(out, matrix, lock_sha256, wall_seconds=172800,
            resume=resume, clock=clock, execution_evidence=execution_evidence)
        budget_path = self.out / 'E2_BUDGET.json'
        if resume: require(read(budget_path) == budget, 'resume cannot reset budget')
        else: save_new(budget_path, budget)

    def _append(self, kind, **fields):
        now = self.clock()
        require(not self.events or now >= self.events[-1]['wall_epoch'], 'wall clock regressed')
        super()._append(kind, wall_epoch=now, **fields)

    def remaining_seconds(self):
        now = self.clock()
        require(not self.events or now >= self.events[-1]['wall_epoch'], 'resume clock regressed')
        return remaining(self.budget, lambda: now)


def validate_candidate(candidate, *, clock=time.time):
    require(candidate['schema'] == CANDIDATE_SCHEMA and candidate['decision'] == 'REQUIRES_INDEPENDENT_REVIEW', 'candidate is not the defined E2 contract')
    require(candidate['workspace_directory'] == str(ROOT), 'execution workspace drift')
    require(isinstance(candidate['preparer'], str) and candidate['preparer'].strip(), 'named preparer required')
    require(set(candidate['site_receipts']) == set(RECEIPTS), 'exact four site receipts required')
    require(candidate['matrix'] == build_matrix(), 'candidate matrix changed')
    verify_sources(candidate['source_manifest'])
    for key, expected in RECEIPTS.items():
        require(candidate['site_receipts'][key]['sha256'] == expected, 'site receipt SHA differs: ' + key)
        checked(candidate['site_receipts'][key])  # Opaque: site reviewer checks semantics.
    checked(candidate['budget']['origin_receipt'])
    remaining(candidate['budget'], clock)
    for name in ('identity_reference', 'prediction_receipt', 'provenance_receipt'):
        checked(candidate[name])
    identity = read(checked(candidate['identity_reference']))
    gpu = identity.get('gpu_identity_expected', [])
    require(len(gpu) == 2 and [g['index'] for g in gpu] == [0, 1] and
        [g['uuid'] for g in gpu] == candidate['gpu_uuids'] and len(set(candidate['gpu_uuids'])) == 2,
        'current two-GPU UUID binding mismatch')
    baseline_identity = read(ROOT / 'scripts/maxopt_bridge/runtime_vendor/reference/identity_reference.json')
    for key in ('python_version', 'packages', 'model_closure_manifest', 'model_closure_sha256', 'model_bytes', 'worker_identity_expected'):
        require(identity.get(key) == baseline_identity[key], 'identity model/runtime field changed: ' + key)
    for current, baseline in zip(gpu, baseline_identity['gpu_identity_expected']):
        require({k:v for k,v in current.items() if k != 'uuid'} == {k:v for k,v in baseline.items() if k != 'uuid'}, 'hardware/driver identity changed beyond authorized UUID')
    require(all(isinstance(value, str) and value.startswith('GPU-') for value in candidate['gpu_uuids']), 'invalid GPU UUID')
    require(candidate['execution_evidence'] == MODE, 'production candidate cannot use mock mode')
    output = Path(candidate['output_directory'])
    require(output.is_absolute() and '..' not in output.parts and not output.is_relative_to(PACKAGE)
        and not PACKAGE.is_relative_to(output), 'output must be disjoint from source')
    for p in (output, *output.parents): require(not p.is_symlink(), 'output symlink')
    require(Path(candidate['python']).is_file() and Path(candidate['python']).is_absolute(), 'accepted Python missing')
    require(Path(candidate['model']).is_dir() and Path(candidate['model']).is_absolute(), 'model directory missing')
    arms = frozen_arms()
    require(set(candidate['checkpoints']) == set(arms), 'exact five raw arms required')
    for policy, arm in arms.items():
        require(candidate['checkpoints'][policy]['sha256'] == arm['checkpoint_sha256'], 'checkpoint mapping mismatch: ' + policy)
        checked(candidate['checkpoints'][policy])
    from . import formal_runtime as fr
    for window, ref in candidate['inputs'].items():
        require(window in candidate['matrix']['windows'], 'unknown window')
        expected = next((FROZEN / 'inputs').glob('*_' + window + '.jsonl'))
        require(ref['sha256'] == digest(expected), 'window input differs')
        rows = [json.loads(line) for line in checked(ref).read_text().splitlines()]
        fr.validate_requests(rows, window)
    require(set(candidate['inputs']) == set(candidate['matrix']['windows']), 'missing window')
    require(candidate['resources'] == resources() and candidate['required_ports'] == [], 'resource contract changed')
    return candidate


def resources():
    return dict(identity_seconds=300, minimum_cpu_count=16, minimum_free_bytes=137438953472,
        minimum_memory_bytes=68719476736, output_bytes_soft_cap=68719476736, wall_seconds=172800)


def prepare(bindings_path, target, *, budget_origin_utc):
    """Only bind site paths/UUIDs/opaque receipts; never infer unknown receipt schemas."""
    spec = read(bindings_path)
    allowed = {'preparer', 'output_directory', 'model', 'python', 'gpu_uuids', 'checkpoints',
        'site_receipts', 'identity_reference', 'prediction_receipt', 'provenance_receipt', 'budget_origin_receipt'}
    require(set(spec) == allowed, 'site bindings fields differ from documented contract')
    origin = datetime.fromisoformat(budget_origin_utc.replace('Z', '+00:00'))
    require(origin.tzinfo is not None, 'budget origin requires explicit timezone')
    inputs = {window: reference(next((FROZEN / 'inputs').glob('*_' + window + '.jsonl')))
              for window in build_matrix()['windows']}
    candidate = dict(schema=CANDIDATE_SCHEMA, decision='REQUIRES_INDEPENDENT_REVIEW',
        preparer=spec['preparer'], workspace_directory=str(ROOT), output_directory=spec['output_directory'],
        model=spec['model'], python=spec['python'], gpu_uuids=spec['gpu_uuids'], execution_evidence=MODE,
        matrix=build_matrix(), source_manifest=reference(PACKAGE / 'SOURCE_MANIFEST.json'),
        inputs=inputs, checkpoints=spec['checkpoints'], site_receipts=spec['site_receipts'],
        identity_reference=spec['identity_reference'], prediction_receipt=spec['prediction_receipt'],
        provenance_receipt=spec['provenance_receipt'], required_ports=[], resources=resources(),
        budget=dict(origin_epoch=origin.timestamp(), deadline_epoch=origin.timestamp()+172800,
                    limit_seconds=172800, origin_receipt=spec['budget_origin_receipt']))
    validate_candidate(candidate)
    save_new(target, candidate)
    return reference(target)


def review_candidate(candidate_ref, review_ref):
    candidate = validate_candidate(read(checked(candidate_ref)))
    review = read(checked(review_ref))
    require(review.get('schema') == 'maxopt-v7-e2-execution-review-v1' and
        review.get('decision') == 'AUTHORIZED_AFTER_INDEPENDENT_REVIEW' and
        review.get('candidate') == candidate_ref and review.get('source_manifest') == candidate['source_manifest'] and
        review.get('reviewer') and review['reviewer'] != candidate['preparer'], 'independent review must bind exact candidate/source')
    required_checks = {'site_receipt_semantics', 'checkpoint_tensor_smoke', 'identity_and_model',
        'frozen_cpu_predictions', 'v6_provenance', 'budget_origin', 'multi_arm_runtime', 'canary_disposition'}
    require(set(review.get('checks', {})) == required_checks and
        all(v is True for v in review['checks'].values()), 'review checks incomplete')
    require(isinstance(review.get('evidence'), list) and review['evidence'], 'review evidence required')
    for ref in review['evidence']: checked(ref)
    return candidate


def authorize(candidate_ref, review_ref, output):
    candidate = review_candidate(candidate_ref, review_ref)
    envelope = dict(schema=LOCK_SCHEMA, decision='AUTHORIZED_AFTER_INDEPENDENT_REVIEW',
        candidate=candidate_ref, independent_review=review_ref,
        python=candidate['python'], workspace_directory=candidate['workspace_directory'],
        output_directory=candidate['output_directory'])
    save_new(output, envelope)
    return reference(output)


def verify_lock(path, expected_sha256, output=None):
    envelope = read(checked(dict(path=str(Path(path).absolute()), sha256=expected_sha256)))
    require(envelope['schema'] == LOCK_SCHEMA and envelope['decision'] == 'AUTHORIZED_AFTER_INDEPENDENT_REVIEW', 'unreviewed lock cannot launch')
    candidate = review_candidate(envelope['candidate'], envelope['independent_review'])
    require(all(envelope[k] == candidate[k] for k in ('python','workspace_directory','output_directory')), 'envelope path drift')
    if output is not None: require(Path(output).absolute() == Path(candidate['output_directory']), 'CLI output differs')
    return candidate


def execute_campaign(lock_path, expected_sha256, output, *, resume=False, retry_receipt=None, technical_stop_receipt=None):
    candidate = verify_lock(lock_path, expected_sha256, output)
    from . import formal_runtime as fr
    from .e2_deployment import E2Deployment, make_plan, run_cell
    with fr.exclusive_campaign(output):
        candidate = verify_lock(lock_path, expected_sha256, output)
        output = Path(output)
        if (output / 'E2_CAMPAIGN_MANIFEST.json').exists():
            require(resume, 'sealed output cannot be overwritten')
            verify_seal(output, 'E2_CAMPAIGN_MANIFEST.json')
            return read(checked(read(output / 'FINALIZATION_ACCEPTED.json')['result']))
        deployment = E2Deployment(candidate)
        runtime, context = fr.load_runtime(), fr.f4.load_context()
        identity = read(checked(candidate['identity_reference']))
        inputs = {w: [json.loads(line) for line in checked(ref).read_text().splitlines()]
                  for w, ref in candidate['inputs'].items()}
        limits = read(fr.RUNTIME_TEMPLATE)['limits']
        reserve = limits['setup_seconds'] + 2100 + limits['cleanup_seconds'] + 600
        # A too-short allocation fails admission without starting a cell.
        fr.check_resources(candidate, output, required_remaining_seconds=remaining(candidate['budget']))
        campaign = E2Campaign(output, candidate['matrix'], expected_sha256,
                              budget=candidate['budget'], resume=resume)
        if not resume: shutil.copyfile(lock_path, output / 'EXPERIMENT_LOCK.json')
        require(digest(output / 'EXPERIMENT_LOCK.json') == expected_sha256, 'stored lock drift')
        campaign.recover_recorded_results()
        if retry_receipt: campaign.authorize_retry(*retry_receipt)
        if technical_stop_receipt: campaign.stop_technical(*technical_stop_receipt)
        def prelaunch(_remaining):
            verify_lock(lock_path, expected_sha256, output)
            deployment.verify_current()
            require(campaign.remaining_seconds() >= reserve, 'budget reserve exhausted before launch')
            resource = fr.check_resources(candidate, output, required_remaining_seconds=campaign.remaining_seconds())
            observed = fr.capture_current_identity(candidate['model'], runtime, identity, 300)
            require(campaign.remaining_seconds() >= limits['setup_seconds']+2100+limits['cleanup_seconds'], 'identity consumed cleanup reserve')
            return dict(identity=observed, resources=resource, lock_sha256=expected_sha256)
        def runner(cell, attempt, folder, unused_remaining):
            plan = make_plan(cell, context, identity, candidate=candidate, deployment=deployment)
            return run_cell(runtime, context, plan, inputs[cell['window_id']], identity, folder,
                deployment=deployment, candidate=candidate, attempt=attempt, lock_sha256=expected_sha256,
                remaining_seconds=campaign.remaining_seconds(), output_limit=candidate['resources']['output_bytes_soft_cap'],
                campaign_out=output, prelaunch=prelaunch)
        summary = campaign.run(runner, reserve_seconds=reserve)
        if summary['complete']:
            return finalize(campaign, candidate, lock_path, expected_sha256, deployment, runtime, identity)
        return summary


def finalize(campaign, candidate, lock_path, expected_sha256, deployment, runtime, identity):
    """Retry archival in new immutable attempts; never overwrite partial evidence."""
    from . import formal_runtime as fr
    output = campaign.out
    accepted = output / 'FINALIZATION_ACCEPTED.json'
    if accepted.exists():
        record = read(accepted)
        folder = Path(record['folder'])
        require(folder.parent == output / 'finalization_attempts', 'finalization receipt escaped output')
        require(checked(record['manifest']) == folder / 'FINALIZATION_MANIFEST.json', 'finalization manifest binding mismatch')
        verify_seal(folder, 'FINALIZATION_MANIFEST.json')
        require(checked(record['result']) == folder / 'E2_CAMPAIGN_RESULT.json', 'finalization result escaped accepted folder')
        result = read(checked(record['result']))
        current = campaign.summary()
        require(result['lock_sha256'] == expected_sha256, 'finalization lock mismatch')
        require(result.get('complete') is True and current['complete'] is True and
                result.get('cells') == current['cells'], 'finalization result does not match completed journal')
    else:
        verify_lock(lock_path, expected_sha256, output)
        deployment.verify_current()
        require(campaign.remaining_seconds() > 300, 'insufficient finalization identity budget; retain evidence')
        attempts = output / 'finalization_attempts'
        attempts.mkdir(exist_ok=True)
        folder = attempts / ('attempt_' + format(len(list(attempts.iterdir()))+1, '02d'))
        folder.mkdir(exist_ok=False)
        after = fr.capture_current_identity(candidate['model'], runtime, identity, 300)
        save_new(folder / 'IDENTITY_AFTER.json', after)
        require(not after['errors'], 'final identity/release audit failed; retain unsealed evidence')
        result = campaign.summary()
        save_new(folder / 'E2_CAMPAIGN_RESULT.json', result)
        handoff = folder / 'cpu_handoff'
        handoff.mkdir(exist_ok=False)
        shutil.copytree(PACKAGE, handoff / 'executor', ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
        refs = {'candidate': read(lock_path)['candidate'], 'review': read(lock_path)['independent_review']}
        refs.update(candidate['site_receipts'])
        refs.update({name: candidate[name] for name in ('identity_reference','prediction_receipt','provenance_receipt')})
        refs['budget_origin'] = candidate['budget']['origin_receipt']
        refs.update({'checkpoint_'+name: ref for name,ref in candidate['checkpoints'].items()})
        refs.update({'review_evidence_'+str(i): ref for i,ref in enumerate(read(checked(refs['review']))['evidence'])})
        index = {}
        for name, ref in refs.items():
            source = checked(ref)
            target = handoff / 'dependencies' / (name + source.suffix)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            require(digest(target) == ref['sha256'], 'handoff copy changed')
            index[name] = dict(original=ref, path=target.relative_to(folder).as_posix())
        save_new(handoff / 'REFERENCE_INDEX.json', index)
        require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= candidate['resources']['output_bytes_soft_cap'], 'output budget exceeded; retain unsealed')
        require(campaign.remaining_seconds() > 0, 'deadline exhausted during finalization; retain evidence')
        fr.seal(folder, 'FINALIZATION_MANIFEST.json')
        save_new(accepted, dict(folder=str(folder), result=reference(folder/'E2_CAMPAIGN_RESULT.json'),
            manifest=reference(folder/'FINALIZATION_MANIFEST.json')))
    # Both fresh and resumed finalization obey the original budget and output cap.
    require(campaign.remaining_seconds() > 0, 'deadline exhausted before final seal; retain evidence')
    require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= candidate['resources']['output_bytes_soft_cap'],
            'output budget exceeded before final seal; retain evidence')
    # A crash after ACCEPTED but before this seal resumes here without duplicating files.
    fr.seal(output, 'E2_CAMPAIGN_MANIFEST.json')
    return result
