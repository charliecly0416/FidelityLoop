"""Bridge formal orchestration over the unchanged E1 v3 physical implementation.

Imports and plan construction are CPU-only. Production uses execute_campaign
with an external, independently reviewed EXPERIMENT_LOCK SHA. CPU fixtures must
explicitly supply a fake backend and carry CPU_MOCK_ONLY in every output.
"""
import asyncio
import contextlib
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import subprocess
import time
import traceback
from datetime import datetime, timezone

from . import f4, f4_runtime
from .campaign import Campaign, WINDOW_IDS, validate_matrix, verify_seal
from .final_inputs import bound, digest, read, registration, save_new
from .metrics import evaluate_requests
from .preflight_identity import hash_model, compare_identity
from .preflight_runtime import RUNTIME_TEMPLATE, load_runtime, verify_sources
from .protocol import ROOT, F4_IDS
from scripts.maxopt_v2.calibrated import validate_rows
from scripts.maxopt_v2.workload import hash_object

MODE = 'GPU_FORMAL_BRIDGE'
MOCK = 'CPU_MOCK_ONLY'
LOCK_SCHEMA = 'maxopt-bridge-experiment-lock-v1'
CODE_FILES = ('formal_runtime.py','campaign.py','final_inputs.py','f4.py','f4_runtime.py',
              'protocol.py','metrics.py','preflight_runtime.py','preflight_identity.py',
              'gpu_formal_launcher.py','runtime_monitor.py')
BINDINGS = {'authorization','environment_acceptance','environment_identity','route_freeze','policy_freeze',
            'final_input_manifest','final_input_acceptance','prediction_lock','runtime_acceptance'}
LOCK_FIELDS = {'schema','scope','decision','preparer','workspace_directory','output_directory',
               'execution_evidence','bindings','matrix','inputs','code_files','source_version_sha256',
               'provenance_sha256','model','python','gpu_uuids','resources','required_ports','independent_review'}


def validate_requests(rows, window):
    validate_rows(rows, 1800, True)
    if not rows or any(r['window_id'] != window or r['split'] != 'locked_test' for r in rows):
        raise ValueError('exact final window and locked_test membership required')
    online = [r for r in rows if r['job_type']=='online']
    offline = [r for r in rows if r['job_type']=='offline']
    if (not online or len(offline)!=120 or [r['arrival_s'] for r in offline]!=list(range(0,1800,15)) or
            any((r['input_tokens'],r['max_output_tokens'])!=(256,256) for r in offline) or
            any(not 128<=r['input_tokens']<=2048 or r['max_output_tokens']!=128 for r in online) or
            any(r['deadline_s']!=r['arrival_s']+(60 if r['job_type']=='online' else 300) for r in rows) or
            any(r['request_id'].startswith('setup_g') for r in rows)):
        raise ValueError('final request population/token/deadline contract differs')


def make_plan(cell, context, identity, *, model, python, execution_evidence=MODE, deployment=None):
    template=read(RUNTIME_TEMPLATE);policy=cell['policy_id']
    raw_policy=policy not in ('all2',*F4_IDS)
    raw_identity=None
    if raw_policy:
        if deployment is None:raise ValueError('no independently registered raw PPO physical factory')
        from .ppo_deployment import Deployment
        if type(deployment) is not Deployment:raise ValueError('verified raw PPO deployment required')
        deployment.verify_current();raw_identity=deployment.identity(policy)
        if deployment.synthetic and execution_evidence!=MOCK:raise ValueError('synthetic raw PPO requires CPU mock evidence')
    plan=dict(schema='maxopt-bridge-all2-runtime-plan-v1' if policy=='all2' else 'maxopt-bridge-f4-runtime-plan-v1',
        bridge_contract_sha256=context['receipt']['contract_sha256'], formal=True,
        run_id=cell['run_id'], window_id=cell['window_id'], policy_id=policy, repeat=cell['repeat'],
        kind=cell['kind'], execution_evidence=execution_evidence,
        horizon_seconds=2100, window_seconds=1800, drain_seconds=300,
        policy_spec={'name':'all2'} if policy=='all2' or raw_policy else f4_runtime.policy_spec(context['arms'][policy]),
        adapter_initial_target=2, service_model=context['models']['E'], guard_safety=context['safety'],
        simulator_contract=context['simulator_contract'], prices=context['simulator_contract']['accounting'],
        model=str(Path(model).resolve()), python=str(Path(python).resolve()),
        gpu_identity_expected=identity['gpu_identity_expected'])
    if raw_policy:
        plan.update(schema='maxopt-bridge-raw-ppo-runtime-plan-v1',deployment_identity=raw_identity,
                    policy_spec=dict(name='raw_ppo_capacity',training_model=raw_identity['training_model']))
    for key in ('runtime','effective_runtime_expected','limits','setup_probe','synthetic_api'):
        plan[key]=template[key]
    return copy.deepcopy(plan)


def select_controller(runtime, context, plan, identity, *, deployment=None):
    """Reject drift before creating journals/backend; never call runtime.run()."""
    cell={k:plan[k] for k in ('run_id','window_id','policy_id','repeat','kind')}
    expected=make_plan(cell,context,identity,model=plan['model'],python=plan['python'],
                       execution_evidence=plan['execution_evidence'],deployment=deployment)
    if plan!=expected or plan['execution_evidence'] not in (MODE,MOCK):
        raise ValueError('formal plan differs from frozen common contract')
    if (cell['window_id'] not in WINDOW_IDS or type(cell['repeat']) is not int or
            cell['run_id']!=f"{cell['policy_id']}__{cell['window_id']}__r{cell['repeat']}" or
            ((cell['kind'],cell['repeat'])!=('CAPACITY',1) if cell['policy_id']=='all2'
             else cell['kind']!='COMPARISON' or cell['repeat'] not in (1,2,3))):
        raise ValueError('unregistered formal cell identity')
    gpu=identity.get('gpu_identity_expected',[])
    if len(gpu)!=2 or [r['index'] for r in gpu]!=[0,1] or len({r['uuid'] for r in gpu})!=2:
        raise ValueError('exact two-device current GPU identity required')
    runtime._validate_runtime_plan(plan)
    if cell['policy_id'] not in ('all2',*F4_IDS):
        from .ppo_deployment import controller_class
        return controller_class(runtime,deployment,cell['policy_id'],allow_synthetic=plan['execution_evidence']==MOCK)
    return runtime.FormalV2Controller if cell['policy_id']=='all2' else f4_runtime.controller_class(runtime,context)


def check_ports(ports):
    if not isinstance(ports,list) or len(set(ports))!=len(ports) or any(type(p) is not int or not 1<=p<=65535 for p in ports):
        raise ValueError('required_ports must be explicit unique port numbers')
    with contextlib.ExitStack() as stack:
        for port in ports:
            sock=stack.enter_context(socket.socket(socket.AF_INET,socket.SOCK_STREAM))
            try:sock.bind(('127.0.0.1',port))
            except OSError as exc:raise ValueError('required port unavailable: '+str(port)) from exc


def capture_current_identity(model, runtime, reference, seconds):
    """Use the accepted current-site derivation; never the source site's UUIDs."""
    packages={}
    for name in reference['packages']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    observed=dict(python_version=platform.python_version(),packages=packages,
                  gpu_snapshot=runtime.gate.evidence.snapshot_gpu(5),model_path=str(model))
    observed.update(hash_model(Path(model),deadline=time.monotonic()+seconds,reference=reference))
    observed.update(compare_identity(observed,reference))
    return observed


def file_inventory(folder):
    folder=Path(folder).resolve();files={}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():raise ValueError('raw evidence cannot contain symlinks')
        if path.is_file():
            before=path.stat();sha=digest(path);after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('evidence changed while hashing')
            files[path.relative_to(folder).as_posix()]=dict(bytes=after.st_size,sha256=sha)
    return files


def seal(folder, name, *, pause=time.sleep):
    folder=Path(folder)
    if (folder/name).exists():raise ValueError('evidence seal cannot be overwritten')
    before=file_inventory(folder);pause(.05)
    if before!=file_inventory(folder):raise ValueError('raw output is not quiescent')
    save_new(folder/name,dict(schema='maxopt-bridge-evidence-seal-v1',files=before))
    verify_seal(folder,name)


async def observe_supervised(runtime, instance, deadline, output_limit, campaign_out):
    from .runtime_monitor import supervise
    return await supervise(runtime, instance, deadline, output_limit, campaign_out)


def terminal_metrics(rows, terminals):
    if set(terminals)!={r['request_id'] for r in rows}:return None
    records={rid:dict(completed_at=t['at_s'] if t['status']=='completed' else None) for rid,t in terminals.items()}
    return evaluate_requests(rows,records,horizon=2100)


def run_cell(runtime, context, plan, rows, identity, out, *, attempt, lock_sha256, remaining_seconds,
             output_limit, campaign_out, backend_factory=None, supervisor=None, prelaunch=None, pause=time.sleep, deployment=None):
    cls=select_controller(runtime,context,plan,identity,**({'deployment':deployment} if deployment is not None else {}))
    return _run_cell(runtime,context,plan,rows,identity,out,attempt=attempt,lock_sha256=lock_sha256,
        remaining_seconds=remaining_seconds,output_limit=output_limit,campaign_out=campaign_out,
        backend_factory=backend_factory,supervisor=supervisor,prelaunch=prelaunch,pause=pause,
        deployment=deployment,controller_class=cls)


def _run_cell(runtime, context, plan, rows, identity, out, *, attempt, lock_sha256, remaining_seconds,
              output_limit, campaign_out, controller_class, backend_factory=None, supervisor=None,
              prelaunch=None, pause=time.sleep, deployment=None):
    """Orchestrate inherited methods; injection is allowed only in labeled mocks."""
    mode=plan['execution_evidence']
    if mode==MOCK and backend_factory is None:raise ValueError('mock mode requires an explicit fake backend')
    if mode==MODE and (backend_factory is not None or supervisor is not None or prelaunch is None):
        raise ValueError('production permits no backend/supervisor substitution and requires lock prelaunch')
    cls=controller_class;validate_requests(rows,plan['window_id'])
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save_new(out/'expected_requests.json',rows);save_new(out/'run_plan.json',plan)
    instance=None;original={};errors=[];started=time.monotonic()
    with (out/'controller_stdout.log').open('x') as stdout, (out/'controller_stderr.log').open('x') as stderr:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try:
                if prelaunch:
                    accepts_remaining=getattr(getattr(prelaunch,'__code__',None),'co_argcount',0)>0
                    checked=prelaunch(remaining_seconds) if accepts_remaining else prelaunch();save_new(out/'prelaunch.json',checked)
                    if checked.get('identity',{}).get('errors'):
                        raise ValueError('current identity audit failed; complete evidence retained in prelaunch.json')
                instance=cls(plan,out,**({'backend_factory':backend_factory} if backend_factory else {}))
                runner=supervisor or observe_supervised
                original=asyncio.run(runner(runtime,instance,started+remaining_seconds,output_limit,campaign_out))
                if deployment is not None:deployment.verify_current()
            except BaseException as exc:
                errors.append(dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
                traceback.print_exc()
    terminals=instance.terminals if instance else {}
    if instance is not None and not original:
        original=getattr(instance,'monitoring_result',{})
    monitoring=getattr(instance,'monitoring_status',None)
    clean=(original.get('cleanup_complete') is True and instance is not None and
           all(d['state']=='off' for d in instance.actuator.devices))
    quiet=(instance is None or all(j.closed and not j.thread.is_alive() for j in (instance.raw,instance.ledger)))
    # A failed prelaunch never owned workers; failed construction may have open
    # journals and must stay unsealed. Only successful construction owns cleanup.
    if instance is None:clean=not errors or not (out/'controller_events.jsonl').exists()
    metrics=terminal_metrics(rows,terminals)
    monitor_clean=monitoring is None or monitoring['complete'] and not monitoring['fault']
    technical=original.get('technical_valid') is True and clean and quiet and metrics is not None and not errors and monitor_clean
    save_new(out/'request_terminals.json',dict(terminals=terminals,missing_request_ids=sorted({r['request_id'] for r in rows}-set(terminals))))
    save_new(out/'wrapper_errors.json',errors)
    result=dict(schema='maxopt-bridge-formal-cell-result-v1',run_id=plan['run_id'],attempt=attempt,
        lock_sha256=lock_sha256,execution_evidence=mode,technical_valid=technical,
        cleanup_complete=clean,safe_to_seal=bool(clean and quiet and monitor_clean),request_metrics=metrics,
        monitoring_status=monitoring,
        inherited_result_file='run_result.json' if (out/'run_result.json').exists() else None,
        inherited_gate_used_for_bridge_decision=False,elapsed_wall_seconds=time.monotonic()-started,
        classification='TECHNICAL_INVALID' if not technical else 'COMPLETED' if metrics['P1plus'] else 'NEGATIVE_RESULT')
    save_new(out/'CELL_RESULT.json',result)
    if result['safe_to_seal']:seal(out,'CELL_MANIFEST.json',pause=pause)
    return out/'CELL_RESULT.json'


def verify_references(value, collected=None):
    """Check exact absolute file references, including receipt leaf artifacts."""
    collected={} if collected is None else collected
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            path=bound(value);key=str(path)
            if key in collected and collected[key]['sha256']!=value['sha256']:
                raise ValueError('conflicting reference digests')
            if key not in collected:
                collected[key]=value.copy()
                if path.suffix=='.json':verify_references(read(path),collected)
        else:
            for child in value.values():verify_references(child,collected)
    elif isinstance(value,list):
        for child in value:verify_references(child,collected)
    return collected


def verify_lock(path, expected_sha256, output, *, root=ROOT):
    """Pure file checks precede any device query, output creation or launch."""
    root=Path(root).resolve();output=Path(output).resolve()
    path=bound(dict(path=str(Path(path).absolute()),sha256=expected_sha256));lock=read(path)
    if (set(lock)!=LOCK_FIELDS or lock.get('schema')!=LOCK_SCHEMA or lock.get('scope')!='S09_FORMAL_BRIDGE' or
            lock.get('decision')!='AUTHORIZED_AFTER_INDEPENDENT_REVIEW' or lock.get('execution_evidence')!=MODE or
            lock.get('workspace_directory')!=str(root) or lock.get('output_directory')!=str(output) or
            not lock.get('preparer') or output.is_relative_to(root) or root.is_relative_to(output)):
        raise ValueError('exact current formal execution permit required')
    registration(root);verify_sources()
    if (digest(root/'SOURCE_VERSION.json')!=lock['source_version_sha256'] or
            digest(root/'COMMIT_PROVENANCE.json')!=lock['provenance_sha256'] or
            set(lock['bindings'])!=BINDINGS):
        raise ValueError('source provenance or prerequisite closure differs')
    required={'scripts/maxopt_bridge/'+name for name in CODE_FILES}
    if not required<=set(lock['code_files']):raise ValueError('formal code closure incomplete')
    for relative,sha in lock['code_files'].items():
        p=root/relative
        if not p.resolve().is_relative_to(root):raise ValueError('code path escapes workspace')
        bound(dict(path=str(p),sha256=sha))
    # Verify the module running this gate, not just a detached workspace copy.
    for name in CODE_FILES:
        if digest(Path(__file__).with_name(name))!=lock['code_files']['scripts/maxopt_bridge/'+name]:
            raise ValueError('loaded Bridge source differs from lock')
    refs=verify_references({k:lock[k] for k in ('bindings','matrix','inputs','independent_review')})
    docs={name:read(bound(ref)) for name,ref in lock['bindings'].items()}
    review=read(bound(lock['independent_review']))
    if (review.get('schema')!='maxopt-bridge-experiment-lock-review-v1' or review.get('decision')!='PASS' or
            review.get('scope')!='S09_FORMAL_BRIDGE' or not review.get('reviewer') or review['reviewer']==lock['preparer'] or
            review.get('reviewed_lock_payload_sha256')!=hash_object({k:v for k,v in lock.items() if k!='independent_review'})):
        raise ValueError('independent review does not bind exact EXPERIMENT_LOCK')
    auth,env,identity=(docs[k] for k in ('authorization','environment_acceptance','environment_identity'))
    if (auth.get('schema')!='maxopt-bridge-current-authorization-v1' or
            auth.get('source_head')!=read(root/'SOURCE_VERSION.json').get('head') or
            auth.get('development_environment_acceptance')!=lock['bindings']['environment_acceptance'] or
            env.get('environment_gate')!='ACCEPTED_PASS_DEVELOPMENT_ENVIRONMENT_ONLY' or
            lock['gpu_uuids']!=auth.get('accepted_current_gpu_uuids') or
            lock['gpu_uuids']!=[g['uuid'] for g in identity['gpu_identity_expected']] or
            len(set(lock['gpu_uuids']))!=2):
        raise ValueError('current authorization/accepted environment/UUIDs differ')
    matrix=validate_matrix(read(bound(lock['matrix'])))
    route,policy=docs['route_freeze'],docs['policy_freeze']
    for phase,doc in (('S06',route),('S07',policy)):
        if (doc.get('schema')!='maxopt-bridge-pre-generation-stage-freeze-v1' or doc.get('phase')!=phase or
                doc.get('decision')!='ACCEPTED' or not doc.get('frozen_artifacts') or
                doc.get('authorization_sha256')!=lock['bindings']['authorization']['sha256']):
            raise ValueError('S06/S07 actual stage freeze required')
    if (route.get('planned_policy_ids')!=matrix['planned_policy_ids'] or
            policy.get('planned_policy_ids')!=matrix['planned_policy_ids'] or
            policy.get('effective_policy_ids')!=matrix['effective_policy_ids'] or
            policy.get('route_freeze_sha256')!=lock['bindings']['route_freeze']['sha256'] or
            route.get('route')!=matrix['planned_route'] or policy.get('effective_route')!=matrix['effective_route']):
        raise ValueError('planned/effective route differs from actual S06/S07 freeze')
    if matrix['effective_policy_ids']!=list(F4_IDS):
        if not policy.get('deployment_registry'):
            raise ValueError('raw PPO physical factory not independently registered; no fallback')
        from .ppo_deployment import load_pair
        deployment=load_pair(lock['bindings']['policy_freeze'])
        if set(deployment.policies)!=set(matrix['effective_policy_ids'][4:]):
            raise ValueError('physical raw PPO matrix differs from registered pair')
        for relative,ref in deployment.registry['code_files'].items():
            if lock['code_files'].get(relative)!=ref['sha256']:
                raise ValueError('raw PPO runtime code not bound by engineering review/lock')
        config=deployment.registry['common_config']
        if lock['code_files'].get('scripts/maxopt_bridge/ppo_common.json')!=config['sha256']:
            raise ValueError('raw PPO shared configuration missing from runtime code closure')
        verify_references(list(deployment.refs.values()),refs)
    else:
        disposition='SYMMETRIC_PPO_EXIT' if len(matrix['planned_policy_ids'])==6 else 'LAWFUL_SKIP_F4_ONLY'
        if (policy.get('disposition')!=disposition or policy.get('checkpoints')!=[] or not policy.get('skip_reason') or
                (disposition=='SYMMETRIC_PPO_EXIT' and policy.get('not_run_policy_ids')!=matrix['planned_policy_ids'][4:])):
            raise ValueError('S07 lawful skip or symmetric exit evidence missing')
    manifest=docs['final_input_manifest'];accept=docs['final_input_acceptance'];pred=docs['prediction_lock']
    if (manifest.get('schema')!='maxopt-bridge-final-input-manifest-v1' or
            manifest.get('status')!='GENERATED_PENDING_INDEPENDENT_AUDIT' or
            accept.get('decision')!='PASS' or accept.get('scope')!='FINAL_INPUT_AUDIT' or
            accept.get('manifest_sha256')!=lock['bindings']['final_input_manifest']['sha256'] or
            pred.get('schema')!='maxopt-bridge-prediction-lock-v1' or pred.get('decision')!='FROZEN_BEFORE_PHYSICAL' or
            pred.get('final_input_manifest_sha256')!=lock['bindings']['final_input_manifest']['sha256'] or
            pred.get('matrix_sha256')!=lock['matrix']['sha256'] or
            pred.get('policy_freeze_sha256')!=lock['bindings']['policy_freeze']['sha256'] or not pred.get('frozen_artifacts')):
        raise ValueError('independent final-input audit and pre-physical predictions required')
    runtime_review=docs['runtime_acceptance']
    if (runtime_review.get('decision')!='PASS' or runtime_review.get('scope')!='S09_ENGINEERING_ONLY' or
            runtime_review.get('code_files')!=lock['code_files']):
        raise ValueError('runtime engineering review does not bind this code')
    base=Path(lock['bindings']['final_input_manifest']['path']).parent
    for relative,sha in manifest['files'].items():
        p=base/relative
        if not p.resolve().is_relative_to(base):raise ValueError('final input artifact escapes closure')
        ref=dict(path=str(p),sha256=sha);verify_references(ref,refs)
    if set(lock['inputs'])!=set(WINDOW_IDS):raise ValueError('exact two final input files required')
    inputs={}
    for window,ref in lock['inputs'].items():
        if ref!={'path':str(base/'inputs'/(window+'.jsonl')),'sha256':manifest['files'].get('inputs/'+window+'.jsonl')}:
            raise ValueError('input SHA or location differs from audited final manifest')
        rows=[json.loads(line) for line in bound(ref).read_text().splitlines()]
        validate_requests(rows,window);inputs[window]=rows
    resources=lock['resources']
    if (set(resources)!={'wall_seconds','minimum_free_bytes','output_bytes_soft_cap','identity_seconds','minimum_cpu_count','minimum_memory_bytes'} or
            any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in resources.values()) or
            not Path(lock['model']).is_absolute() or not Path(lock['python']).is_absolute()):
        raise ValueError('explicit positive resource bounds and runtime paths required')
    if any(Path(p).is_relative_to(output) for p in refs) or path.is_relative_to(output):
        raise ValueError('formal output cannot contain its immutable prerequisites')
    return lock,docs,matrix,inputs,refs


def _read_mem_available(proc_meminfo=Path('/proc/meminfo')):
    values={}
    for line in Path(proc_meminfo).read_text(encoding='utf-8').splitlines():
        if ':' not in line:continue
        key,value=line.split(':',1);values[key]=value.strip()
    raw=values.get('MemAvailable')
    if not raw:raise ValueError('MemAvailable missing from proc meminfo')
    parts=raw.split()
    if not parts or not parts[0].isdigit():raise ValueError('invalid MemAvailable')
    return int(parts[0])*1024


def _read_cgroup_v1_path(proc_cgroup,cgroup_root):
    for line in Path(proc_cgroup).read_text(encoding='utf-8').splitlines():
        fields=line.split(':',2)
        if len(fields)!=3:continue
        controllers,path=fields[1],fields[2]
        if 'memory' in controllers.split(','):
            root=Path(cgroup_root);base=root if root.name=='memory' else root/'memory'
            return (base/path.lstrip('/')).resolve()
    return None


def _read_cgroup_v1_memory_chain(proc_cgroup=Path('/proc/self/cgroup'),cgroup_root=Path('/sys/fs/cgroup')):
    leaf=_read_cgroup_v1_path(proc_cgroup,cgroup_root)
    if leaf is None:return None
    base=Path(cgroup_root) if Path(cgroup_root).name=='memory' else Path(cgroup_root)/'memory'
    chain=[];current=leaf
    try:relative=current.relative_to(base)
    except ValueError:raise ValueError('cgroup-v1 path escapes memory mount')
    for _ in range(len(relative.parts)+1):
        limit_file=current/'memory.limit_in_bytes';usage_file=current/'memory.usage_in_bytes'
        raw_limit=limit_file.read_text(encoding='utf-8').strip();raw_usage=usage_file.read_text(encoding='utf-8').strip()
        try:limit=int(raw_limit);usage=int(raw_usage)
        except ValueError as exc:raise ValueError('invalid cgroup-v1 memory reading') from exc
        if limit<0 or usage<0:raise ValueError('negative cgroup-v1 memory reading')
        unlimited=limit>=2**60;headroom=None if unlimited else limit-usage
        chain.append(dict(path=str(current),raw_limit=raw_limit,raw_usage=raw_usage,limit_bytes=limit,usage_bytes=usage,finite=not unlimited,headroom_bytes=headroom))
        if current==base:break
        current=current.parent
    else:raise ValueError('cgroup-v1 ancestor traversal incomplete')
    return dict(version='v1',path=str(leaf),chain=chain)


def _read_cgroup_v2_memory(cgroup_root=Path('/sys/fs/cgroup'),proc_cgroup=Path('/proc/self/cgroup')):
    leaf=None
    for line in Path(proc_cgroup).read_text(encoding='utf-8').splitlines():
        fields=line.split(':',2)
        if len(fields)==3 and fields[1]=='':leaf=(Path(cgroup_root)/fields[2].lstrip('/')).resolve();break
    if leaf is None:leaf=Path(cgroup_root).resolve()
    limit_file=leaf/'memory.max';current_file=leaf/'memory.current'
    if not limit_file.is_file() or not current_file.is_file():return None
    raw_limit=limit_file.read_text(encoding='utf-8').strip();raw_current=current_file.read_text(encoding='utf-8').strip()
    try:current=int(raw_current)
    except ValueError as exc:raise ValueError('invalid cgroup-v2 memory.current') from exc
    if current<0:raise ValueError('negative cgroup-v2 memory.current')
    unlimited=raw_limit=='max'
    if not unlimited:
        try:limit=int(raw_limit)
        except ValueError as exc:raise ValueError('invalid cgroup-v2 memory.max') from exc
        if limit<0:raise ValueError('negative cgroup-v2 memory.max')
        headroom=limit-current
    else:limit=None;headroom=None
    return dict(version='v2',path=str(leaf),chain=[dict(path=str(leaf),raw_limit=raw_limit,raw_usage=raw_current,limit_bytes=limit,usage_bytes=current,finite=not unlimited,headroom_bytes=headroom)])


def _discover_memory(*,proc_meminfo,cgroup_root,proc_cgroup):
    mem_lines=Path(proc_meminfo).read_text(encoding='utf-8').splitlines()
    host_raw=next((line.split(':',1)[1].strip() for line in mem_lines if line.startswith('MemAvailable:')),None)
    host=_read_mem_available(proc_meminfo)
    cgroup=_read_cgroup_v1_memory_chain(proc_cgroup,cgroup_root)
    if cgroup is None:cgroup=_read_cgroup_v2_memory(cgroup_root,proc_cgroup)
    finite=[x['headroom_bytes'] for x in (cgroup or {}).get('chain',[]) if x.get('finite')]
    effective=min([host,*finite]) if finite else host
    return dict(host_mem_available_bytes=host,host_mem_available_raw=host_raw,cgroup=cgroup,effective_memory_headroom_bytes=effective)


def available_memory_bytes(*,proc_meminfo=Path('/proc/meminfo'),cgroup_root=Path('/sys/fs/cgroup'),proc_cgroup=Path('/proc/self/cgroup')):
    proc_meminfo,cgroup_root,proc_cgroup=map(Path,(proc_meminfo,cgroup_root,proc_cgroup))
    return _discover_memory(proc_meminfo=proc_meminfo,cgroup_root=cgroup_root,proc_cgroup=proc_cgroup)['effective_memory_headroom_bytes']


def _default_scheduler_reader(job_id):
    env=os.environ.copy();env['SLURM_TIME_FORMAT']='%s'
    proc=subprocess.run(['scontrol','-o','show','job',str(job_id)],capture_output=True,text=True,env=env,check=False)
    return dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)


def _read_scheduler_remaining_seconds(*,scheduler_reader=None,clock=time.time,environ=os.environ):
    job_id=environ.get('SLURM_JOB_ID')
    if not job_id:raise ValueError('SLURM_JOB_ID missing')
    raw=(scheduler_reader or _default_scheduler_reader)(job_id)
    if isinstance(raw,str):raw=dict(returncode=0,stdout=raw,stderr='')
    if raw.get('returncode',0)!=0:raise ValueError('scheduler allocation query failed')
    fields={}
    for token in str(raw.get('stdout','')).split():
        if '=' in token:
            key,value=token.split('=',1);fields[key]=value.strip('"')
    end=fields.get('EndTime')
    if end in (None,'','Unknown','N/A'):raise ValueError('scheduler EndTime missing')
    try:end_epoch=float(end)
    except ValueError as exc:raise ValueError('scheduler EndTime is not epoch seconds') from exc
    remaining=end_epoch-float(clock())
    if not math.isfinite(remaining):raise ValueError('scheduler remaining time is not finite')
    return dict(job_id=str(job_id),end_epoch=end_epoch,remaining_seconds=max(0.0,remaining),raw=raw,fields=fields)


def discover_resource_state(lock,output,*,scheduler_reader=None,cgroup_root=Path('/sys/fs/cgroup'),proc_cgroup=Path('/proc/self/cgroup'),proc_meminfo=Path('/proc/meminfo'),clock=time.time):
    cgroup_root,proc_cgroup,proc_meminfo=map(Path,(cgroup_root,proc_cgroup,proc_meminfo))
    target=Path(output)
    while not target.exists():
        parent=target.parent
        if parent==target:break
        target=parent
    memory=_discover_memory(proc_meminfo=proc_meminfo,cgroup_root=cgroup_root,proc_cgroup=proc_cgroup)
    scheduler=_read_scheduler_remaining_seconds(scheduler_reader=scheduler_reader,clock=clock)
    return dict(schema='maxopt-bridge-resource-admission-v1',timestamp_epoch=float(clock()),timestamp=datetime.now(timezone.utc).isoformat(),affinity_cpu_count=len(os.sched_getaffinity(0)),**memory,free_output_bytes=shutil.disk_usage(target).free,scheduler=scheduler,required_ports=list(lock['required_ports']))


def _next_evidence_path(output,evidence_path=None):
    if evidence_path is not None:return Path(evidence_path)
    output=Path(output);parent=output if output.exists() else output.parent;parent.mkdir(parents=True,exist_ok=True)
    index=1
    while (parent/f'resource_admission_attempt_{index:02d}.json').exists():index+=1
    return parent/f'resource_admission_attempt_{index:02d}.json'


def admit_resources(lock,output,state,*,required_remaining_seconds=None):
    budget=lock['resources'];reasons=[]
    if state.get('effective_memory_headroom_bytes',-1)<0:reasons.append('negative memory headroom')
    if state.get('affinity_cpu_count',0)<budget['minimum_cpu_count']:reasons.append('insufficient CPU')
    if state.get('effective_memory_headroom_bytes',0)<budget['minimum_memory_bytes']:reasons.append('insufficient memory')
    if state.get('free_output_bytes',0)<budget['minimum_free_bytes']:reasons.append('insufficient output disk reserve')
    if required_remaining_seconds is not None:
        if not math.isfinite(required_remaining_seconds) or required_remaining_seconds<0:reasons.append('invalid applicable remaining registered budget')
        elif state['scheduler']['remaining_seconds']<required_remaining_seconds:reasons.append('scheduler allocation shorter than applicable remaining registered budget')
    try:check_ports(lock['required_ports'])
    except ValueError as exc:reasons.append(str(exc))
    state['required_remaining_seconds']=required_remaining_seconds;state['decision']='REJECT' if reasons else 'ACCEPT';state['rejection_reasons']=reasons
    if reasons:raise ValueError('; '.join(reasons))
    return state


def check_resources(lock,output,*,required_remaining_seconds=None,evidence_path=None,scheduler_reader=None,cgroup_root=Path('/sys/fs/cgroup'),proc_cgroup=Path('/proc/self/cgroup'),proc_meminfo=Path('/proc/meminfo'),clock=time.time):
    cgroup_root,proc_cgroup,proc_meminfo=map(Path,(cgroup_root,proc_cgroup,proc_meminfo))
    evidence=_next_evidence_path(output,evidence_path)
    state=dict(schema='maxopt-bridge-resource-admission-v1',timestamp_epoch=float(clock()),timestamp=datetime.now(timezone.utc).isoformat(),decision='REJECT',rejection_reasons=[])
    try:
        if platform.system()!='Linux' or Path(sys.executable).resolve()!=Path(lock['python']).resolve():raise ValueError('accepted physical backend OS/interpreter differs')
        if required_remaining_seconds is None:
            target=Path(output)
            while not target.exists():target=target.parent
            budget=lock['resources'];cpu=len(os.sched_getaffinity(0))
            memory=available_memory_bytes() if (proc_meminfo==Path('/proc/meminfo') and cgroup_root==Path('/sys/fs/cgroup') and proc_cgroup==Path('/proc/self/cgroup')) else available_memory_bytes(proc_meminfo=proc_meminfo,cgroup_root=cgroup_root,proc_cgroup=proc_cgroup)
            free=shutil.disk_usage(target).free
            if cpu<budget['minimum_cpu_count'] or memory<budget['minimum_memory_bytes'] or free<budget['minimum_free_bytes']:raise ValueError('insufficient allocated CPU/memory/output disk reserve')
            check_ports(lock['required_ports']);state.update(affinity_cpu_count=cpu,effective_memory_headroom_bytes=memory,free_output_bytes=free,required_ports=list(lock['required_ports']),decision='ACCEPT',rejection_reasons=[])
        else:
            target=Path(output)
            while not target.exists():
                parent=target.parent
                if parent==target:break
                target=parent
            state.update(_discover_memory(proc_meminfo=proc_meminfo,cgroup_root=cgroup_root,proc_cgroup=proc_cgroup))
            state['affinity_cpu_count']=len(os.sched_getaffinity(0));state['free_output_bytes']=shutil.disk_usage(target).free
            state['scheduler']=_read_scheduler_remaining_seconds(scheduler_reader=scheduler_reader,clock=clock)
            state['required_ports']=list(lock['required_ports']);state=admit_resources(lock,output,state,required_remaining_seconds=required_remaining_seconds)
    except BaseException as exc:
        state['decision']='REJECT';state['rejection_reasons']=list(state.get('rejection_reasons',[]))+[str(exc)];state['error_type']=type(exc).__name__
        evidence.parent.mkdir(parents=True,exist_ok=True);save_new(evidence,state);raise
    evidence.parent.mkdir(parents=True,exist_ok=True);save_new(evidence,state)
    return state


@contextlib.contextmanager
def exclusive_campaign(output):
    """Kernel lock releases on crash; concurrent resume must never launch twice."""
    import fcntl
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/('.'+output.name+'.controller.lock')).open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('another controller owns this campaign') from exc
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def build_handoff(campaign, lock_path, lock, refs, root, identity_after, *, pause=time.sleep):
    summary=campaign.summary()
    if not summary['complete'] or identity_after.get('errors'):
        raise ValueError('only a completed/registered-stop and clean campaign may seal')
    campaign._verify_attempts();root=Path(root);handoff=campaign.out/'cpu_handoff';handoff.mkdir(exist_ok=False)
    snapshot=handoff/'source_snapshot';snapshot.mkdir();files=read(root/'SOURCE_DELIVERY_MANIFEST.json')['files']
    wanted={r['path']:r['sha256'] for r in files};wanted.update(lock['code_files'])
    wanted.update({name:digest(root/name) for name in ('SOURCE_VERSION.json','COMMIT_PROVENANCE.json','SOURCE_DELIVERY_MANIFEST.json')})
    for relative,sha in wanted.items():
        source=bound(dict(path=str(root/relative),sha256=sha));destination=snapshot/relative
        if not source.resolve().is_relative_to(root.resolve()) or not destination.resolve().is_relative_to(snapshot.resolve()):
            raise ValueError('source snapshot path escapes its closure')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        if digest(destination)!=sha:raise ValueError('self-contained source copy differs')
    references=dict(refs);references[str(Path(lock_path).resolve())]=dict(path=str(Path(lock_path).resolve()),sha256=campaign.lock_sha256)
    for event in campaign.events:
        if event.get('receipt'):verify_references(event['receipt'],references)
    index=[];dependencies=handoff/'dependencies';dependencies.mkdir()
    for i,(original,ref) in enumerate(sorted(references.items())):
        source=bound(ref);destination=dependencies/(format(i,'04d')+'_'+source.name)
        shutil.copyfile(source,destination)
        if digest(destination)!=ref['sha256']:raise ValueError('self-contained dependency copy differs')
        index.append(dict(original=original,path=destination.relative_to(campaign.out).as_posix(),sha256=ref['sha256']))
    save_new(handoff/'REFERENCE_INDEX.json',index)
    save_new(campaign.out/'IDENTITY_AFTER.json',identity_after)
    save_new(campaign.out/'CAMPAIGN_RESULT.json',summary)
    (handoff/'RECOMPUTE_ZH.md').write_text('CPU交接：先调用scripts.maxopt_bridge.campaign.verify_seal(output, "CAMPAIGN_MANIFEST.json")逐文件核SHA。原绝对路径只用于来源追踪，REFERENCE_INDEX.json映射到交接内精确副本；不要重写锁。以CAMPAIGN_RESULT.json保留全部计划格/attempt/NOT_RUN状态，用各attempt的expected_requests.json与request_terminals.json调用metrics.evaluate_requests复算严格P1+。controller_events/request_events、逐代worker stdout/stderr/ownership/cleanup原样保留；按run_plan内相同价格与既有metrics.cost_breakdown做CPU账本核算，并区分setup/window/cleanup。此交接不包含Qwen模型权重；raw PPO路线在dependencies保存登记的小型PPO权重、sidecar、registry、配置和完整来源引用。交接不启动运行、不产生S10结论。\n',encoding='utf-8')
    if (campaign.remaining_seconds()<=0 or
            sum(p.stat().st_size for p in campaign.out.rglob('*') if p.is_file())>lock['resources']['output_bytes_soft_cap']):
        raise ValueError('final raw/handoff resource budget exceeded; retain unsealed evidence')
    seal(campaign.out,'CAMPAIGN_MANIFEST.json',pause=pause)
    return summary


def execute_campaign(lock_path, expected_sha256, output, *, resume=False, retry_receipt=None, technical_stop_receipt=None):
    checked=verify_lock(lock_path,expected_sha256,output)
    lock=checked[0];required_remaining=lock['resources']['wall_seconds']
    if resume:
        journal=Path(output).resolve()/'campaign_events.jsonl'
        if not journal.is_file():raise ValueError('resume requires an existing campaign journal')
        first=json.loads(journal.read_text(encoding='utf-8').splitlines()[0])
        if first.get('kind')!='campaign_start' or first.get('wall_seconds')!=lock['resources']['wall_seconds']:
            raise ValueError('resume campaign journal does not bind frozen wall budget')
        required_remaining=max(0.0,float(lock['resources']['wall_seconds'])-(time.time()-float(first['started_at'])))
    check_resources(lock,output,required_remaining_seconds=required_remaining)
    with exclusive_campaign(output):
        return _execute_campaign(checked,lock_path,expected_sha256,output,resume=resume,
                                 retry_receipt=retry_receipt,technical_stop_receipt=technical_stop_receipt)


def _execute_campaign(checked, lock_path, expected_sha256, output, *, resume, retry_receipt, technical_stop_receipt):
    lock,docs,matrix,inputs,refs=checked
    output=Path(output).resolve()
    if (output/'CAMPAIGN_MANIFEST.json').exists():
        if not resume:raise ValueError('sealed campaign already exists')
        verify_seal(output,'CAMPAIGN_MANIFEST.json');return read(output/'CAMPAIGN_RESULT.json')
    deployment=None
    if matrix['effective_policy_ids']!=list(F4_IDS):
        from .ppo_deployment import load_pair
        deployment=load_pair(lock['bindings']['policy_freeze'])
    runtime=load_runtime();context=f4.load_context();reference=docs['environment_identity']
    campaign=Campaign(output,matrix,expected_sha256,wall_seconds=lock['resources']['wall_seconds'],resume=resume)
    if not resume:
        shutil.copyfile(lock_path,output/'EXPERIMENT_LOCK.json')
    bound(dict(path=str(output/'EXPERIMENT_LOCK.json'),sha256=expected_sha256))
    campaign.recover_recorded_results()
    if retry_receipt:campaign.authorize_retry(*retry_receipt)
    if technical_stop_receipt:campaign.stop_technical(*technical_stop_receipt)
    def prelaunch(remaining_seconds):
        verify_lock(lock_path,expected_sha256,output)
        if deployment is not None:deployment.verify_current()
        resource=check_resources(lock,output,required_remaining_seconds=remaining_seconds)
        observed=capture_current_identity(lock['model'],runtime,reference,lock['resources']['identity_seconds'])
        return dict(identity=observed,resources=resource,lock_sha256=expected_sha256)
    def runner(cell,attempt,folder,remaining):
        extra={'deployment':deployment} if deployment is not None else {}
        plan=make_plan(cell,context,reference,model=lock['model'],python=lock['python'],**extra)
        return run_cell(runtime,context,plan,inputs[cell['window_id']],reference,folder,attempt=attempt,
            lock_sha256=expected_sha256,remaining_seconds=remaining,output_limit=lock['resources']['output_bytes_soft_cap'],
            campaign_out=output,prelaunch=prelaunch,**extra)
    limits=read(RUNTIME_TEMPLATE)['limits']
    reserve=limits['setup_seconds']+2100+limits['cleanup_seconds']+2*lock['resources']['identity_seconds']
    summary=campaign.run(runner,reserve_seconds=reserve)
    if summary['complete']:
        verify_lock(lock_path,expected_sha256,output)
        if deployment is not None:deployment.verify_current()
        after=capture_current_identity(lock['model'],runtime,reference,lock['resources']['identity_seconds'])
        if after['errors']:
            folder=output/'finalization_attempts'/('attempt_'+format(len(list((output/'finalization_attempts').glob('*')))+1,'02d'))
            folder.mkdir(parents=True,exist_ok=False);save_new(folder/'IDENTITY_AFTER.json',after)
            raise ValueError('final identity/release audit failed; campaign remains unsealed')
        summary=build_handoff(campaign,lock_path,lock,refs,ROOT,after)
    return summary
