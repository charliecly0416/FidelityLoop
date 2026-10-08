"""Shared causal raw-PPO inference; original training/physics stay unchanged."""
import copy
import math
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .ppo_capacity import CapacityNetwork
from .ppo_env import (CAPACITY_MASK, OFFLINE_MASK, STATES, _VisibleSimulator,
                      encode_observation, filter_target, public_observation)
from .metrics import cost_breakdown, evaluate_requests
from . import ppo_capacity, ppo_training
from .final_inputs import bound,digest,read,registration
from .formal_runtime import verify_references
from .preflight_runtime import verify_sources
from .protocol import ROOT,F4_IDS
from scripts.maxopt_v3.audit import audit as audit_events

CLOCK_SEMANTICS='single_actual_decision_elapsed_v1'
REQUIRED_CODE=('ppo_deployment.py','ppo_env.py','ppo_capacity.py','ppo_training.py')
SYNTHETIC='SYNTHETIC_ENGINEERING_ONLY'


def validate_public(public):
    if set(public)!={'time_seconds','target','since_target_change_seconds','queues','devices','api_jobs'}:
        raise ValueError('observation must contain only the public whitelist')
    def finite(value,nonnegative=False):
        if type(value) not in (int,float) or not math.isfinite(value) or (nonnegative and value<0):
            raise ValueError('nonfinite or negative public age/time')
    def jobs(items):
        for job in items:
            if set(job)!={'age_seconds','remaining_deadline_seconds'}:
                raise ValueError('job exposes fields outside the causal whitelist')
            finite(job['age_seconds'],True);finite(job['remaining_deadline_seconds'])
    finite(public['time_seconds'],True);finite(public['since_target_change_seconds'],True)
    if type(public['target']) is not int or public['target'] not in (0,1,2):
        raise ValueError('invalid observable target')
    if set(public['queues'])!={'online','offline'} or len(public['devices'])!=2:
        raise ValueError('expected both queues and two devices')
    for items in public['queues'].values():jobs(items)
    for device in public['devices']:
        if set(device)!={'state','state_age_seconds','jobs'} or device['state'] not in STATES:
            raise ValueError('invalid public lifecycle state')
        finite(device['state_age_seconds'],True);jobs(device['jobs'])
    jobs(public['api_jobs'])
    return public


class DecisionPolicy:
    """A fresh target/filter state per episode; CPU forward only, no sampling."""
    def __init__(self,network):
        if type(network) is not CapacityNetwork or any(p.device.type!='cpu' for p in network.parameters()):
            raise ValueError('exact original CPU CapacityNetwork required')
        self.network=network.eval()
        self.state=dict(target=2,last_change_at=-60)
        self.last_time=None

    def decide(self,public):
        validate_public(public);now=public['time_seconds']
        if (public['target']!=self.state['target'] or
                public['since_target_change_seconds']!=now-self.state['last_change_at'] or
                (self.last_time is not None and now<=self.last_time)):
            raise ValueError('target provenance or monotonic decision time differs')
        vector=encode_observation(public)
        with torch.inference_mode():
            logits,offline,value=self.network(torch.as_tensor(vector)[None],
                torch.tensor([CAPACITY_MASK]),torch.tensor([OFFLINE_MASK]))
            if any(not torch.isfinite(t).all() for t in (logits,offline,value)):
                raise ValueError('nonfinite deterministic forward output')
            proposal=int(logits.argmax(-1).item());offline_action=int(offline.argmax(-1).item())
            logprob=(torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([proposal]))+
                     torch.distributions.Categorical(logits=offline).log_prob(torch.tensor([offline_action])))
        if offline_action!=0:raise ValueError('raw deployment must keep sole offline action zero')
        target,changed=filter_target(proposal,self.state['target'],self.state['last_change_at'],now)
        record=dict(public=copy.deepcopy(public),observation=vector.tolist(),capacity_mask=list(CAPACITY_MASK),
            offline_mask=list(OFFLINE_MASK),capacity_logits=logits[0].tolist(),offline_logits=offline[0].tolist(),
            proposal=proposal,offline_action=offline_action,behavior_log_prob=float(logprob.item()),
            value=float(value.item()),filtered_target=target,executed_target=target,actuator_target=target,last_change_at=changed)
        self.state.update(target=target,last_change_at=changed);self.last_time=now
        return record


def replay(network,rows,contract,model,*,allow_locked_test=False):
    """Original S04 visible simulator, same callback boundary and settlement."""
    policy=DecisionPolicy(network);decisions=[]
    sim=_VisibleSimulator(rows,contract,model,'all2',allow_locked_test=allow_locked_test)
    def decide(observation):
        public=public_observation(sim,policy.state['target'],policy.state['last_change_at'])
        record=policy.decide(public);record['scheduled_tick']=observation['time']
        decisions.append(record);return record['executed_target']
    sim.policy=SimpleNamespace(decide=decide)
    result=sim.run();metrics=evaluate_requests(rows,result['requests'],horizon=sim.end)
    costs=cost_breakdown(rows,metrics,occupied_seconds=sim.occupied,starts=sim.starts,shutdowns=sim.shutdowns,
        api_ids=sim.api_ids,prices=contract['accounting'])
    for key,value in result['summary']['costs'].items():
        if not math.isclose(value,costs[key],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('original simulator and deployment ledger differ')
    event_audit=audit_events(rows,result['events'],result['summary'],contract,model,
        policy=None,requests=result['requests'],scale=False,reservation=False)
    executed=[(e['time'],e['target']) for e in result['events'] if e['kind']=='decision']
    if executed!=[(r['scheduled_tick'],r['executed_target']) for r in decisions]:
        raise ValueError('raw network decisions differ from original actuator event targets')
    return dict(result=result,event_audit=event_audit,request_metrics=metrics,costs=costs,decisions=decisions)


def physical_public(controller,tick,*,clock=time.monotonic_ns):
    """One synchronous actual-time snapshot, never scheduled-time age clamping."""
    if type(tick) is not int or not 0<=tick<controller.plan['horizon_seconds']:
        raise ValueError('invalid registered policy tick')
    version=len(controller.events);observed_ns=clock()
    if controller.origin_ns is None or type(observed_ns) is not int:
        raise ValueError('missing physical observation origin/clock')
    elapsed=(observed_ns-controller.origin_ns)/1e9
    if not tick<=elapsed<tick+1 or controller.window_closed or observed_ns>=controller.cutoff_ns:
        raise ValueError('early, missed or closed physical policy tick')
    def job(rid):
        if rid not in controller.released:raise ValueError('unreleased request entered causal state')
        row=controller.by_id[rid]
        if row['arrival_s']>tick:raise ValueError('future request entered causal state')
        return dict(age_seconds=elapsed-row['arrival_s'],remaining_deadline_seconds=row['deadline_s']-elapsed)
    state=controller.adapter.state
    public=dict(time_seconds=elapsed,target=state['target'],since_target_change_seconds=elapsed-state['last_change_at'],
        queues={kind:[job(rid) for rid in controller.queues[kind]] for kind in ('online','offline')},
        devices=[dict(state=device['state'],state_age_seconds=elapsed-controller._device_since[g],
                      jobs=[job(rid) for rid in sorted(device['jobs'])])
                 for g,device in enumerate(controller.actuator.devices)],
        api_jobs=[job(rid) for rid in controller.api_active])
    if len(controller.events)!=version:raise ValueError('controller changed during synchronous snapshot')
    validate_public(public)
    return public,dict(scheduled_tick=tick,observed_monotonic_ns=observed_ns,
        observed_elapsed_seconds=elapsed,clock_semantics=CLOCK_SEMANTICS)


class PhysicalAdapter:
    """Original FormalV2 raw admission path; not a GuardAdapter subtype."""
    def __init__(self,network,controller,clock):
        self.policy=DecisionPolicy(network);self.controller=controller;self.clock=clock
        self.owner_thread=threading.get_ident();self.last_tick=-1

    @property
    def state(self):return self.policy.state

    def decide(self,unused_simple_observation):
        tick=self.controller._current_tick
        if threading.get_ident()!=self.owner_thread or tick!=self.last_tick+1:
            raise ValueError('wrong callback thread or repeated/missed policy tick')
        public,timing=physical_public(self.controller,tick,clock=self.clock)
        record=self.policy.decide(public);record.update(timing,adapter='raw_ppo_capacity')
        self.last_tick=tick
        return record['executed_target'],record


class Deployment:
    """Verified pair plus their immutable leaf refs; each run gets fresh state."""
    def __init__(self,freeze_ref,registry_ref,registry,policies,refs,synthetic):
        self.freeze_ref,self.registry_ref=freeze_ref,registry_ref
        self.registry,self.policies,self.refs=registry,policies,refs
        self.synthetic=synthetic
        self._loaded_identity_sha256=self._identity_sha256()

    def _identity_sha256(self):
        return ppo_training.object_sha(dict(freeze_ref=self.freeze_ref,registry_ref=self.registry_ref,
            registry=self.registry,refs=self.refs,synthetic=self.synthetic,
            policies={name:{key:value for key,value in policy.items() if key!='network'}
                      for name,policy in self.policies.items()}))

    def _verify_identity(self):
        if self._identity_sha256()!=self._loaded_identity_sha256:
            raise ValueError('deployment identity changed after loading')

    def verify_current(self):
        self._verify_identity()
        for ref in self.refs.values():bound(ref)
        _sources(self.registry)
        for policy in self.policies.values():
            if ppo_training.tensor_digest(policy['network'].state_dict())!=policy['tensor_sha256']:
                raise ValueError('deployment weights changed in memory')
        return self.refs

    def identity(self,policy_id):
        self._verify_identity()
        if policy_id not in self.policies:raise ValueError('unregistered raw PPO policy; no fallback')
        item=self.policies[policy_id]
        return dict(policy_id=policy_id,training_model=item['training_model'],
            policy_freeze_sha256=self.freeze_ref['sha256'],registry_sha256=self.registry_ref['sha256'],
            checkpoint_sha256=item['checkpoint_sha256'],tensor_sha256=item['tensor_sha256'],clock_semantics=CLOCK_SEMANTICS)


def _sources(registry):
    if registry.get('source_root')!=str(ROOT.resolve()):raise ValueError('deployment source workspace differs')
    registration(ROOT);verify_sources()
    for name,key in [('SOURCE_VERSION.json','source_version_sha256'),('COMMIT_PROVENANCE.json','provenance_sha256'),
                     ('SOURCE_DELIVERY_MANIFEST.json','source_delivery_manifest_sha256')]:
        if digest(ROOT/name)!=registry.get(key):raise ValueError('deployment original source identity changed')
    required={'scripts/maxopt_bridge/'+name for name in REQUIRED_CODE}
    if not required<=set(registry.get('code_files',{})):raise ValueError('deployment code closure incomplete')
    for relative,ref in registry['code_files'].items():
        if ref['path']!=str(ROOT/relative) or not (ROOT/relative).resolve().is_relative_to(ROOT):
            raise ValueError('deployment code path differs from registered workspace')
        bound(ref)
    for name in REQUIRED_CODE:
        if digest(Path(__file__).with_name(name))!=registry['code_files']['scripts/maxopt_bridge/'+name]['sha256']:
            raise ValueError('loaded deployment code differs from registry')
    config=read(bound(registry['common_config']))
    if (registry['common_config']['path']!=str(Path(__file__).with_name('ppo_common.json')) or
            config!=ppo_capacity.COMMON_CONFIG or config!=ppo_training.COMMON_CONFIG):
        raise ValueError('loaded common configuration differs')
    return ppo_training.verify_s04(bound(registry['s04_acceptance']))


def load_pair(freeze_ref,*,allow_synthetic=False):
    """Only the accepted S07 pair; no pilot, initialization, diagnostic or trainer."""
    freeze=read(bound(freeze_ref));synthetic=freeze.get('execution_evidence')==SYNTHETIC
    if synthetic and not allow_synthetic:raise ValueError('synthetic fixture is not an accepted production policy')
    if (freeze.get('schema')!='maxopt-bridge-pre-generation-stage-freeze-v1' or freeze.get('phase')!='S07' or
            freeze.get('decision')!=('SYNTHETIC_FIXTURE_ONLY' if synthetic else 'ACCEPTED') or
            freeze.get('disposition')!='FROZEN_RAW_PPO' or freeze.get('route')!='F4_plus_two_raw_PPO' or
            freeze.get('effective_route')!=freeze.get('route') or not freeze.get('frozen_artifacts')):
        raise ValueError('accepted S07 raw PPO pair freeze required')
    planned=freeze.get('planned_policy_ids',[]);checkpoints=freeze.get('checkpoints',[])
    if (len(planned)!=6 or any(type(p) is not str or not re.fullmatch(r'[A-Za-z0-9_]+',p) for p in planned) or
            len(set(planned))!=6 or planned[:4]!=list(F4_IDS) or
            freeze.get('effective_policy_ids')!=planned or len(checkpoints)!=2 or
            {c.get('training_model') for c in checkpoints}!={'C','C30'} or
            {c.get('policy_id') for c in checkpoints}!=set(planned[4:])):
        raise ValueError('S07 policy IDs must map bijectively to the C/C30 pair')
    registry_ref=freeze.get('deployment_registry')
    if registry_ref not in freeze['frozen_artifacts']:raise ValueError('deployment registry not frozen by S07')
    registry=read(bound(registry_ref))
    if (registry.get('schema')!='maxopt-bridge-ppo-deployment-registry-v1' or
            registry.get('clock_semantics')!=CLOCK_SEMANTICS or registry.get('scope')!=(SYNTHETIC if synthetic else 'S07_DEPLOYMENT_IDENTITY')):
        raise ValueError('unregistered deployment registry or clock semantics')
    refs=verify_references(freeze_ref);bindings=_sources(registry)
    acceptance_path=bound(registry['s04_acceptance']);acceptance=read(acceptance_path)
    delivery_path=acceptance_path.parent/'executor/s04/DELIVERY_SHA256.json'
    delivery=read(delivery_path)
    evidence=[delivery_path,acceptance_path.parent/'executor/r0/R0_RECEIPT.json',Path(acceptance['review_path'])]
    evidence.extend(ROOT/name for name in delivery['owned_files'])
    for path in evidence:verify_references(dict(path=str(path.resolve()),sha256=digest(path)),refs)
    context,_,digests,_=ppo_training.load_inputs();selection=ppo_training.selection_contract(context,digests)
    lock_ref=registry['training_lock'];lock=read(bound(lock_ref))
    if synthetic and lock.get('execution_evidence')!=SYNTHETIC:raise ValueError('synthetic training lock label required')
    ppo_training.validate_run_lock(lock,'formal',bindings,selection,digests)
    authorization_path=bound(registry['formal_authorization'])
    ppo_training.verify_formal(authorization_path,lock)
    authorization=read(authorization_path)
    if synthetic and authorization.get('execution_evidence')!=SYNTHETIC:
        raise ValueError('synthetic formal authorization label required')
    verify_references(dict(path=authorization['s06_acceptance_path'],
                           sha256=authorization['s06_acceptance_sha256']),refs)
    selections=read(bound(registry['selection']))
    if (selections.get('selection_contract')!=selection or
            selections.get('selection_contract_sha256')!=ppo_training.object_sha(selection) or
            selections.get('lock_sha256')!=lock_ref['sha256'] or set(selections.get('groups',{}))!={'C','C30'} or
            selections.get('physical_arm_status')!='NOT_YET_AUTHORIZED'):
        raise ValueError('formal selection identity or paired eligibility differs')
    policies={}
    for item in checkpoints:
        group=selections['groups'][item['training_model']]
        ranked=ppo_training.rank_candidates(group['diagnostic'],selection);chosen=ranked['selected']
        if chosen is None or group.get('selected')!=chosen:
            raise ValueError('checkpoint is not the qualified common-selection winner')
        if any(c['training_environment']!=item['training_model'] for c in group['diagnostic']):
            raise ValueError('mixed training group in selection')
        path=bound(item['file']);receipt=read(bound(item['receipt']));metadata=receipt.get('metadata',{})
        if (str(path)!=chosen['path'] or digest(path)!=chosen['checkpoint_sha256'] or
                receipt.get('checkpoint_sha256')!=item['file']['sha256'] or
                any(chosen.get(k)!=v for k,v in metadata.items()) or
                bool(metadata.get('synthetic_fixture',False))!=synthetic or
                any(type(metadata.get(k)) is not int for k in ('seed','checkpoint_index','episode_completed','actual_steps')) or
                metadata.get('kind')!='training_checkpoint' or metadata.get('mode')!='formal' or
                metadata.get('training_environment')!=item['training_model'] or metadata.get('seed') not in range(101,106) or
                metadata.get('checkpoint_index') not in lock['checkpoint_episodes'] or
                metadata.get('episode_completed')!=metadata.get('checkpoint_index') or
                metadata.get('actual_steps')!=metadata['checkpoint_index']*ppo_training.COMMON_CONFIG['episode_seconds'] or
                metadata.get('evaluation_due') is not True or metadata.get('lock_sha256')!=lock_ref['sha256'] or
                metadata.get('selection_contract_sha256')!=ppo_training.object_sha(selection) or
                any(metadata.get(k)!=v for k,v in bindings.items())):
            raise ValueError('wrong group, pilot, initial or stale checkpoint identity')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('metadata')!=metadata or ppo_training.tensor_digest(payload['model_state_dict'])!=receipt.get('tensor_sha256'):
            raise ValueError('checkpoint payload/sidecar/tensor mismatch')
        with torch.random.fork_rng(devices=[]):network=CapacityNetwork()
        expected=network.state_dict();weights=payload['model_state_dict']
        if (set(weights)!=set(expected) or any(not torch.is_tensor(t) or t.shape!=expected[k].shape or
                t.dtype!=expected[k].dtype or not torch.isfinite(t).all() for k,t in weights.items())):
            raise ValueError('checkpoint tensor type/shape/nonfinite mismatch')
        network.load_state_dict(weights,strict=True);network.eval();network.requires_grad_(False)
        policies[item['policy_id']]=dict(network=network,training_model=item['training_model'],
            checkpoint_sha256=item['file']['sha256'],tensor_sha256=receipt['tensor_sha256'])
    deployment=Deployment(copy.deepcopy(freeze_ref),copy.deepcopy(registry_ref),registry,policies,refs,synthetic)
    deployment.verify_current();return deployment


def predict(deployment,policy_id,rows,contract,model,*,allow_locked_test=False):
    if type(deployment) is not Deployment:raise ValueError('verified deployment pair required')
    identity=deployment.identity(policy_id);deployment.verify_current()
    result=replay(deployment.policies[policy_id]['network'],rows,contract,model,allow_locked_test=allow_locked_test)
    deployment.verify_current();result['deployment_identity']=identity
    return result


def controller_class(runtime,deployment,policy_id,*,allow_synthetic=False):
    if type(deployment) is not Deployment:raise ValueError('verified deployment pair required')
    identity=deployment.identity(policy_id);deployment.verify_current()
    if deployment.synthetic and not allow_synthetic:raise ValueError('synthetic fixture cannot deploy physically')
    class RawPPOController(runtime.FormalV2Controller):
        def __init__(self,plan,out,*,backend_factory=None):
            deployment.verify_current()
            if (plan.get('schema')!='maxopt-bridge-raw-ppo-runtime-plan-v1' or
                    plan.get('policy_id')!=policy_id or plan.get('deployment_identity')!=identity or
                    plan.get('adapter_initial_target')!=2):
                raise ValueError('raw PPO plan identity or initial target differs')
            if deployment.synthetic and backend_factory is None:
                raise ValueError('synthetic controller requires an explicit fake backend')
            runtime._validate_runtime_plan(plan)
            if plan.get('formal') is True and (plan.get('window_seconds'),plan.get('drain_seconds'))!=(1800,300):
                raise ValueError('raw PPO formal timing differs')
            self._device_since=[0.0,0.0];self._current_tick=0;self._reserve=False
            self.adapter=PhysicalAdapter(deployment.policies[policy_id]['network'],self,lambda:runtime.time.monotonic_ns())
            runtime.gate.Controller.__init__(self,copy.deepcopy(plan),out,
                backend_factory=backend_factory or runtime.AsyncWorkerBackend,policy=self.adapter,formal=False)
    return RawPPOController
