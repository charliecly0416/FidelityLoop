"""Five frozen CPU inference arms; unchanged v4 causal observation/physical adapter."""
import copy
import torch
from . import formal_runtime as fr, ppo_deployment as pd
from .e2_runtime import checked, frozen_arms, require, verify_sources, MODE, MOCK
from scripts.maxopt_v2.workload import hash_object


class E2Deployment:
    def __init__(self, candidate):
        self.candidate = copy.deepcopy(candidate)
        self.binding_sha = hash_object(candidate)
        self.policies = {}
        for policy, arm in frozen_arms().items():
            ref = candidate['checkpoints'][policy]
            require(ref['sha256'] == arm['checkpoint_sha256'], 'checkpoint identity differs')
            payload = torch.load(checked(ref), map_location='cpu', weights_only=True)
            state = payload['model_state_dict']
            require(pd.ppo_training.tensor_digest(state) == arm['tensor_sha256'], 'checkpoint tensor SHA differs')
            with torch.random.fork_rng(devices=[]): network = pd.CapacityNetwork()
            network.load_state_dict(state, strict=True)
            network.eval(); network.requires_grad_(False)
            self.policies[policy] = dict(network=network, training_model=arm['training_environment'],
                checkpoint_sha256=arm['checkpoint_sha256'], tensor_sha256=arm['tensor_sha256'])
        self.verify_current()

    def verify_current(self):
        require(hash_object(self.candidate) == self.binding_sha, 'deployment binding changed')
        verify_sources(self.candidate['source_manifest'])
        require(set(self.policies) == set(frozen_arms()), 'loaded arm set changed')
        for policy, arm in frozen_arms().items():
            checked(self.candidate['checkpoints'][policy])
            item = self.policies[policy]
            require(item['training_model'] == arm['training_environment'] and
                item['checkpoint_sha256'] == arm['checkpoint_sha256'] and
                item['tensor_sha256'] == arm['tensor_sha256'] and
                pd.ppo_training.tensor_digest(item['network'].state_dict()) == arm['tensor_sha256'],
                'loaded tensor/arm identity changed')
        return self.candidate['checkpoints']

    def identity(self, policy):
        require(policy in self.policies, 'unregistered raw arm; no fallback')
        item = self.policies[policy]
        return dict(policy_id=policy, training_model=item['training_model'],
            checkpoint_sha256=item['checkpoint_sha256'], tensor_sha256=item['tensor_sha256'],
            e2_binding_sha256=self.binding_sha, clock_semantics=pd.CLOCK_SEMANTICS)


def make_plan(cell, context, identity, *, candidate, deployment, execution_evidence=MODE):
    require(cell in candidate['matrix']['cells'], 'cell absent from frozen E2 matrix')
    require(type(deployment) is E2Deployment, 'verified E2 deployment required')
    deployment.verify_current()
    raw = cell['policy_id'] != 'all2'
    # Only the common all2 configuration is inherited; raw identity added explicitly.
    base = dict(cell, policy_id='all2') if raw else cell
    plan = fr.make_plan(base, context, identity, model=candidate['model'], python=candidate['python'],
                        execution_evidence=execution_evidence)
    plan.update({k: cell[k] for k in ('run_id','policy_id','kind')})
    if raw:
        ident = deployment.identity(cell['policy_id'])
        plan.update(schema='maxopt-bridge-raw-ppo-runtime-plan-v1', deployment_identity=ident,
                    policy_spec=dict(name='raw_ppo_capacity', training_model=ident['training_model']))
    plan['e2_deployment_binding_sha256'] = deployment.binding_sha
    return plan


def controller_class(runtime, context, plan, identity, *, candidate, deployment):
    cell = next((c for c in candidate['matrix']['cells'] if c['run_id'] == plan['run_id']), None)
    require(cell is not None, 'unknown E2 run identity')
    expected = make_plan(cell, context, identity, candidate=candidate, deployment=deployment,
                         execution_evidence=plan['execution_evidence'])
    require(plan == expected and plan['execution_evidence'] in (MODE, MOCK), 'E2 plan drift')
    runtime._validate_runtime_plan(plan)
    if cell['policy_id'] == 'all2': return runtime.FormalV2Controller
    policy = cell['policy_id']
    class E2RawController(runtime.FormalV2Controller):
        def __init__(self, selected_plan, out, *, backend_factory=None):
            deployment.verify_current()
            require(selected_plan == expected, 'controller input drift')
            self._device_since=[0.0,0.0]; self._current_tick=0; self._reserve=False
            self.adapter = pd.PhysicalAdapter(deployment.policies[policy]['network'], self,
                                               lambda: runtime.time.monotonic_ns())
            runtime.gate.Controller.__init__(self, copy.deepcopy(selected_plan), out,
                backend_factory=backend_factory or runtime.AsyncWorkerBackend, policy=self.adapter, formal=False)
    return E2RawController


def run_cell(runtime, context, plan, rows, identity, out, *, candidate, deployment, **kwargs):
    cls = controller_class(runtime, context, plan, identity, candidate=candidate, deployment=deployment)
    return fr._run_cell(runtime, context, plan, rows, identity, out, controller_class=cls,
                        deployment=deployment, **kwargs)
