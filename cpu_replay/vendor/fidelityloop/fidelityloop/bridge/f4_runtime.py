"""Strict F4 registration above the byte-preserved E1 v3 runtime.

The runtime module is supplied by the verified vendor loader. No GPU is launched
by registering adapters/classes. All physical transitions remain inherited.
"""
import copy

from fidelityloop.bridge.f4 import validate_arm


def policy_spec(arm):
    arm = validate_arm(arm)
    return ({'base_policy': arm['policy']} if arm['scale'] else arm['policy'])


def make_adapter(runtime, arm, service_model, safety):
    arm = validate_arm(arm)
    if arm['scale']:
        return runtime.GuardAdapter(dict(policy_spec=policy_spec(arm),
                                         service_model=service_model, guard_safety=safety))
    return runtime.TargetAdapter(arm['policy'])


def controller_class(runtime, context):
    """Replace only the original constructor's one-ID adapter selection.

    These initialization fields mirror the frozen FormalV2Controller constructor;
    its observations, ticks, admission, backend, clock and cleanup are inherited.
    Supplying original GuardAdapter instances preserves the inherited isinstance
    dispatch branch for *both* registered guarded thresholds.
    """
    class F4Controller(runtime.FormalV2Controller):
        def __init__(self, plan, out, *, backend_factory=None):
            if (plan.get('schema') != 'maxopt-bridge-f4-runtime-plan-v1' or
                    plan.get('bridge_contract_sha256') != context['receipt']['contract_sha256']):
                raise ValueError('F4 runtime requires its own accepted contract binding')
            runtime._validate_runtime_plan(plan)
            if plan.get('formal') is True and (plan.get('window_seconds'), plan.get('drain_seconds')) != (1800, 300):
                raise ValueError('formal F4 timing differs from the registered contract')
            policy_id = plan.get('policy_id')
            if policy_id not in context['arms']:
                raise ValueError('unregistered F4 runtime policy identity')
            arm = validate_arm(context['arms'][policy_id])
            if plan.get('policy_spec') != policy_spec(arm):
                raise ValueError('runtime F4 ID/spec mismatch')
            if (plan.get('service_model') != context['models']['E'] or
                    plan.get('guard_safety') != context['safety']):
                raise ValueError('runtime guard service or safety differs from frozen context')
            if plan.get('adapter_initial_target') != 2:
                raise ValueError('F4 runtime requires adapter initial target 2')
            self.adapter = make_adapter(runtime, arm, plan['service_model'], plan['guard_safety'])
            self._device_since = [0.0, 0.0]
            self._current_tick = 0
            self._reserve = False
            runtime.gate.Controller.__init__(
                self, copy.deepcopy(plan), out,
                backend_factory=backend_factory or runtime.AsyncWorkerBackend,
                policy=self.adapter, formal=False)

    return F4Controller
