"""Thin, strict interface over the original PPO network and correct GAE/update.

The historical names scale_action/offline_action exist only at the algorithm
boundary. scale_action is the sampled target proposal, never a filtered target;
offline_action is a fixed zero with zero log-probability and entropy.
"""
import copy
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from scripts.maxopt_bridge.ppo_env import OBS_DIM, OFFLINE_MASK, validate_mask


_network = importlib.import_module('scripts.mock-schedule.training.ppo_network')
_trainer = importlib.import_module('scripts.mock-schedule.training.ppo_trainer')
COMMON_CONFIG = json.loads(Path(__file__).with_name('ppo_common.json').read_text())


def _batch_mask(mask, batch, width, device):
    if mask is None:
        raise ValueError('missing action mask')
    try:
        tensor = torch.as_tensor(mask, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError('invalid action mask') from exc
    if tensor.shape != (batch, width):
        raise ValueError('mask width or batch length mismatch')
    if (not torch.isfinite(tensor).all() or not ((tensor == 0) | (tensor == 1)).all()
            or not tensor.any(dim=1).all()):
        raise ValueError('mask must be binary, finite and nonempty in every row')
    return tensor


class CapacityNetwork(_network.ActorCriticNetwork):
    def __init__(self):
        super().__init__(obs_dim=OBS_DIM, hidden_dim=COMMON_CONFIG['hidden_dim'],
                         num_layers=COMMON_CONFIG['num_layers'])

    def _init_weights(self):
        gains = COMMON_CONFIG['initialization']
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gains['hidden_relu_gain'])
                nn.init.zeros_(module.bias)
        for head, gain in ((self.scale_head, gains['actor_gain']),
                           (self.offline_head, gains['actor_gain']),
                           (self.value_head, gains['value_gain'])):
            nn.init.orthogonal_(head.weight, gain=gain)
            nn.init.zeros_(head.bias)

    def forward(self, obs, scale_mask=None, offline_mask=None):
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM or not torch.isfinite(obs).all():
            raise ValueError('observation shape/nonfinite mismatch')
        scale = _batch_mask(scale_mask, obs.shape[0], 3, obs.device)
        offline = _batch_mask(offline_mask, obs.shape[0], 4, obs.device)
        expected = torch.tensor(OFFLINE_MASK, device=obs.device).expand(obs.shape[0], -1)
        if not torch.equal(offline, expected):
            raise ValueError('offline factor must be fixed to sole legal action zero')
        return super().forward(obs, scale, offline)

    def evaluate_actions(self, obs, scale_actions, offline_actions, scale_mask=None, offline_mask=None):
        scale = _batch_mask(scale_mask, obs.shape[0], 3, obs.device)
        if (scale_actions.shape != (obs.shape[0],) or offline_actions.shape != (obs.shape[0],)
                or scale_actions.dtype != torch.long or offline_actions.dtype != torch.long
                or not ((scale_actions >= 0) & (scale_actions < 3)).all()
                or not (offline_actions == 0).all()
                or not scale.gather(1, scale_actions[:, None]).all()):
            raise ValueError('actions must be legal capacity proposals and offline zero')
        return super().evaluate_actions(obs, scale_actions, offline_actions, scale, offline_mask)


@dataclass
class CapacityTransition:
    obs: torch.Tensor
    proposal: int
    filtered_target: int
    executed_target: int
    behavior_mask: list
    behavior_log_prob: float
    value: float
    reward: float
    done: bool
    offline_mask: list

    @property
    def scale_action(self):
        return self.proposal

    @property
    def offline_action(self):
        return 0

    @property
    def scale_mask(self):
        return self.behavior_mask

    @property
    def log_prob(self):
        return self.behavior_log_prob


class CapacityTrainer(_trainer.PPOTrainer):
    def __init__(self, model):
        super().__init__(model, device='cpu', **copy.deepcopy(COMMON_CONFIG['trainer']))

    def compute_gae(self, trajectories, is_temporal=False, bootstrap_values=None):
        if is_temporal:
            raise ValueError('temporal PPO is outside the capacity contract')
        if not trajectories or any(not trajectory for trajectory in trajectories):
            raise ValueError('rollout batch and each trajectory must be nonempty')
        if bootstrap_values is None or len(bootstrap_values) != len(trajectories):
            raise ValueError('one explicit bootstrap is required per trajectory')
        for trajectory, bootstrap in zip(trajectories, bootstrap_values):
            if not math.isfinite(float(bootstrap)):
                raise ValueError('bootstrap must be finite')
            if trajectory[-1].done and bootstrap != 0:
                raise ValueError('terminal episode must have zero bootstrap')
            for transition in trajectory:
                if not isinstance(transition, CapacityTransition):
                    raise ValueError('capacity transition with proposal provenance is required')
                mask = validate_mask(transition.behavior_mask, 3)
                offline = validate_mask(transition.offline_mask, 4)
                if offline != OFFLINE_MASK:
                    raise ValueError('offline factor must remain fixed')
                if (any(type(action) is not int or action not in (0, 1, 2)
                        for action in (transition.proposal, transition.filtered_target, transition.executed_target))
                        or not mask[transition.proposal] or transition.filtered_target != transition.executed_target):
                    raise ValueError('invalid proposal/filter/execution provenance')
                if (transition.obs.shape != (OBS_DIM,) or not torch.isfinite(transition.obs).all()
                        or type(transition.done) is not bool
                        or any(not math.isfinite(float(v)) for v in
                               (transition.behavior_log_prob, transition.value, transition.reward))
                        or transition.behavior_log_prob > 1e-7):
                    raise ValueError('invalid transition values')
        # The unchanged method retains correct terminal masking and bootstrap.
        return super().compute_gae(trajectories, False, bootstrap_values)


def make_transition(obs, value, reward, done, info):
    action = info['action']
    return CapacityTransition(torch.as_tensor(obs, dtype=torch.float32).clone(),
        action['proposal'], action['filtered_target'], action['executed_target'],
        list(action['behavior_mask']), action['behavior_log_prob'], float(value),
        float(reward), bool(done), list(OFFLINE_MASK))


def collect_rollout(env, model, steps, *, state=None):
    """Collection interface for S05; no optimizer, checkpoint selection or reset leakage.

    The last next-state value is evaluated only when the episode did not end.
    Returning state lets a caller continue a truncated episode after updating.
    Collection stops at the first terminal boundary; it never bootstraps reset.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError('rollout steps must be a positive integer')
    obs, info = env.reset() if state is None else state
    trajectory = []
    for _ in range(steps):
        with torch.no_grad():
            proposal, _, logprob, value, _ = model.get_action(
                torch.as_tensor(obs)[None], torch.tensor([info['capacity_mask']]),
                torch.tensor([info['offline_mask']]))
        next_obs, reward, terminal, truncated, next_info = env.step(
            int(proposal.item()), behavior_log_prob=float(logprob.item()), behavior_mask=info['capacity_mask'])
        if truncated:
            raise RuntimeError('simulator must not substitute rollout truncation for terminal')
        trajectory.append(make_transition(obs, value.item(), reward, terminal, next_info))
        obs, info = next_obs, next_info
        if terminal:
            break
    bootstrap = 0.0
    if not trajectory[-1].done:
        with torch.no_grad():
            _, _, value = model(torch.as_tensor(obs)[None], torch.tensor([info['capacity_mask']]),
                                torch.tensor([info['offline_mask']]))
        bootstrap = float(value.item())
    return trajectory, bootstrap, (obs, info)
