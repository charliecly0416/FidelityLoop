"""PPO 训练器实现 (优化版)。

包含:
- GAE (Generalized Advantage Estimation) 计算
- PPO-Clip 损失函数
- Mini-batch 更新循环
- KL 散度早停机制
- 自适应学习率调整
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

from .ppo_network import ActorCriticNetwork


@dataclass
class Transition:
    """单步转移数据。"""

    obs: torch.Tensor  # [obs_dim]
    scale_action: int
    offline_action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    scale_mask: Optional[List[int]] = None
    offline_mask: Optional[List[int]] = None
    future_shield_triggered_15: Optional[float] = None


@dataclass
class TemporalTransition:
    """时序模型的单步转移数据。

    与普通 Transition 的区别：obs 是完整的历史序列。
    """

    obs_history: torch.Tensor  # [history_len, obs_dim]
    scale_action: int
    offline_action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    scale_mask: Optional[List[int]] = None
    offline_mask: Optional[List[int]] = None


class RolloutBuffer:
    """Rollout 缓冲区，存储多个 episode 的轨迹数据。"""

    def __init__(self):
        self.trajectories: List[List[Transition]] = []

    def add_trajectory(self, trajectory: List[Transition]):
        """添加一个 episode 的轨迹。"""
        if trajectory:
            self.trajectories.append(trajectory)

    def clear(self):
        """清空缓冲区。"""
        self.trajectories = []

    def __len__(self):
        return sum(len(traj) for traj in self.trajectories)


class PPOTrainer:
    """PPO 训练器 (优化版)。

    实现 PPO-Clip 算法:
    1. 收集多个 episode 的轨迹
    2. 计算 GAE 优势估计
    3. 多轮 mini-batch 更新
    4. KL 散度早停机制
    5. 自适应学习率调整
    """

    def __init__(
        self,
        model,  # ActorCriticNetwork or TransformerActorCritic
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        K_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
        lr_schedule: str = "constant",  # "constant", "linear", "adaptive", "cosine_warmup"
        total_iterations: int = 1000,
        warmup_iterations: int = 50,  # warmup 迭代次数 (用于 cosine_warmup)
        target_kl: Optional[float] = None,  # KL 早停阈值
        kl_coef: float = 0.0,  # KL 惩罚系数 (用于自适应)
        normalize_advantages: bool = True,
        clip_value_loss: bool = True,
        cost_critic_loss_weight: float = 0.0,
        aux_shield_loss_weight: float = 0.0,
        aux_shield_loss_warmup_steps: int = 0,
    ):
        """初始化训练器。

        Args:
            model: Actor-Critic 网络
            lr: 学习率
            gamma: 折扣因子 (高值以考虑扩容延迟)
            gae_lambda: GAE lambda 参数
            epsilon_clip: PPO clip 范围
            value_coef: 价值损失权重
            entropy_coef: 熵正则化权重
            max_grad_norm: 梯度裁剪阈值
            K_epochs: 每批数据重用次数
            batch_size: Mini-batch 大小
            device: 计算设备
            lr_schedule: 学习率调度 ("constant", "linear", "adaptive", "cosine_warmup")
            total_iterations: 总迭代次数 (用于学习率衰减)
            warmup_iterations: warmup 迭代次数 (用于 cosine_warmup)
            target_kl: KL 散度早停阈值 (None 表示不使用)
            kl_coef: KL 惩罚系数 (用于自适应 KL 控制)
            normalize_advantages: 是否标准化优势函数
            clip_value_loss: 是否裁剪价值损失
        """
        self.model = model
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

        # 学习率调度
        self.lr_schedule = lr_schedule
        self.initial_lr = lr
        self.current_lr = lr
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations

        # KL 散度控制
        self.target_kl = target_kl
        self.kl_coef = kl_coef
        self.adaptive_kl_coef = kl_coef  # 自适应系数

        # 其他选项
        self.normalize_advantages = normalize_advantages
        self.clip_value_loss = clip_value_loss
        self.cost_critic_loss_weight = max(float(cost_critic_loss_weight), 0.0)
        self.aux_shield_loss_weight = max(float(aux_shield_loss_weight), 0.0)
        self.aux_shield_loss_warmup_steps = max(int(aux_shield_loss_warmup_steps), 0)

        # TensorBoard writer
        self.writer: Optional[SummaryWriter] = None

        # 训练统计
        self.current_iteration = 0
        self.early_stop_count = 0  # 早停计数
        self.last_advantage_diagnostics: Dict[str, float] = {}

    @staticmethod
    def _tensor_diagnostics(prefix: str, values: torch.Tensor) -> Dict[str, float]:
        detached = values.detach()
        count = int(detached.numel())
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        nan_count = int(torch.isnan(detached).sum().item())
        posinf_count = int(torch.isposinf(detached).sum().item())
        neginf_count = int(torch.isneginf(detached).sum().item())
        stats: Dict[str, float] = {
            f"{prefix}_count": float(count),
            f"{prefix}_finite_count": float(finite_count),
            f"{prefix}_nan_count": float(nan_count),
            f"{prefix}_posinf_count": float(posinf_count),
            f"{prefix}_neginf_count": float(neginf_count),
            f"{prefix}_inf_count": float(posinf_count + neginf_count),
        }
        if finite_count == 0:
            stats.update(
                {
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_std": 0.0,
                    f"{prefix}_min": 0.0,
                    f"{prefix}_max": 0.0,
                }
            )
            return stats

        finite_values = detached[finite_mask]
        stats.update(
            {
                f"{prefix}_mean": float(finite_values.mean().item()),
                f"{prefix}_std": (
                    float(finite_values.std().item()) if finite_count > 1 else 0.0
                ),
                f"{prefix}_min": float(finite_values.min().item()),
                f"{prefix}_max": float(finite_values.max().item()),
            }
        )
        return stats

    def set_writer(self, writer: Optional[SummaryWriter]):
        """设置 TensorBoard writer。"""
        self.writer = writer

    def _update_learning_rate(self, iteration: int, avg_kl: Optional[float] = None):
        """更新学习率。

        Args:
            iteration: 当前迭代次数
            avg_kl: 平均 KL 散度 (用于自适应调整)
        """
        import math

        if self.lr_schedule == "linear":
            # 线性衰减
            frac = 1.0 - iteration / self.total_iterations
            self.current_lr = self.initial_lr * max(frac, 0.1)
        elif self.lr_schedule == "cosine_warmup":
            # Cosine with warmup
            if iteration < self.warmup_iterations:
                # 线性 warmup
                self.current_lr = self.initial_lr * (iteration + 1) / self.warmup_iterations
            else:
                # Cosine 衰减
                progress = (iteration - self.warmup_iterations) / max(
                    1, self.total_iterations - self.warmup_iterations
                )
                self.current_lr = self.initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.lr_schedule == "adaptive" and avg_kl is not None and self.target_kl is not None:
            # 自适应调整: KL 过大则降低学习率，过小则提高
            if avg_kl > self.target_kl * 2.0:
                self.current_lr = max(self.current_lr * 0.5, self.initial_lr * 0.01)
            elif avg_kl < self.target_kl * 0.5:
                self.current_lr = min(self.current_lr * 1.5, self.initial_lr * 2.0)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

    def _update_adaptive_kl_coef(self, avg_kl: float):
        """自适应更新 KL 惩罚系数。"""
        if self.target_kl is None or self.kl_coef == 0:
            return

        if avg_kl > self.target_kl * 1.5:
            self.adaptive_kl_coef = min(self.adaptive_kl_coef * 2.0, 1.0)
        elif avg_kl < self.target_kl * 0.5:
            self.adaptive_kl_coef = max(self.adaptive_kl_coef * 0.5, self.kl_coef * 0.1)

    def compute_gae(
        self,
        trajectories: List,
        is_temporal: bool = False,
        bootstrap_values: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """计算 GAE 优势估计。

        Args:
            trajectories: 多个 episode 的轨迹列表
            is_temporal: 是否为时序模型的轨迹 (TemporalTransition)
            bootstrap_values: 可选的每条轨迹末尾 value bootstrap。用于
                nonterminal truncated rollout；默认 None 保持完整 episode
                terminal 口径。

        Returns:
            obs: [N, obs_dim] or [N, history_len, obs_dim] (时序模式)
            scale_actions: [N]
            offline_actions: [N]
            old_log_probs: [N]
            advantages: [N]
            returns: [N]
            scale_masks: [N, 3] or None
            offline_masks: [N, 5] or None
        """
        all_obs = []
        all_scale_actions = []
        all_offline_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_scale_masks = []
        all_offline_masks = []
        all_aux_targets = []
        has_masks = False
        has_aux_targets = False

        for traj_idx, trajectory in enumerate(trajectories):
            if len(trajectory) == 0:
                continue

            rewards = [t.reward for t in trajectory]
            values = [t.value for t in trajectory]
            dones = [float(t.done) for t in trajectory]
            bootstrap_value = 0.0
            if bootstrap_values is not None and traj_idx < len(bootstrap_values):
                bootstrap_value = float(bootstrap_values[traj_idx])

            # GAE 计算 (反向遍历)
            advantages = []
            gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0.0 if dones[t] else bootstrap_value
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae)

            # Returns = Advantages + Values
            returns = [adv + val for adv, val in zip(advantages, values)]

            # 收集数据
            for i, t in enumerate(trajectory):
                # 根据是否为时序模式选择观测字段
                if is_temporal:
                    all_obs.append(t.obs_history)
                else:
                    all_obs.append(t.obs)
                all_scale_actions.append(t.scale_action)
                all_offline_actions.append(t.offline_action)
                all_old_log_probs.append(t.log_prob)
                all_advantages.append(advantages[i])
                all_returns.append(returns[i])

                if t.scale_mask is not None:
                    all_scale_masks.append(t.scale_mask)
                    all_offline_masks.append(t.offline_mask or [1] * 5)
                    has_masks = True
                aux_target = getattr(t, "future_shield_triggered_15", None)
                if aux_target is not None:
                    all_aux_targets.append(float(aux_target))
                    has_aux_targets = True

        if len(all_obs) == 0:
            # 返回空张量
            empty = torch.tensor([], device=self.device)
            return empty, empty, empty, empty, empty, empty, None, None, None

        # 转换为 Tensor
        obs = torch.stack(all_obs).to(self.device)
        scale_actions = torch.tensor(all_scale_actions, dtype=torch.long, device=self.device)
        offline_actions = torch.tensor(all_offline_actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(all_returns, dtype=torch.float32, device=self.device)

        raw_advantages = advantages.clone()
        normalized_applied = bool(self.normalize_advantages and advantages.numel() > 1)

        # 优势标准化
        if normalized_applied:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.last_advantage_diagnostics = {
            "advantage_normalize_advantages": float(bool(self.normalize_advantages)),
            "advantage_normalization_applied": float(normalized_applied),
        }
        self.last_advantage_diagnostics.update(
            self._tensor_diagnostics("advantage_raw", raw_advantages)
        )
        self.last_advantage_diagnostics.update(
            self._tensor_diagnostics("advantage_normalized", advantages)
        )

        # Masks
        scale_masks = None
        offline_masks = None
        if has_masks and len(all_scale_masks) == len(all_obs):
            scale_masks = torch.tensor(all_scale_masks, dtype=torch.float32, device=self.device)
            offline_masks = torch.tensor(all_offline_masks, dtype=torch.float32, device=self.device)
        if has_aux_targets and len(all_aux_targets) != len(all_obs):
            has_aux_targets = False

        aux_targets = (
            torch.tensor(all_aux_targets, dtype=torch.float32, device=self.device)
            if has_aux_targets
            else None
        )

        return obs, scale_actions, offline_actions, old_log_probs, advantages, returns, scale_masks, offline_masks, aux_targets

    def update(
        self,
        trajectories: List,
        iteration: int,
        is_temporal: bool = False,
        bootstrap_values: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """执行 PPO 更新。

        Args:
            trajectories: 多个 episode 的轨迹列表
            iteration: 当前迭代次数
            is_temporal: 是否为时序模型的轨迹
            bootstrap_values: 可选的每条轨迹末尾 value bootstrap。仅用于
                nonterminal truncated rollout；默认 None 保持原行为。

        Returns:
            metrics: 训练指标字典
        """
        self.current_iteration = iteration

        # 计算 GAE
        data = self.compute_gae(
            trajectories,
            is_temporal=is_temporal,
            bootstrap_values=bootstrap_values,
        )
        obs, scale_actions, offline_actions, old_log_probs, advantages, returns, scale_masks, offline_masks, aux_targets = data

        n_samples = obs.size(0)
        if n_samples == 0:
            return {"error": "no samples"}

        # 累积指标
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_cost_value_mean = 0.0
        total_aux_shield_loss = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        update_count = 0
        nonfinite_policy_loss_count = 0
        nonfinite_value_loss_count = 0
        nonfinite_entropy_count = 0
        nonfinite_kl_count = 0
        nonfinite_total_loss_count = 0
        early_stopped = False

        # 获取旧的价值估计 (用于 value clipping)
        with torch.no_grad():
            _, _, old_values = self.model(obs, scale_masks, offline_masks)

        # K 轮更新
        for epoch in range(self.K_epochs):
            # 随机打乱索引
            indices = torch.randperm(n_samples, device=self.device)

            epoch_kl_sum = 0.0
            epoch_batch_count = 0

            # Mini-batch 更新
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]

                # 获取 batch 数据
                batch_obs = obs[batch_indices]
                batch_scale_actions = scale_actions[batch_indices]
                batch_offline_actions = offline_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_aux_targets = aux_targets[batch_indices] if aux_targets is not None else None

                batch_scale_masks = scale_masks[batch_indices] if scale_masks is not None else None
                batch_offline_masks = offline_masks[batch_indices] if offline_masks is not None else None

                # Forward pass
                scale_logits, offline_logits, values = self.model(
                    batch_obs, batch_scale_masks, batch_offline_masks
                )

                # 创建分布
                scale_dist = Categorical(logits=scale_logits)
                offline_dist = Categorical(logits=offline_logits)

                # 计算新的 log 概率
                new_log_probs = (
                    scale_dist.log_prob(batch_scale_actions)
                    + offline_dist.log_prob(batch_offline_actions)
                )

                # 熵
                entropy = scale_dist.entropy().mean() + offline_dist.entropy().mean()

                # 计算 KL 散度
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()
                    epoch_kl_sum += approx_kl.item()
                    epoch_batch_count += 1

                # KL 早停检查 (在 epoch 内)
                if self.target_kl is not None and approx_kl.item() > self.target_kl * 1.5:
                    early_stopped = True
                    break

                # === Policy Loss (PPO-Clip) ===
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 可选: 添加 KL 惩罚项
                if self.adaptive_kl_coef > 0:
                    policy_loss = policy_loss + self.adaptive_kl_coef * approx_kl

                # === Value Loss ===
                value_pred = values
                if self.clip_value_loss:
                    value_pred_clipped = batch_old_values + torch.clamp(
                        value_pred - batch_old_values,
                        -self.epsilon_clip,
                        self.epsilon_clip,
                    )
                    value_loss_unclipped = (value_pred - batch_returns) ** 2
                    value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((value_pred - batch_returns) ** 2).mean()

                cost_value_mean = torch.tensor(0.0, device=self.device)
                cost_loss = torch.tensor(0.0, device=self.device)
                cost_value = getattr(self.model, "cost_value", None)
                if callable(cost_value):
                    predicted_cost = cost_value(batch_obs)
                    cost_value_mean = predicted_cost.detach().mean()
                    if self.cost_critic_loss_weight > 0.0:
                        cost_loss = 0.5 * (predicted_cost ** 2).mean()
                aux_shield_loss = torch.tensor(0.0, device=self.device)
                aux_shield_weight = self._aux_shield_weight(iteration)
                aux_logits_fn = getattr(self.model, "aux_shield_logits", None)
                if (
                    aux_shield_weight > 0.0
                    and batch_aux_targets is not None
                    and callable(aux_logits_fn)
                ):
                    aux_logits = aux_logits_fn(batch_obs)
                    if aux_logits is not None:
                        aux_pred = aux_logits[:, 0]
                        aux_shield_loss = nn.functional.binary_cross_entropy_with_logits(
                            aux_pred,
                            batch_aux_targets,
                        )

                # === Total Loss ===
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.cost_critic_loss_weight * cost_loss
                    + aux_shield_weight * aux_shield_loss
                    - self.entropy_coef * entropy
                )

                if not bool(torch.isfinite(policy_loss).all().item()):
                    nonfinite_policy_loss_count += 1
                if not bool(torch.isfinite(value_loss).all().item()):
                    nonfinite_value_loss_count += 1
                if not bool(torch.isfinite(entropy).all().item()):
                    nonfinite_entropy_count += 1
                if not bool(torch.isfinite(approx_kl).all().item()):
                    nonfinite_kl_count += 1
                if not bool(torch.isfinite(loss).all().item()):
                    nonfinite_total_loss_count += 1

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

                # 记录指标
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.epsilon_clip).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_cost_value_mean += float(cost_value_mean.item())
                total_aux_shield_loss += float(aux_shield_loss.item())
                total_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                update_count += 1

            # Epoch 级别的 KL 早停检查
            if early_stopped:
                self.early_stop_count += 1
                break

            avg_epoch_kl = epoch_kl_sum / max(epoch_batch_count, 1)
            if self.target_kl is not None and avg_epoch_kl > self.target_kl:
                self.early_stop_count += 1
                break

        # 计算平均 KL
        avg_kl = total_kl / max(update_count, 1)

        # 更新学习率和自适应 KL 系数
        self._update_learning_rate(iteration, avg_kl)
        self._update_adaptive_kl_coef(avg_kl)

        # 平均指标
        metrics = {
            "policy_loss": total_policy_loss / max(update_count, 1),
            "value_loss": total_value_loss / max(update_count, 1),
            "cost_value_mean": total_cost_value_mean / max(update_count, 1),
            "cost_critic_loss_weight": self.cost_critic_loss_weight,
            "aux_shield_loss": total_aux_shield_loss / max(update_count, 1),
            "aux_shield_loss_weight": self.aux_shield_loss_weight,
            "aux_shield_effective_weight": self._aux_shield_weight(iteration),
            "aux_shield_loss_warmup_steps": float(self.aux_shield_loss_warmup_steps),
            "entropy": total_entropy / max(update_count, 1),
            "approx_kl": avg_kl,
            "clip_fraction": total_clip_fraction / max(update_count, 1),
            "n_samples": n_samples,
            "n_updates": update_count,
            "lr": self.current_lr,
            "early_stopped": int(early_stopped),
            "adaptive_kl_coef": self.adaptive_kl_coef,
            "policy_loss_nonfinite_count": float(nonfinite_policy_loss_count),
            "value_loss_nonfinite_count": float(nonfinite_value_loss_count),
            "entropy_nonfinite_count": float(nonfinite_entropy_count),
            "approx_kl_nonfinite_count": float(nonfinite_kl_count),
            "total_loss_nonfinite_count": float(nonfinite_total_loss_count),
        }
        metrics.update(self.last_advantage_diagnostics)

        # TensorBoard logging
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, iteration)

        return metrics

    def _aux_shield_weight(self, iteration: int) -> float:
        if self.aux_shield_loss_weight <= 0.0:
            return 0.0
        if self.aux_shield_loss_warmup_steps <= 0:
            return self.aux_shield_loss_weight
        progress = min(max(float(iteration + 1) / float(self.aux_shield_loss_warmup_steps), 0.0), 1.0)
        return self.aux_shield_loss_weight * progress

    def save(self, path: str):
        """保存模型和优化器状态。"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iteration": self.current_iteration,
                "lr": self.current_lr,
                "adaptive_kl_coef": self.adaptive_kl_coef,
            },
            path,
        )

    def load(self, path: str):
        """加载模型和优化器状态。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iteration = checkpoint.get("iteration", 0)
        self.current_lr = checkpoint.get("lr", self.initial_lr)
        self.adaptive_kl_coef = checkpoint.get("adaptive_kl_coef", self.kl_coef)


def test_trainer():
    """测试训练器。"""
    print("Testing PPOTrainer...")

    # 创建模型和训练器
    model = ActorCriticNetwork(obs_dim=67)
    trainer = PPOTrainer(
        model,
        device="cpu",
        target_kl=0.015,
        kl_coef=0.2,
        lr_schedule="adaptive",
    )

    # 创建模拟轨迹
    trajectories = []
    for _ in range(3):  # 3 个 episode
        trajectory = []
        for t in range(100):  # 每个 episode 100 步
            trajectory.append(
                Transition(
                    obs=torch.randn(67),
                    scale_action=np.random.randint(0, 3),
                    offline_action=np.random.randint(0, 5),
                    log_prob=-1.5 + np.random.randn() * 0.1,
                    value=np.random.randn(),
                    reward=np.random.randn() * 0.1,
                    done=(t == 99),
                    scale_mask=[1, 1, 1],
                    offline_mask=[1, 1, 1, 1, 1],
                )
            )
        trajectories.append(trajectory)

    # 执行更新
    metrics = trainer.update(trajectories, iteration=0)
    print(f"Metrics: {metrics}")

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert metrics["n_samples"] == 300
    assert "early_stopped" in metrics

    print("All tests passed!")


if __name__ == "__main__":
    test_trainer()
