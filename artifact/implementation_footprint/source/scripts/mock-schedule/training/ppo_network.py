"""Actor-Critic 网络实现。

双头网络结构：
- 共享 backbone 提取特征
- scale_head: 输出扩缩容动作 logits (3 类)
- offline_head: 输出离线准入级别 logits (4 类)
- value_head: 输出状态价值估计
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """双头 Actor-Critic 网络，用于多硬件调度决策。

    动作空间: MultiDiscrete([3, 4])
    - scale_action: 0=NONE, 1=SCALE_UP, 2=SCALE_DOWN
    - offline_level: 0=PAUSE, 1=LOW, 2=MEDIUM, 3=HIGH
    """

    def __init__(
        self,
        obs_dim: int = 67,
        hidden_dim: int = 256,
        num_layers: int = 2,
        enable_cost_critic_head: bool = False,
        enable_aux_shield_head: bool = False,
        aux_shield_outputs: int = 1,
    ):
        """初始化网络。

        Args:
            obs_dim: 观测维度 (默认 67 = 19 + 11*4 + 4)
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.enable_cost_critic_head = bool(enable_cost_critic_head)
        self.enable_aux_shield_head = bool(enable_aux_shield_head)
        self.aux_shield_outputs = max(int(aux_shield_outputs), 1)

        # 构建共享 backbone
        layers = []
        in_dim = obs_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Actor heads (两个独立的动作头)
        self.scale_head = nn.Linear(hidden_dim, 3)  # [NONE, UP, DOWN]
        self.offline_head = nn.Linear(hidden_dim, 4)  # [PAUSE, LOW, MEDIUM, HIGH]

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)
        self.cost_head = (
            nn.Linear(hidden_dim, 1) if self.enable_cost_critic_head else None
        )
        self.aux_shield_head = (
            nn.Linear(hidden_dim, self.aux_shield_outputs)
            if self.enable_aux_shield_head
            else None
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """正交初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Actor heads 使用更小的初始化
        nn.init.orthogonal_(self.scale_head.weight, gain=0.01)
        nn.init.orthogonal_(self.offline_head.weight, gain=0.01)

        # Value head 使用标准初始化
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        if self.cost_head is not None:
            nn.init.orthogonal_(self.cost_head.weight, gain=1.0)
        if self.aux_shield_head is not None:
            nn.init.orthogonal_(self.aux_shield_head.weight, gain=0.01)

    def shared_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def forward(
        self,
        obs: torch.Tensor,
        scale_mask: Optional[torch.Tensor] = None,
        offline_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            obs: [batch, obs_dim] 观测向量
            scale_mask: [batch, 3] 动作掩码 (1=有效, 0=无效)
            offline_mask: [batch, 4] 动作掩码

        Returns:
            scale_logits: [batch, 3] 扩缩容动作 logits
            offline_logits: [batch, 4] 离线准入 logits
            value: [batch] 状态价值
        """
        # 共享特征提取
        features = self.shared_features(obs)

        # Actor logits
        scale_logits = self.scale_head(features)
        offline_logits = self.offline_head(features)

        # 应用动作掩码 (将无效动作的 logits 设为极小值)
        if scale_mask is not None:
            scale_logits = scale_logits.masked_fill(scale_mask == 0, -1e9)
        if offline_mask is not None:
            offline_logits = offline_logits.masked_fill(offline_mask == 0, -1e9)

        # Value
        value = self.value_head(features).squeeze(-1)

        return scale_logits, offline_logits, value

    def cost_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return cost critic value when the optional head is enabled.

        Phase 3 keeps this as a plug-point: the default network has no cost
        head and returns zeros, preserving old checkpoint compatibility.
        """
        if self.cost_head is None:
            return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
        features = self.shared_features(obs)
        return self.cost_head(features).squeeze(-1)

    def aux_shield_logits(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """Return shield-override prediction logits when enabled.

        This head is a training-time auxiliary task. It is intentionally not
        used by get_action(), so inference behavior is unchanged.
        """
        if self.aux_shield_head is None:
            return None
        return self.aux_shield_head(self.shared_features(obs))

    def get_action(
        self,
        obs: torch.Tensor,
        scale_mask: Optional[torch.Tensor] = None,
        offline_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作并返回相关信息。

        Args:
            obs: [batch, obs_dim] 观测向量
            scale_mask: [batch, 3] 动作掩码
            offline_mask: [batch, 4] 动作掩码
            deterministic: 是否使用确定性策略 (argmax)

        Returns:
            scale_action: [batch] 扩缩容动作
            offline_action: [batch] 离线准入动作
            log_prob: [batch] 联合 log 概率
            value: [batch] 状态价值
            entropy: [batch] 策略熵
        """
        scale_logits, offline_logits, value = self.forward(obs, scale_mask, offline_mask)

        # 创建分布
        scale_dist = Categorical(logits=scale_logits)
        offline_dist = Categorical(logits=offline_logits)

        if deterministic:
            # 确定性策略: 选择概率最高的动作
            scale_action = scale_logits.argmax(dim=-1)
            offline_action = offline_logits.argmax(dim=-1)
        else:
            # 随机策略: 采样
            scale_action = scale_dist.sample()
            offline_action = offline_dist.sample()

        # 计算联合 log 概率 (两个独立动作的 log 概率相加)
        log_prob = scale_dist.log_prob(scale_action) + offline_dist.log_prob(offline_action)

        # 计算熵 (鼓励探索)
        entropy = scale_dist.entropy() + offline_dist.entropy()

        return scale_action, offline_action, log_prob, value, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        scale_actions: torch.Tensor,
        offline_actions: torch.Tensor,
        scale_mask: Optional[torch.Tensor] = None,
        offline_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定动作的 log 概率、价值和熵。

        Args:
            obs: [batch, obs_dim] 观测
            scale_actions: [batch] 扩缩容动作
            offline_actions: [batch] 离线准入动作
            scale_mask: [batch, 3] 动作掩码
            offline_mask: [batch, 4] 动作掩码

        Returns:
            log_prob: [batch] 联合 log 概率
            value: [batch] 状态价值
            entropy: [batch] 策略熵
        """
        scale_logits, offline_logits, value = self.forward(obs, scale_mask, offline_mask)

        scale_dist = Categorical(logits=scale_logits)
        offline_dist = Categorical(logits=offline_logits)

        log_prob = scale_dist.log_prob(scale_actions) + offline_dist.log_prob(offline_actions)
        entropy = scale_dist.entropy() + offline_dist.entropy()

        return log_prob, value, entropy


def test_network():
    """测试网络前向传播。"""
    print("Testing ActorCriticNetwork...")

    # 创建网络
    net = ActorCriticNetwork(obs_dim=67, hidden_dim=256)
    print(f"Network created: {sum(p.numel() for p in net.parameters())} parameters")

    # 测试前向传播
    batch_size = 4
    obs = torch.randn(batch_size, 67)
    scale_mask = torch.ones(batch_size, 3)
    scale_mask[:, 1] = 0  # 禁止 SCALE_UP
    offline_mask = torch.ones(batch_size, 4)

    scale_logits, offline_logits, value = net(obs, scale_mask, offline_mask)
    print(f"scale_logits shape: {scale_logits.shape}")
    print(f"offline_logits shape: {offline_logits.shape}")
    print(f"value shape: {value.shape}")

    # 测试动作采样
    scale_action, offline_action, log_prob, value, entropy = net.get_action(
        obs, scale_mask, offline_mask
    )
    print(f"scale_action: {scale_action}")
    print(f"offline_action: {offline_action}")
    print(f"log_prob: {log_prob}")
    print(f"entropy: {entropy}")

    # 验证 mask 生效 (SCALE_UP 不应被选中)
    assert (scale_action != 1).all(), "Mask not working: SCALE_UP should be masked"

    print("All tests passed!")


if __name__ == "__main__":
    test_network()
