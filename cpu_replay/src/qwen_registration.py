"""CPU configuration probe only; importing this does not supply GPU profiles."""
from dataclasses import dataclass

from vidur.config.model_config import QwenModelConfig


@dataclass
class Qwen25_7BInstructConfig(QwenModelConfig):
    num_layers: int = 28
    num_q_heads: int = 28
    num_kv_heads: int = 4
    embedding_dim: int = 3584
    mlp_hidden_dim: int = 18944
    rope_theta: float = 1000000.0

    @staticmethod
    def get_name():
        return "Qwen/Qwen2.5-7B-Instruct"
