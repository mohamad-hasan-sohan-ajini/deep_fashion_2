"""Model configuration"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_dim: int = 1024
    num_heads: int = 16
    dropout: float = .1
