"""Model configuration"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 1024
    nhead: int = 16
    dropout: float = .1
    num_layers: int = 6
