"""Model configuration"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 512
    nhead: int = 8
    dropout: float = .1
    num_layers: int = 6
    height: int = 32
    width: int = 32
    max_objects: int = 10
