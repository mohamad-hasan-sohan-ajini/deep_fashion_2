"""Model configuration"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 512
    nhead: int = 8
    dropout: float = .1
    num_layers: int = 6
    height: int = 8
    width: int = 8
    max_objects: int = 10
    num_classes: int = 13 + 1
    num_keypoints: int = 294
