"""Transformer model implementation"""

from typing import Callable, Optional

import torch
from torch import Tensor, nn

from config import ModelConfig
from object_queries import ObjectQueries
from positional_encoding import (
    FixedPositionalEncoding2D,
    LearnablePositionalEncoding2D,
)
from utils import get_vgg19_backbone, get_resnet_backbone


class TransformerModel(nn.Module):
    def __init__(
            self,
            backbone_builder: Callable = get_vgg19_backbone,
            num_layers: int = 18,
    ) -> None:
        super().__init__()
        # Feature extraction
        self.feature_extractor = backbone_builder(num_layers)
        # Transformer encoder
        # Transformer decoder
        # Positional encoder
        # object queries
