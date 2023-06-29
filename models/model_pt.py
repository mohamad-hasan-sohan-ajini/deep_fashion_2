"""Transformer model implementation"""

from typing import Callable

import torch
from torch import Tensor, nn

from config import ModelConfig
from object_queries import ObjectQueries
from positional_encoding import (
    FixedPositionalEncoding2D,
    LearnablePositionalEncoding2D,
)
from utils import get_vgg_backbone, get_resnet_backbone


class TransformerModel(nn.Module):
    def __init__(
            self,
            backbone_builder: Callable = get_resnet_backbone,
            feature_num_layers: int = 18,
            positional_encoding_builder: Callable = FixedPositionalEncoding2D,
    ) -> None:
        super().__init__()
        # Feature extraction
        self.feature_extractor = backbone_builder(feature_num_layers)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            ModelConfig.d_model,
            ModelConfig.nhead,
            dropout=ModelConfig.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            ModelConfig.num_layers,
        )
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            ModelConfig.d_model,
            ModelConfig.nhead,
            dropout=ModelConfig.dropout,
            batch_first=True,
        )
        self.decoder_layer = nn.TransformerDecoder(
            decoder_layer,
            ModelConfig.num_layers,
        )
        # Positional encoder
        self.positional_encoder = positional_encoding_builder(
            ModelConfig.d_model,
            ModelConfig.height,
            ModelConfig.width,
        )
        # object queries
        self.object_queries = ObjectQueries(
            ModelConfig.d_model,
            ModelConfig.max_objects,
        )
