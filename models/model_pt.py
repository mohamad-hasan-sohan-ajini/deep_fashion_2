"""Transformer model implementation"""

from typing import Callable

import torch
from torch import Tensor, nn

from models.config import ModelConfig
from models.object_queries import ObjectQueries
from models.positional_encoding import (
    PositionalEncoding2D,
    FixedPositionalEncoding2D,
    LearnablePositionalEncoding2D,
)
from models.utils import get_vgg_backbone, get_resnet_backbone


class TransformerModel(nn.Module):
    def __init__(
            self,
            backbone_builder: Callable = get_resnet_backbone,
            feature_num_layers: int = 18,
            positional_encoding_builder: PositionalEncoding2D = FixedPositionalEncoding2D,
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
        self.decoder = nn.TransformerDecoder(
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
        self.class_ffn = nn.Linear(
            ModelConfig.d_model,
            ModelConfig.num_classes,
        )
        self.bbox_ffn = nn.Linear(ModelConfig.d_model, 4)
        self.keypoints_ffn = nn.Linear(
            ModelConfig.d_model,
            ModelConfig.num_keypoints,
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # extract features
        x = self.feature_extractor(images)
        # add positional encoding
        x = self.positional_encoder(x)
        # amend data shape
        batch_size, d_model, *_ = x.size()
        x = x.view(batch_size, d_model, -1).permute(0, 2, 1).contiguous()
        # transformer encoder
        memory = self.encoder(x)
        # get targets
        targets = self.object_queries().repeat(batch_size, 1, 1)
        # transformer decoder
        x = self.decoder(targets, memory)
        # run heads
        predicted_classes = self.class_ffn(x)
        predicted_bboxes = self.bbox_ffn(x)
        predicted_keypoints = self.keypoints_ffn(x)
        return predicted_classes, predicted_bboxes, predicted_keypoints


if __name__ == '__main__':
    model = TransformerModel()

    x = torch.randn(16, 3, 256, 256)
    print(f'{x.size() = }')
    predicted_classes, predicted_bboxes, predicted_keypoints = model(x)
    print(f'{predicted_classes.shape = }')
    print(f'{predicted_bboxes.shape = }')
    print(f'{predicted_keypoints.shape = }')
