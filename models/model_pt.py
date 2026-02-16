"""Transformer model implementation"""

from typing import Callable

import torch
from torch import Tensor, nn

from models.config import ModelConfig
from models.object_queries import ObjectQueries
from models.positional_encoding import PositionalEncoding2D


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        batch_first: bool,
        positional_encoder: nn.Module,
    ):
        super().__init__()
        self.positional_encoder = positional_encoder
        self.mha = nn.MultiheadAttention(
            d_model,
            d_model // 64,
            dropout,
            batch_first=batch_first,
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(inplace=True),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        q = k = self.positional_encoder(x)
        x = x + self.dropout(self.mha(q, k, x)[0])
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        batch_first: bool,
        positional_encoder: nn.Module,
    ):
        super().__init__()
        self.positional_encoder = positional_encoder
        self.mha1 = nn.MultiheadAttention(
            d_model,
            d_model // 64,
            dropout,
            batch_first=batch_first,
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.mha2 = nn.MultiheadAttention(
            d_model,
            d_model // 64,
            dropout,
            batch_first=batch_first,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(inplace=True),
        )
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        # target needs no positional encoding (It must be learned through object queries)
        tgt = tgt + self.dropout(self.mha1(tgt, tgt, tgt)[0])
        tgt = self.ln1(tgt)
        k = self.positional_encoder(memory)
        tgt = tgt + self.dropout(self.mha2(tgt, k, memory)[0])
        tgt = self.ln2(tgt)
        tgt = tgt + self.dropout(self.ff(tgt))
        tgt = self.ln3(tgt)
        return tgt


class TransformerModel(nn.Module):
    def __init__(
        self,
        backbone_builder: Callable,
        feature_num_layers: int,
        positional_encoding_builder: PositionalEncoding2D,
    ) -> None:
        super().__init__()
        # Feature extraction
        self.feature_extractor = backbone_builder(
            feature_num_layers,
            ModelConfig.d_model,
        )
        # Memory positional encoder
        self.positional_encoder = positional_encoding_builder(
            ModelConfig.d_model,
            ModelConfig.height,
            ModelConfig.width,
        )
        # Transformer encoder
        self.encoder1 = TransformerEncoderLayer(
            ModelConfig.d_model,
            ModelConfig.dropout,
            True,
            self.positional_encoder,
        )
        self.encoder2 = TransformerEncoderLayer(
            ModelConfig.d_model,
            ModelConfig.dropout,
            True,
            self.positional_encoder,
        )
        self.encoder3 = TransformerEncoderLayer(
            ModelConfig.d_model,
            ModelConfig.dropout,
            True,
            self.positional_encoder,
        )
        # Transformer decoder
        self.decoder1 = TransformerDecoderLayer(
            ModelConfig.d_model,
            ModelConfig.dropout,
            True,
            self.positional_encoder,
        )
        self.decoder2 = TransformerDecoderLayer(
            ModelConfig.d_model,
            ModelConfig.dropout,
            True,
            self.positional_encoder,
        )
        self.decoder3 = TransformerDecoderLayer(
            ModelConfig.d_model,
            ModelConfig.dropout,
            True,
            self.positional_encoder,
        )
        # object queries
        self.object_queries = ObjectQueries(
            ModelConfig.d_model,
            ModelConfig.max_objects,
        )
        self.class_ffn = nn.Sequential(
            nn.Linear(ModelConfig.d_model, ModelConfig.d_model),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ModelConfig.d_model, ModelConfig.d_model),
        )
        self.bbox_ffn = nn.Sequential(
            nn.Linear(ModelConfig.d_model, ModelConfig.d_model),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ModelConfig.d_model, 4),
            nn.Sigmoid(),
        )
        self.keypoints_ffn = nn.Sequential(
            nn.Linear(ModelConfig.d_model, ModelConfig.d_model),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ModelConfig.d_model, 2 * ModelConfig.num_keypoints),
            nn.Sigmoid(),
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # extract features
        memory = self.feature_extractor(images)
        # amend data shape
        batch_size, d_model, *_ = memory.size()
        memory = memory.view(batch_size, d_model, -1).permute(0, 2, 1).contiguous()
        # transformer encoder
        memory = self.encoder1(memory)
        memory = self.encoder2(memory)
        memory = self.encoder3(memory)
        # get targets
        target = self.object_queries().repeat(batch_size, 1, 1)
        # transformer decoder
        target = self.decoder1(target, memory)
        target = self.decoder2(target, memory)
        target = self.decoder3(target, memory)
        # run heads
        predicted_classes = self.class_ffn(target)
        predicted_bboxes = self.bbox_ffn(target)
        predicted_keypoints = self.keypoints_ffn(target)
        predicted_keypoints = predicted_keypoints.view(
            batch_size,
            -1,
            ModelConfig.num_keypoints,
            2,
        )
        return predicted_classes, predicted_bboxes, predicted_keypoints


if __name__ == "__main__":
    import torch
    from models.positional_encoding import (
        FixedPositionalEncoding2D,
        LearnablePositionalEncoding2D,
    )
    from models.utils import get_vgg_backbone, get_resnet_backbone
    from models.model_pt import TransformerModel

    model = TransformerModel(
        get_resnet_backbone,
        18,
        FixedPositionalEncoding2D,
    )

    x = torch.randn(16, 3, 256, 256)
    print(f"{x.size() = }")
    predicted_classes, predicted_bboxes, predicted_keypoints = model(x)
    print(f"{predicted_classes.shape = }")
    print(f"{predicted_bboxes.shape = }")
    print(f"{predicted_keypoints.shape = }")
