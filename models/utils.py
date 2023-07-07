"""Utilities"""

import torch
from torch import Tensor, nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.main(x)
        return x


def get_vgg_backbone(num_layers: int, d_model: int) -> nn.Module:
    # valid num_layers: {11, 13, 16, 19}
    vgg_model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        f'vgg{num_layers}_bn',
        pretrained=True,
    )
    num_maxpools = 0
    layers = []
    for layer in vgg_model.features:
        layers.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            num_maxpools += 1
            if num_maxpools == 3:
                break
    layers.append(ConvLayer(256, d_model))
    features = nn.Sequential(*layers)
    return features


def get_resnet_backbone(num_layers: int, d_model: int) -> nn.Module:
    # valid num_layers: {18, 34, 50, 101, 152}
    resnet_model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        f'resnet{num_layers}',
        pretrained=True,
    )
    in_channels = 128 if num_layers in [18, 34] else 512
    features = nn.Sequential(
        resnet_model.conv1,
        resnet_model.bn1,
        resnet_model.relu,
        resnet_model.maxpool,
        resnet_model.layer1,
        resnet_model.layer2,
        ConvLayer(in_channels, d_model),
    )
    return features
