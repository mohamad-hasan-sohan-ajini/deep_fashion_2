"""Utilities"""

import torch
from torch import nn


def get_vgg19_backbone(_: int) -> nn.Module:
    vgg_model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'vgg19',
        pretrained=True,
    )
    return vgg_model.features


def get_resnet_backbone(num_layers: int) -> nn.Module:
    # valid num_layers: {18, 34, 50, 101, 152}
    resnet_model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        f'resnet{num_layers}',
        pretrained=True,
    )
    features = nn.Sequential(
        resnet_model.conv1,
        resnet_model.bn1,
        resnet_model.relu,
        resnet_model.maxpool,
        resnet_model.layer1,
        resnet_model.layer2,
        resnet_model.layer3,
        resnet_model.layer4,
    )
    return features
