"""Utilities"""

from torch import Tensor, nn
from torchvision import models


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
    if num_layers not in {11, 13, 16, 19}:
        raise ValueError("VGG num_layers must be one of 11, 13, 16, or 19")

    vgg_model = models.get_model(f"vgg{num_layers}_bn", weights="DEFAULT")
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
    if num_layers not in {18, 34, 50, 101, 152}:
        raise ValueError("ResNet num_layers must be one of 18, 34, 50, 101, or 152")

    resnet_model = models.get_model(f"resnet{num_layers}", weights="DEFAULT")
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
