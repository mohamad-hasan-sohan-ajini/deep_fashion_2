"""Pytorch models"""

import math

import torch
from torch import Tensor, nn


class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D Positional Encoding"""

    def __init__(
            self,
            d_model: int,
            height: int,
            width: int,
    ):
        """Initialize

        :param d_model: Model hidden size
        :type d_model: int
        :param max_len: maximum length of input signal
        :type max_len: int
        """
        super().__init__()
        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = (
            torch
            .sin(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch
            .cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[d_model::2, :, :] = (
            torch
            .sin(pos_h * div_term)
            .transpose(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, width)
        )
        pe[d_model + 1::2, :, :] = (
            torch
            .cos(pos_h * div_term)
            .transpose(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, width)
        )
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to the signal

        :param x: input of shape [batch_size, channels, height, width]
        :type x: torch.Tensor
        :returns: Positional encoded added signal
        :rtype: torch.Tensor
        """
        return x + self.pe
