"""Pytorch models"""

import math

import torch
from torch import Tensor, nn


class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D Positional Encoding Base Class"""
    def __init__(
            self,
            d_model: int,
            height: int,
            width: int,
    ) -> None:
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
        self._pe = pe

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to the signal

        :param x: input of shape [batch_size, channels, height, width]
        :type x: torch.Tensor
        :returns: Positional encoded added signal
        :rtype: torch.Tensor
        """
        return x + self.pe


class FixedPositionalEncoding2D(PositionalEncoding2D):
    """Sinusoidal 2D Positional Encoding"""
    def __init__(
            self,
            d_model: int,
            height: int,
            width: int,
    ) -> None:
        super().__init__(d_model, height, width)
        self.register_buffer('pe', self._pe)


class LearnablePositionalEncoding2D(PositionalEncoding2D):
    """Sinusoidal 2D Positional Encoding"""

    def __init__(
            self,
            d_model: int,
            height: int,
            width: int,
    ) -> None:
        super().__init__(d_model, height, width)
        self.register_parameter('pe', nn.Parameter(self._pe))


if __name__ == '__main__':
    d_model = 128
    height, width = 32, 32
    fix_pe2d = FixedPositionalEncoding2D(d_model, height, width)
    learnable_pe2d = LearnablePositionalEncoding2D(d_model, height, width)

    x = torch.randn(16, d_model, 32, 32)
    yf = fix_pe2d(x)
    yl = learnable_pe2d(x)
