"""A container for object queries"""

import torch
from torch import Tensor, nn


class ObjectQueries(nn.Module):
    def __init__(
            self,
            d_model: int = 1024,
            max_objects: int = 10,
    ) -> None:
        super().__init__()
        params = nn.Parameter(torch.randn(max_objects, d_model))
        self.register_parameter('queries', params)

    def forward(self) -> Tensor:
        return self.queries


if __name__ == '__main__':
    oq = ObjectQueries()
    print(oq().size())
