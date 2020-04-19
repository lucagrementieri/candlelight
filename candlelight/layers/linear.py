from typing import Tuple

import torch
import torch.nn as nn

from candlelight.functional import linear


class Linear(nn.Module):
    def __init__(self, nodes: int, domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        default_values = torch.linspace(
            domain[0], domain[1], nodes, dtype=torch.float32
        )
        default_values = torch.tensor(
            [
                -4.0498,
                -3.5387,
                -2.5938,
                -1.2334,
                -0.0649,
                -0.1820,
                0.9610,
                1.4660,
                2.7531,
                3.4607,
                3.9308,
            ]
        )
        self.value = nn.Parameter(default_values, requires_grad=False)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear(input, self.value, self.domain)
