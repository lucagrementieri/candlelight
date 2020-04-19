from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from candlelight.functional import linear


class MonotoneLinear(nn.Module):
    def __init__(self, nodes: int, domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        default_values = torch.linspace(
            domain[0], domain[1], nodes, dtype=torch.float32
        )
        self.value = nn.Parameter(default_values, requires_grad=True)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        increments = F.relu(self.value[1:] - self.value[:-1]) + self.value[:-1]
        increments = torch.cat((self.value[[0]], increments))
        return linear(input, increments, self.domain)
