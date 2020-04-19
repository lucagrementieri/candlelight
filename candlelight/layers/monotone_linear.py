from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from candlelight.functional import linear


class MonotoneLinear(nn.Module):
    def __init__(self, nodes: int, domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        default_values = torch.full(
            (nodes,), (domain[1] - domain[0]) / (nodes - 1), dtype=torch.float32
        )
        default_values[0] = 0
        self.start = nn.Parameter(
            torch.tensor(domain[0], dtype=torch.float32), requires_grad=True
        )
        self.increments = nn.Parameter(default_values, requires_grad=True)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        increments = F.relu(self.increments)
        cumulative_value = torch.cumsum(increments, dim=0) + self.start
        return linear(input, cumulative_value, self.domain)
