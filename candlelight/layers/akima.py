from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from candlelight.functional import akima


class Akima(nn.Module):
    def __init__(self, nodes: int, domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        default_values = F.relu(torch.linspace(
            domain[0], domain[1], nodes, dtype=torch.float32
        ))
        self.value = nn.Parameter(default_values, requires_grad=True)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return akima(input, self.value, self.domain)
