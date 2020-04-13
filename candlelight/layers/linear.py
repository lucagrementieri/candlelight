from typing import Tuple

import torch
import torch.nn as nn

from candlelight.functional import akima


class Linear(nn.Module):
    def __init__(self, nodes: int, domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        default_values = torch.linspace(
            domain[0], domain[1], nodes, dtype=torch.float32
        )
        self.value = nn.Parameter(default_values, requires_grad=True)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return akima(input, self.value, self.domain)
