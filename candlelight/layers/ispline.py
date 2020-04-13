from typing import Tuple

import torch
import torch.nn as nn

from candlelight.functional import ispline1d


class Ispline1D(nn.Module):
    def __init__(self, nodes: int, domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(nodes), requires_grad=True)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ispline1d(input, self.weight, self.domain)
