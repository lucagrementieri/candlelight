from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from candlelight.functional import linear


class Linear(nn.Module):
    def __init__(self, nodes: int, domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        default_values = F.relu(
            torch.linspace(domain[0], domain[1], nodes, dtype=torch.float32)
        )
        default_values = torch.tensor(
            [
                -0.0277,
                -0.2972,
                0.0717,
                -0.5302,
                -0.1691,
                -0.0635,
                0.6766,
                1.1537,
                2.2375,
                3.5700,
                4.5641,
            ]
        )
        self.value = nn.Parameter(default_values, requires_grad=False)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear(input, self.value, self.domain)
