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
                -4.4584,
                -4.6660,
                -4.4053,
                -4.2198,
                -4.0385,
                -3.8759,
                -3.6479,
                -3.4513,
                -3.3151,
                -3.0596,
                -2.8994,
                -2.7457,
                -2.5312,
                -2.2760,
                -2.0497,
                -1.6593,
                -1.4961,
                -1.1880,
                -0.8908,
                -0.4667,
                -0.1091,
                0.0292,
                -0.0997,
                -0.0926,
                -0.0929,
                -0.0926,
                -0.1071,
                0.1248,
                0.6232,
                0.7126,
                0.7976,
                0.7934,
                0.8795,
                1.1539,
                1.4688,
                1.7880,
                1.9781,
                2.2734,
                2.5303,
                2.7463,
                2.9187,
                3.0851,
                3.2771,
                3.5125,
                3.6194,
                3.8880,
                4.0728,
                4.2732,
                4.4323,
                4.6388,
                4.5455,
            ]
        )
        self.value = nn.Parameter(default_values, requires_grad=False)
        self.domain = domain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear(input, self.value, self.domain)
