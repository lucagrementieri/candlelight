from typing import Tuple

import torch


def barycentric1d(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.size(0) - 1
    inverse_factorial = torch.exp(
        -torch.lgamma(torch.arange(n + 1, dtype=torch.float64) + 1)
    ).float()
    w = inverse_factorial * inverse_factorial.flip(dims=(0,))
    w[::2] *= -1
    x = torch.linspace(domain[0], domain[1], n + 1, dtype=torch.float32)
    d = input.unsqueeze(dim=-1) - x + 1e-8
    c = w / d
    return torch.matmul(c, value) / torch.sum(c, dim=-1)
