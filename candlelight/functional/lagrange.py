from typing import Tuple

import torch


def lagrange(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.size(0) - 1
    inverse_factorial = torch.exp(
        -torch.lgamma(torch.arange(n + 1, dtype=torch.float64) + 1)
    )
    a = (
        pow(n / (domain[1] - domain[0]), n)
        * inverse_factorial
        * inverse_factorial.flip(dims=(0,))
    )
    a[(n + 1) % 2 :: 2] *= -1
    c = a * value
    x = torch.linspace(
        domain[0], domain[1], n + 1, dtype=torch.float64, device=input.device
    )
    d = torch.unsqueeze(input.unsqueeze(dim=-1) - x, dim=-2)
    d = d.repeat(*(1,) * input.ndim, n + 1, 1)
    d[..., torch.arange(n + 1), torch.arange(n + 1)] = 1
    d = d.prod(dim=-1)
    return torch.matmul(d, c).float()
