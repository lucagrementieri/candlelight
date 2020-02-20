from typing import Tuple

import torch


def lagrange1d(
    input: torch.Tensor, node: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = node.size(0) - 1
    inverse_factorial = torch.exp(
        -torch.lgamma(torch.arange(1, n + 2, dtype=torch.float64))
    )
    a = (
        pow(n / (domain[1] - domain[0]), n)
        * inverse_factorial
        * inverse_factorial.flip(dims=(0,))
    ).float()
    a[(n + 1) % 2 :: 2] *= -1
    c = a * node
    x = torch.linspace(domain[0], domain[1], n + 1, dtype=torch.float32)
    x = x.view(*(1,) * input.ndim, -1)
    d = torch.unsqueeze(input.unsqueeze(dim=-1) - x, dim=-2)
    d = d.repeat(*(1,) * input.ndim, n + 1, 1)
    d[..., torch.arange(n + 1), torch.arange(n + 1)] = 1
    d = d.prod(dim=-1)
    return torch.matmul(d, c)
