from typing import Tuple

import torch


def barycentric(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    eps = 1e-16
    n = value.size(0) - 1
    w = -torch.lgamma(torch.arange(n + 1, dtype=torch.float64) + 1)
    w += w.flip(dims=(0,))
    w = w.exp_()
    w[::2] *= -1
    x = torch.linspace(
        domain[0], domain[1], n + 1, dtype=torch.float64, device=input.device
    )
    d = input.unsqueeze(dim=-1) - x
    d[torch.abs(d) < eps] = eps
    c = w / d
    r = torch.matmul(c, value.double()) / torch.sum(c, dim=-1)
    return r.float()
