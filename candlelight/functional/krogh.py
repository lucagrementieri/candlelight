from typing import Tuple

import torch


def krogh(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.size(0) - 1
    x = torch.linspace(
        domain[0], domain[1], n + 1, dtype=torch.float64, device=input.device
    )
    q = torch.cumprod(input.double().unsqueeze(dim=-1) - x, dim=-1)
    p = torch.full_like(input, value[0].item())
    for i in range(n):
        value[i + 1 :] -= value[i]
        value[i + 1 :] /= x[i + 1 :] - x[i]
        p += q[..., i] * value[i + 1]
    return p
