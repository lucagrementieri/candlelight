from typing import Tuple

import torch
import torch.nn.functional as F


def cubic(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.size(0) - 1
    h = (domain[1] - domain[0]) / n
    A = torch.eye(n + 1) + torch.diagflat(torch.full((n,), 0.5), 1)
    A += A.T
    A[0, 1] = A[-1, -2] = 0
    d = 3 * (value[2:] - 2 * value[1:-1] + value[:-2]) / h ** 2
    d = F.pad(d, [1, 1]).unsqueeze_(-1)
    z, _ = torch.solve(d, A)
    z = z.squeeze_(-1)

    interval = torch.clamp((input - domain[0]) // h, 0, n - 1).long()
    x = torch.linspace(
        domain[0], domain[1], n + 1, dtype=torch.float32, device=input.device
    )
    distance_left = input - x[interval]
    distance_right = x[1 + interval] - input
    cubic_left = torch.pow(distance_left, 3)
    cubic_right = torch.pow(distance_right, 3)
    f = z[interval] * cubic_right + z[1 + interval] * cubic_left
    f /= 6 * h
    f += (value[1 + interval] / h - z[1 + interval] * h / 6) * distance_left
    f += (value[interval] / h - z[interval] * h / 6) * distance_right
    return f
