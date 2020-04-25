from typing import Tuple

import torch
import torch.nn.functional as F


def akima(
        input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    eps = 1e-6
    n = value.size(0) - 1
    h = (domain[1] - domain[0]) / n
    m = (value[1:] - value[:-1]) / h
    for _ in range(2):
        m = torch.cat((2 * m[:1] - m[1:2], m, 2 * m[-1:] - m[-2:-1]))
    t = torch.abs(m[3:] - m[2:-1]) * m[1:-2] + torch.abs(m[1:-2] - m[:-3]) * m[2:-1]
    t /= torch.abs(m[3:] - m[2:-1]) + torch.abs(m[1:-2] - m[:-3]) + eps

    input_shape = input.shape
    p = 2 * (input.flatten() - domain[1]) / (domain[1] - domain[0]) + 1
    d = (torch.frac((p + 1) / 2 * n)) * h

    p = torch.clamp(p, -1 + eps, 1 - eps)
    p -= 1 / n
    p = torch.stack((p, torch.zeros_like(p)), dim=-1)
    p = p.view(1, 1, -1, 2)

    d2 = torch.pow(d, 2)
    d3 = d2 * d

    p0 = F.grid_sample(
        value.view(1, 1, 1, -1),
        p + eps,
        mode='nearest',
        padding_mode='border',
        align_corners=True,
    ).view(-1)
    p1 = F.grid_sample(
        t.view(1, 1, 1, -1),
        p,
        mode='nearest',
        padding_mode='border',
        align_corners=True,
    ).view(-1)
    right_t = F.grid_sample(
        t.view(1, 1, 1, -1),
        p + 2 / n,
        mode='nearest',
        padding_mode='border',
        align_corners=True,
    ).view(-1)
    left_m = F.grid_sample(
        m[2:-1].view(1, 1, 1, -1),
        p,
        mode='nearest',
        padding_mode='border',
        align_corners=True,
    ).view(-1)
    p2 = (3 * left_m - 2 * p1 - right_t) / h
    p3 = (p1 + right_t - 2 * left_m) / h ** 2
    r = p3 * d3 + p2 * d2 + p1 * d + p0
    r = r.reshape(input_shape)
    return r
