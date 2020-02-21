from typing import Tuple

import torch


def endpoint_slope(delta1: torch.Tensor, delta2: torch.Tensor) -> torch.Tensor:
    d = (3 * delta1 - delta2) / 2
    if torch.sign(d) != torch.sign(delta1):
        d = torch.zeros_like(delta1)
    elif torch.sign(delta1) != torch.sign(delta2) and torch.abs(d) > torch.abs(
            3 * delta1
    ):
        d = 3 * delta1
    return d


def akima1d(
        input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.size(0) - 1
    h = (domain[1] - domain[0]) / n
    m = (value[1:] - value[:-1]) / h
    for _ in range(2):
        m = torch.cat((2 * m[:1] - m[1:2], m, 2 * m[-1:] - m[-2:-1]))
    t = torch.abs(m[3:] - m[2:-1]) * m[1:-2] + torch.abs(m[1:-2] - m[:-3]) * m[2:-1]
    t /= torch.abs(m[3:] - m[2:-1]) + torch.abs(m[1:-2] - m[:-3]) + 1e-8
    interval = torch.clamp((input - domain[0]) // h, 0, n - 1).long()
    x = torch.linspace(domain[0], domain[1], n + 1, dtype=torch.float32)
    d = input - x[interval]
    d2 = torch.pow(d, 2)
    d3 = d2 * d
    p0 = value[interval]
    p1 = t[interval]
    p2 = (3 * m[2:-2][interval] - 2 * t[interval] - t[interval + 1]) / h
    p3 = (t[interval] + t[interval + 1] - 2 * m[2:-2][interval]) / h ** 2
    return p3 * d3 + p2 * d2 + p1 * d + p0
