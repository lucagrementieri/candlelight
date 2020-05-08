from typing import Tuple

import torch

from candlelight.vector_sampler import VectorSampler


def akima(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    eps = 1e-8
    n = value.size(0) - 1
    h = (domain[1] - domain[0]) / n
    m = (value[1:] - value[:-1]) / h
    m = torch.cat(
        (
            3 * m[:1] - 2 * m[1:2],
            2 * m[:1] - m[1:2],
            m,
            2 * m[-1:] - m[-2:-1],
            3 * m[-1:] - 2 * m[-2:-1],
        )
    )
    s = torch.abs(m[1:] - m[:-1])
    t = (s[2:] * m[1:-2] + s[:-2] * m[2:-1]) / (s[2:] + s[:-2] + eps)

    sampler = VectorSampler(input, domain, n)
    x = torch.linspace(
        domain[0], domain[1], n + 1, dtype=torch.float32, device=input.device
    )

    d = input - sampler.get_left(x)
    d2 = torch.pow(d, 2)
    d3 = d2 * d

    p0 = sampler.get_left(value)
    p1 = t_left = sampler.get_left(t)
    t_right = sampler.get_right(t)
    m_left = sampler.get_left(m[2:-1])
    p2 = (3 * m_left - 2 * t_left - t_right) / h
    p3 = (p1 + t_right - 2 * m_left) / h ** 2
    r = p3 * d3 + p2 * d2 + p1 * d + p0
    return r
