from typing import Tuple

import torch
import torch.nn.functional as F

from candlelight.vector_sampler import VectorSampler


def endpoint_slope(delta1: torch.Tensor, delta2: torch.Tensor) -> torch.Tensor:
    d = (3 * delta1 - delta2) / 2
    if torch.sign(d) != torch.sign(delta1):
        d = torch.zeros_like(delta1)
    elif torch.sign(delta1) != torch.sign(delta2) and torch.abs(d) > torch.abs(
        3 * delta1
    ):
        d = 3 * delta1
    return d


def pchip(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.size(0) - 1
    h = (domain[1] - domain[0]) / n
    delta = (value[1:] - value[:-1]) / h
    d = torch.cat(
        (
            endpoint_slope(delta[:1], delta[1:2]),
            2 * delta[:-1] * delta[1:] / (delta[:-1] + delta[1:] + 1e-8),
            endpoint_slope(delta[-1:], delta[-2:-1]),
        )
    )
    d[1:-1] *= F.relu(torch.sign(delta[1:]) * torch.sign(delta[:-1]))

    sampler = VectorSampler(input, domain, n)
    x = torch.linspace(
        domain[0], domain[1], n + 1, dtype=torch.float32, device=input.device
    )
    distance_left = input - sampler.get_left(x)
    distance_right = distance_left - h
    squared_left = torch.pow(distance_left, 2)
    squared_right = torch.pow(distance_right, 2)
    value_left = sampler.get_left(value)
    value_right = sampler.get_right(value)
    d_left = sampler.get_left(d)
    d_right = sampler.get_right(d)
    f = value_left * squared_right * (distance_left + h / 2)
    f -= value_right * squared_left * (distance_right - h / 2)
    f *= 2 / h ** 3
    f += (
        d_left * squared_right * distance_left + d_right * squared_left * distance_right
    ) / h ** 2
    return f
