from typing import Tuple

import torch

from candlelight.vector_sampler import VectorSampler


def cubic(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.size(0) - 1
    h = (domain[1] - domain[0]) / n
    A = torch.eye(n + 1) + torch.diagflat(torch.full((n,), 0.5), 1)
    A += A.T
    A[0, 1] = A[-1, -2] = 0
    d = 3 * (value[2:] - 2 * value[1:-1] + value[:-2]) / h ** 2
    d = torch.cat((torch.zeros(1), d, torch.zeros(1))).unsqueeze_(-1)
    z, _ = torch.solve(d, A)

    sampler = VectorSampler(input, domain, n)
    x = torch.linspace(
        domain[0], domain[1], n + 1, dtype=torch.float32, device=input.device
    )
    distance_left = input - sampler.get_left(x)
    distance_right = h - distance_left
    cubic_left = torch.pow(distance_left, 3)
    cubic_right = torch.pow(distance_right, 3)

    z_left = sampler.get_left(z)
    z_right = sampler.get_right(z)
    value_left = sampler.get_left(value)
    value_right = sampler.get_right(value)

    f = z_left * cubic_right + z_right * cubic_left
    f /= 6 * h
    f += (value_right / h - z_right * h / 6) * distance_left
    f += (value_left / h - z_left * h / 6) * distance_right
    return f
