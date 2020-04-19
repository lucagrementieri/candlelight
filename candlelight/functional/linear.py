from typing import Tuple

import torch


def linear(
        input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.numel() - 1
    input = input.clamp_(*domain)
    input = input.sub_(domain[0]).mul_(n / (domain[1] - domain[0]))
    left = input.floor().to(torch.int64).clamp_(max=n - 1)
    right = left + 1
    return value[right] * (input - left) + value[left] * (right - input)
