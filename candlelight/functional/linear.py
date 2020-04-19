from typing import Tuple

import torch
import torch.nn.functional as F


def linear(
        input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    input_shape = input.shape
    input = 2 * (input.flatten() - domain[1]) / (domain[1] - domain[0]) + 1
    input = torch.stack((input, torch.zeros_like(input)), dim=-1)
    input = input.view(1, 1, -1, 2)
    value = value.view(1, 1, 1, -1)
    # noinspection PyArgumentList
    interpolation = F.grid_sample(
        value, input, padding_mode='border', align_corners=True
    )
    interpolation = interpolation.reshape(input_shape)
    return interpolation


def linear_g(
    input: torch.Tensor, value: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = value.numel() - 1
    input = input.clamp_(*domain)
    input = input.sub_(domain[0]).mul_(n / (domain[1] - domain[0]))
    left = input.floor().to(torch.int64).clamp_(max=n - 1)
    right = left + 1
    return value[right] * (input - left) + value[left] * (right - input)
