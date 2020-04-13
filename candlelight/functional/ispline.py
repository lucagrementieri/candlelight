from typing import Tuple

import torch


def ispline(
    input: torch.Tensor, weight: torch.Tensor, domain: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    n = weight.size(0) - 1
    h = (domain[1] - domain[0]) / n
    interval = torch.clamp((input - domain[0]) // h, 0, n - 1).long()
    x = torch.linspace(domain[0], domain[1], n + 1, dtype=torch.float32)
    weight_sum = torch.cumsum(weight, dim=0)
    y = weight_sum[interval]
    y_increment = weight[interval + 1] * torch.pow(x - x[interval], 2) / h ** 2
    y_increment[interval < n - 1] /= 2
    y += y_increment
    y_increment = -weight[interval] * torch.pow(x - x[interval + 1], 2) / h ** 2
    y_increment[interval > 0] /= 2
    y += y_increment
    return y
