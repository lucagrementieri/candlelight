from typing import Tuple

import torch
import torch.nn.functional as F


class VectorSampler:
    def __init__(
        self,
        position: torch.Tensor,
        domain: Tuple[float, float],
        n: int,
        eps: float = 1e-8,
    ):
        p = 2 * (position.flatten() - domain[1]) / (domain[1] - domain[0]) + 1
        p = torch.clamp(p, -1 + eps, 1 - eps)
        p -= 0.99 / n
        p = torch.stack((p, torch.zeros_like(p)), dim=-1)
        self.left_p = p.view(1, 1, -1, 2)
        self.right_p = self.left_p.clone()
        self.right_p[..., 0] += 1.98 / n
        self.p_shape = position.shape

    def get_left(self, vector: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            vector.view(1, 1, 1, -1),
            self.left_p,
            mode='nearest',
            padding_mode='border',
            align_corners=True,
        ).view(self.p_shape)

    def get_right(self, vector: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            vector.view(1, 1, 1, -1),
            self.right_p,
            mode='nearest',
            padding_mode='border',
            align_corners=True,
        ).view(self.p_shape)
