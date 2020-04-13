import torch


def akima(
    input: torch.Tensor, node: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    n = value.size(0) - 1
    h = (node[-1] - node[0]) / n
    m = (value[1:] - value[:-1]) / h
    for _ in range(2):
        m = torch.cat((2 * m[:1] - m[1:2], m, 2 * m[-1:] - m[-2:-1]))
    t = torch.abs(m[3:] - m[2:-1]) * m[1:-2] + torch.abs(m[1:-2] - m[:-3]) * m[2:-1]
    t /= torch.abs(m[3:] - m[2:-1]) + torch.abs(m[1:-2] - m[:-3]) + 1e-8
    interval = torch.clamp((input - node[0]) // h, 0, n - 1).long()
    d = input - node[interval]
    d2 = torch.pow(d, 2)
    d3 = d2 * d
    p0 = value[interval]
    p1 = t[interval]
    p2 = (3 * m[2:-2][interval] - 2 * t[interval] - t[interval + 1]) / h
    p3 = (t[interval] + t[interval + 1] - 2 * m[2:-2][interval]) / h ** 2
    return p3 * d3 + p2 * d2 + p1 * d + p0
