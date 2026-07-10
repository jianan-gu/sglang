import torch

precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}


def make_non_contiguous(x: torch.Tensor) -> torch.Tensor:
    # Make a tensor non-contiguous without changing shape.
    if not x.is_contiguous():
        return x

    last_dim = x.shape[-1]
    expanded = torch.empty(*x.shape[:-1], last_dim + 32, dtype=x.dtype, device=x.device)
    expanded[..., :last_dim].copy_(x)
    return expanded.narrow(-1, 0, last_dim)
