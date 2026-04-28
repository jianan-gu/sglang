from typing import Optional, Tuple

import torch


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization (PyTorch native).

    This is a pure PyTorch implementation equivalent to the Triton kernel version.
    It performs per-block FP8 quantization along the last dimension.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and
            its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks used for quantization.
            Default is 128.
        scale_fmt (Optional[str], optional): If not None, scales are rounded to
            powers of 2. Default is None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # FP8 e4m3fn constants
    fp8_max = 448.0
    fp8_min = -448.0

    # Flatten all dims except last
    orig_shape = x.shape
    N = x.size(-1)
    x_flat = x.view(-1, N).float()  # (M, N)
    M = x_flat.size(0)
    num_groups = N // block_size

    # Reshape into blocks: (M, num_groups, block_size)
    x_blocked = x_flat.view(M, num_groups, block_size)

    # Compute per-block absolute max -> (M, num_groups)
    amax = x_blocked.abs().amax(dim=2)

    # Clamp to avoid division by zero
    amax = amax.clamp(min=1e-4)

    # Compute scale
    round_scale = scale_fmt is not None
    if round_scale:
        # Round scale to nearest power of 2 (ceiling in log2 space)
        scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
    else:
        scale = amax / fp8_max

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    # scale shape: (M, num_groups) -> broadcast to (M, num_groups, block_size)
    y = x_blocked / scale.unsqueeze(2)
    y = y.clamp(fp8_min, fp8_max)

    # Reshape output back to original shape and cast to fp8
    y = y.view(orig_shape).to(torch.float8_e4m3fn)

    # Reshape scale to match expected output shape: (*orig_shape[:-1], num_groups)
    s = scale.view(*orig_shape[:-1], num_groups)

    return y, s


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale
