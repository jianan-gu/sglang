import torch
import sgl_kernel
import  torchao
from torchao.quantization.quant_primitives import (
    MappingType,
)
from torchao.dtypes import to_affine_quantized_intx
from torchao.quantization.utils import _get_per_token_block_size
torch.manual_seed(1234)
def _int8_symm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    mapping_type = MappingType.SYMMETRIC
    target_dtype = torch.float8_e4m3fn
    eps = torch.finfo(torch.float32).eps
    quant_min = int(torch.finfo(torch.float8_e4m3fn).min)
    quant_max = int(torch.finfo(torch.float8_e4m3fn).min)
    # print(_get_per_token_block_size(x))
    res =  to_affine_quantized_intx(
        x,
        mapping_type,
        _get_per_token_block_size(x),
        target_dtype,
        eps=eps,
        quant_min=quant_min,
        quant_max=quant_max,
        scale_dtype=torch.float,
    )
    # return res.tensor_impl.get_plain()
    act = res.tensor_impl.int_data
    act_scales = res.tensor_impl.scale
    act_qzeros = res.tensor_impl.zero_point
    return act, act_scales, act_qzeros
def _uint8_asymm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.uint8
    scale_dtype = torch.bfloat16
    eps = torch.finfo(torch.bfloat16).eps
    zero_point_dtype = torch.int32
    quant_min = 0
    quant_max = 255
    if True:
        out = to_affine_quantized_intx(
            x,
            mapping_type,
            _get_per_token_block_size(x),
            target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            scale_dtype=scale_dtype,
            zero_point_dtype=zero_point_dtype,
        )
    else:
        out = to_affine_quantized_intx(
            x,
            mapping_type,
            _get_per_token_block_size(x),
            target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
        )
    # return out.tensor_impl.get_plain()
    act = out.tensor_impl.int_data
    act_scales = out.tensor_impl.scale
    act_qzeros = out.tensor_impl.zero_point
    return act, act_scales, act_qzeros


def unpack_awq_weight(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        awq_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features // (32 // bits))
        awq_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features // (32 // bits))
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        unpacked awq_qweight (`torch.LongTensor`):
            With shape (in_features, out_features).
        zeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features).
    """
    assert bits == 4

    qzeros = awq_qzeros
    qweight = awq_qweight
    qweight = qweight.T.contiguous()

    scales = awq_scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    infeatures = awq_qweight.shape[0]

    wf = torch.tensor(
        list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device
    ).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
        torch.int16 if bits == 8 else torch.int8
    )

    # zeros = zeros + 1

    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)

    weight = weight.view(-1, weight.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])

    zeros = zeros.T.contiguous()
    zeros = awq_reverse_reorder_int_tensor(zeros, bits)
    weight = awq_reverse_reorder_int_tensor(weight, bits)
    
    return weight.contiguous(), zeros.contiguous()

def awq_reverse_reorder_int_tensor(int_tensor, bits: int):
    assert bits == 4

    int_tensor = int_tensor.T.contiguous()
    compress_ratio = 32 // bits
    assert int_tensor.shape[-1] % compress_ratio == 0

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    order_tensor = torch.tensor(
        order_map, dtype=torch.int32, device=int_tensor.device
    ).reshape(1, -1)
    order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)
    order_tensor = order_tensor + torch.arange(
        0,
        int_tensor.shape[1],
        compress_ratio,
        dtype=torch.int32,
        device=int_tensor.device,
    ).reshape(-1, 1)
    order_tensor = order_tensor.reshape(-1)

    reverse_order_tensor = torch.arange(order_tensor.shape[0])[order_tensor]
    reverse_order_tensor = reverse_order_tensor[order_tensor]
    int_tensor = int_tensor[:, reverse_order_tensor]
    return int_tensor

def _autoawq_to_int4pack_w4a8(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int=4,
    group_size: int=128,
):

    t, zp = unpack_awq_weight(awq_qweight, awq_qzeros, awq_scales, bits, group_size)
    # # # # transpose -> [N, K]
    # t = t.T.contiguous()
    awq_qweight_2 = t.T.contiguous().to(torch.uint8)

    # qweight_ = t[:, 1::2].bitwise_left_shift(4).bitwise_or_(t[:, ::2]).to(torch.uint8)
    scales_ = awq_scales.t().contiguous()
    awq_qzeros = zp.t_().contiguous()
    # print(qweight_.shape)
    # print(scales_.shape)
    # print(zp_.shape)
    
    # bitshifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32) * 4
    # awq_qweight = (awq_qweight.unsqueeze(-1) >> bitshifts) & 0xF
    # awq_qweight = awq_qweight.flatten(-2).transpose(-1, -2).to(torch.uint8)
    # awq_qzeros = (awq_qzeros.unsqueeze(-1) >> bitshifts) & 0xF
    # awq_qzeros = awq_qzeros.flatten(-2).to(torch.uint8)
    # awq_qzeros = awq_qzeros.T.contiguous()
    # print(awq_qweight.shape)
    # print(awq_qzeros.shape)
    # print(awq_qweight_2 == awq_qweight)
    # print(zp.T.contiguous() == awq_qzeros)
    qweight_, scales_, zp_ , comp = sgl_kernel.common_ops.da8w4_linear_prepack_cpu(awq_qweight_2, scales_, awq_qzeros)
    # print(qweight_.shape)
    # print(scales_.shape)
    # print(zp_.shape)
    # print(comp.shape)
    return qweight_,  zp_, scales_, comp

def _autoawq_to_int4pack_w4a82(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int=4,
    group_size: int=128,
):

    # t, zp = unpack_awq_weight(awq_qweight, awq_qzeros, awq_scales, bits, group_size)
    # # # # transpose -> [N, K]
    # t = t.T.contiguous()
    # awq_qweight_2 = t.T.contiguous().to(torch.uint8)
    # qweight_ = t[:, 1::2].bitwise_left_shift(4).bitwise_or_(t[:, ::2]).to(torch.uint8)
    scales_ = awq_scales.t().contiguous()
    # awq_qzeros = zp.T.contiguous()
    # print(qweight_.shape)
    # print(scales_.shape)
    # print(zp_.shape)
    
    bitshifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32) * 4
    awq_qweight = (awq_qweight.unsqueeze(-1) >> bitshifts) & 0xF
    awq_qweight = awq_qweight.flatten(-2).transpose(-1, -2).to(torch.uint8)
    # awq_qzeros = (awq_qzeros.unsqueeze(-1) >> bitshifts) & 0xF
    # awq_qzeros = awq_qzeros.flatten(-2).to(torch.uint8)
    # awq_qzeros = awq_qzeros.T.contiguous()
    # breakpoint()
    def _convert_awq_scales_qzeros(scales, qzeros, bits=4):
        new_scales = scales.t().contiguous()
        wf = torch.tensor(
            list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device
        ).unsqueeze(0)
        zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
            torch.int16 if bits == 8 else torch.int8
        )
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
        zeros = zeros.view(-1, zeros.shape[-1])

        zeros = zeros.T.contiguous()
        zeros = awq_reverse_reorder_int_tensor(zeros, bits)
        new_qzeros = zeros.T.contiguous()
        return new_scales, new_qzeros
    new_scales, new_qzeros = _convert_awq_scales_qzeros(awq_scales, awq_qzeros)
    # print(zp.T.contiguous() == awq_qzeros)
    qweight_, scales_, zp_ , comp = sgl_kernel.common_ops.da8w4_linear_prepack_cpu(awq_qweight, new_scales, new_qzeros)
    # print(qweight_.shape)
    # print(scales_.shape)
    # print(zp_.shape)
    # print(comp.shape)
    return qweight_,  zp_, scales_, comp


def unpack_awq(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        awq_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features // (32 // bits))
        awq_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features // (32 // bits))
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        fp16_weight (`torch.LongTensor`):
            With shape (in_features, out_features).
        zeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features).
    """
    assert bits == 4

    qzeros = awq_qzeros
    qweight = awq_qweight
    qweight = qweight.T.contiguous()

    scales = awq_scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    infeatures = awq_qweight.shape[0]

    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
        torch.int16 if bits == 8 else torch.int8
    )

    # zeros = zeros + 1

    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)).to(
        torch.int16 if bits == 8 else torch.int8
    )
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)
    weight = weight.reshape(-1, group_size, weight.shape[2])

    weight = weight.view(-1, weight.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])

    zeros = zeros.T.contiguous()
    zeros = awq_reverse_reorder_int_tensor(zeros, bits)
    weight = awq_reverse_reorder_int_tensor(weight, bits)
    breakpoint()
    # Dequantize weights.
    scales = awq_scales
    zeros = zeros.contiguous()
    scale_zeros = zeros * scales

    g_idx = torch.tensor([i // group_size for i in range(infeatures)], dtype=torch.int32)
    scale_mat = scales[g_idx]
    scale_zeros_mat = scale_zeros[g_idx].to(torch.bfloat16)

    qdq_weight_T = weight * scale_mat - scale_zeros_mat.to(torch.bfloat16)

    fp16_weight = qdq_weight_T.T

    return fp16_weight, zeros
import copy
awq_weight = torch.load("awq_weight.pt")
awq_qzeros = torch.load("awq_qzeros.pt")
# awq_qzeros = torch.randint(0, 15, awq_qzeros.shape)
awq_scales = torch.load("awq_scales.pt")
# awq_scales = torch.ones_like(awq_scales)

awq_weight_ = torch.load("awq_weight.pt")
awq_qzeros_ = torch.load("awq_qzeros.pt")
# awq_qzeros_ = copy.deepcopy(awq_qzeros)# torch.zeros_like(awq_qzeros_)
awq_scales_ = torch.load("awq_scales.pt")
# awq_scales_ = torch.ones_like(awq_scales_)

awq_weight_2 = torch.load("awq_weight.pt")
awq_qzeros_2 = torch.load("awq_qzeros.pt")
# awq_qzeros_2 = copy.deepcopy(awq_qzeros)#torch.zeros_like(awq_qzeros_2)
awq_scales_2 = torch.load("awq_scales.pt")
# awq_scales_2 = torch.ones_like(awq_scales_2)
awq_weight_3 = torch.load("awq_weight.pt")
awq_qzeros_3 = torch.load("awq_qzeros.pt")
# awq_qzeros_2 = copy.deepcopy(awq_qzeros)#torch.zeros_like(awq_qzeros_2)
awq_scales_3 = torch.load("awq_scales.pt")
bf16_weight, zero =  unpack_awq(awq_weight_, awq_qzeros_, awq_scales_,4,128)


bias = torch.zeros(1, bf16_weight.shape[0]).to(torch.bfloat16)

x = torch.rand(1, bf16_weight.size(-1)).to(torch.bfloat16)
x_q, x_s =  torch.ops.sgl_kernel.per_token_quant_int8_cpu_sym(x)
breakpoint()
dummy_zeros = torch.zeros_like(x_s).to(torch.int)
# x_q, x_s.to(torch.float32), dummy_zeros =  _uint8_asymm_per_token_quant(x)

# xref = ((x_q-dummy_zeros)*x_s).to(torch.bfloat16)
xref = ((x_q)*x_s).to(torch.bfloat16)
ref_res = torch.nn.functional.linear(xref, bf16_weight, bias= bias)


qweight_,  zp_, scales_, comp =  _autoawq_to_int4pack_w4a8(awq_weight, awq_qzeros, awq_scales,4,128)
# print(qweight_.shape)
bias1 = torch.zeros(1, bf16_weight.shape[0])

res = sgl_kernel.common_ops.da8w4_linear_cpu(
    x_q, x_s, dummy_zeros, qweight_, scales_, zp_, comp, bias1, torch.bfloat16
)
    # y = torch.ops.torchao.da8w4_linear_cpu.default(
    #     act.contiguous(),
    #     act_scales,
    #     act_qzeros,
    #     packed_weight,
    #     wei_scales,
    #     wei_qzeros,
    #     compensation,
    #     bias.float() if bias is not None else bias,  # requires bias to be float
    #     orig_dtype,  # out_dtype
    # )

qweight_2,  zp_2, scales_2, comp2 =  _autoawq_to_int4pack_w4a82(awq_weight_2, awq_qzeros_2, awq_scales_2,4,128)
res2= torch.ops.torchao.da8w4_linear_cpu(
    x_q, x_s, dummy_zeros, qweight_2, scales_2, zp_2, comp2, bias1, torch.bfloat16
)

import intel_extension_for_pytorch as ipex
lowp_mode = ipex.quantization.WoqLowpMode.INT8
# The weight will be de-packed from INT4 to INT8.
weight_dtype = ipex.quantization.WoqWeightDtype.INT4
# The float activation will be quantized (dynamic, per-token) to INT8.
act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK

qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
    weight_dtype=weight_dtype,
    lowp_mode=lowp_mode,
    act_quant_mode=act_quant_mode,
    group_size=128,
)

ipex_qlinear = ipex.llm.quantization.woq_linear.IPEXWeightOnlyQuantizedLinear.from_weight(
    awq_weight_3,
    awq_scales_3,
    awq_qzeros_3,
    bf16_weight.shape[0],
    bf16_weight.shape[-1],
    qconfig=qconfig,
    g_idx=None,
    bias=bias,
    group_size=128,
    quant_method=1)

res_ipex = ipex_qlinear(x)
print(res)
print(res2)
print(ref_res)
print(res_ipex)
breakpoint()



# q_weight = torch.randint(0,15, bf16_weight.shape).to(torch.uint8)
# q_scale = torch.rand(0,15, ).to(torch.uint8)
# bias = torch.zeros(1, bf16_weight.shape[0]).to(torch.bfloat16)