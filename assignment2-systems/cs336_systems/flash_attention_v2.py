import torch
import triton
import triton.language as tl
from jaxtyping import Float


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["N", "D"]
)
@triton.jit
def flash_attention_forward_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    B, H, N, D,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # IS_CAUSAL: tl.constexpr
):
    pass


@triton.jit
def flash_attention_backward_kernel(
    
):
    pass


def flash_attention_forward_launch(
    Q: Float[torch.Tensor, "b h n d"],
    K: Float[torch.Tensor, "b h n d"],
    V: Float[torch.Tensor, "b h n d"],
    is_causal: bool = False
):
    B, H, N, D = Q.shape
    O = torch.zeros((B, H, N, D))
    L = torch.zeros((B, H, N))
    
    stride_qb, stride_qh, stride_qn, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = V.stride()
    stride_ob, stride_oh, stride_on, stride_od = O.stride()
    stride_lb, stride_lh, stride_ln = L.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H, 1)
    
    flash_attention_forward_kernel[grid](
        Q, K, V, O, L,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_lb, stride_lh, stride_ln,
    )


def flash_attention_backward_launch(
    
):
    pass


class FlashAttentionV2(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return super().forward(*args, **kwargs)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        return super().backward(ctx, *grad_outputs)