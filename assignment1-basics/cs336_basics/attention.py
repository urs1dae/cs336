import math

import einx
import torch
from jaxtyping import Bool, Float, Int
from torch import nn

from .nn_utils import softmax


class RoPE(nn.Module):
    """Rotary positional embedding (RoPE) applied to query/key channels.

    Inputs:
        ``in_features`` with shape ``(..., seq_len, d_k)`` and
        ``token_positions`` with shape ``(..., seq_len)``.
    Outputs:
        Tensor with shape ``(..., seq_len, d_k)``.
    Formula:
        For channel pair ``(2i, 2i+1)`` and position ``p``:
        ``theta_i = theta^(-2i/d_k)``,
        ``[x'_{2i}, x'_{2i+1}]^T = [[cos(p theta_i), -sin(p theta_i)], [sin(p theta_i), cos(p theta_i)]] [x_{2i}, x_{2i+1}]^T``.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        pos = torch.arange(max_seq_len, device=device)
        thetas = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        freqs = torch.outer(pos, thetas)

        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(
        self,
        in_features: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """Rotate each even-odd channel pair according to token position."""
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x0 = in_features[..., ::2]
        x1 = in_features[..., 1::2]

        x0_rot = cos * x0 - sin * x1
        x1_rot = sin * x0 + cos * x1

        out_features = torch.stack((x0_rot, x1_rot), dim=-1).flatten(-2)

        return out_features


def get_alibi_slopes(
    num_heads: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    def slopes_power_of_2(n: int) -> torch.Tensor:
        start = 2 ** (-2 ** (-(math.log2(n) - 3)))
        exponents = torch.arange(1, n + 1, device=device, dtype=dtype)
        return torch.pow(torch.tensor(start, device=device, dtype=dtype), exponents)

    if (num_heads & (num_heads - 1)) == 0:
        return slopes_power_of_2(num_heads)

    n2 = 2 ** math.floor(math.log2(num_heads))
    slopes_1 = slopes_power_of_2(n2)
    slopes_2 = slopes_power_of_2(2 * n2)[0::2][: (num_heads - n2)]
    return torch.cat([slopes_1, slopes_2], dim=0)


class Alibi(nn.Module):
    def __init__(
        self,
        num_heads: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        slopes = get_alibi_slopes(num_heads, device, dtype).view(num_heads, 1, 1)
        pos = torch.arange(0, max_seq_len, device=device)
        rel_pos = pos[None, :] - pos[:, None]
        bias = slopes * rel_pos[None, :, :]

        self.register_buffer("bias", bias, persistent=False)

    def forward(
        self,
        scores: Float[torch.Tensor, "... num_heads seq_len_q seq_len_k"],
    ):
        seq_len_q, seq_len_k = scores.shape[-2], scores.shape[-1]
        return scores + self.bias[None, :, :seq_len_q, :seq_len_k]


class Yarn(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        original_max_seq_len: int,
        scale: float = 4.0,
        mscale: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()

        inv_freq = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))

        self.original_max_seq_len = float(original_max_seq_len)
        self.scale = scale
        self.mscale = mscale
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        in_features: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """Rotate each even-odd channel pair according to token position."""
        p = token_positions
        o = self.original_max_seq_len
        pos = torch.where(p <= o, p, o + (p - o) / self.scale)
        freqs = pos.unsqueeze(-1) * self.inv_freq

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        x0 = in_features[..., ::2]
        x1 = in_features[..., 1::2]

        x0_rot = cos * x0 - sin * x1
        x1_rot = sin * x0 + cos * x1

        out_features = torch.stack((x0_rot, x1_rot), dim=-1).flatten(-2)

        return out_features * math.sqrt(self.mscale)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... seq_len_q d_k"],
    K: Float[torch.Tensor, "... seq_len_k d_k"],
    V: Float[torch.Tensor, "... seq_len_k d_v"],
    mask: Bool[torch.Tensor, "seq_len_q seq_len_k"] | None = None,
) -> Float[torch.Tensor, "... seq_len_q d_v"]:
    # Compute masked attention: softmax(QK^T / sqrt(d_k)) V.
    d_k = K.shape[-1]
    S = einx.dot(
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k",
        Q,
        K,
    ) / math.sqrt(d_k)
    if mask is not None:
        A = softmax(
            S.masked_fill(~mask, float("-inf")),
            dim=-1,
        )
    else:
        A = softmax(S, dim=-1)
    O = einx.dot(
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
        A,
        V,
    )
    return O


class CausalMaskMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention without positional rotation.

    Inputs:
        ``in_features`` with shape ``(..., seq_len, d_model)``.
    Outputs:
        Tensor with shape ``(..., seq_len, d_model)``.
    Formula:
        ``head_h = softmax((Q_h K_h^T)/sqrt(d_k) + M_causal) V_h``,
        ``output = Concat(head_1, ..., head_H) W_O^T``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj_weight = nn.Parameter(
            torch.empty((3, d_model, d_model), device=device, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )

        # Match Linear init style for all projection matrices.
        proj_std = (2 / (d_model + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.qkv_proj_weight,
            mean=0.0,
            std=proj_std,
            a=-3 * proj_std,
            b=3 * proj_std,
        )
        torch.nn.init.trunc_normal_(
            self.o_proj_weight,
            mean=0.0,
            std=proj_std,
            a=-3 * proj_std,
            b=3 * proj_std,
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        """Run causal self-attention with packed QKV projections for all heads."""
        seq_len = in_features.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool).tril()

        Q, K, V = einx.dot(
            "... seq_len d_model, three (num_heads d_k) d_model -> three ... num_heads seq_len d_k",
            in_features,
            self.qkv_proj_weight,
            num_heads=self.num_heads,
        )

        O = scaled_dot_product_attention(Q, K, V, mask)
        O = einx.rearrange(
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
            O,
        )

        out_features = einx.dot(
            "... [d_model], d_out [d_model] -> ... d_out",
            O,
            self.o_proj_weight,
        )
        return out_features


class CausalMaskMultiHeadSelfAttentionWithRope(nn.Module):
    """Causal multi-head self-attention with RoPE on queries and keys.

    Inputs:
        ``in_features`` with shape ``(..., seq_len, d_model)`` and
        ``token_positions`` with shape ``(..., seq_len)``.
    Outputs:
        Tensor with shape ``(..., seq_len, d_model)``.
    Formula:
        ``Q_h, K_h = RoPE(Q_h), RoPE(K_h)``,
        ``head_h = softmax((Q_h K_h^T)/sqrt(d_k) + M_causal) V_h``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj_weight = nn.Parameter(
            torch.empty((3, d_model, d_model), device=device, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )

        # Match Linear init style for all projection matrices.
        proj_std = (2 / (d_model + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.qkv_proj_weight,
            mean=0.0,
            std=proj_std,
            a=-3 * proj_std,
            b=3 * proj_std,
        )
        torch.nn.init.trunc_normal_(
            self.o_proj_weight,
            mean=0.0,
            std=proj_std,
            a=-3 * proj_std,
            b=3 * proj_std,
        )

        self.rope = RoPE(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len_q d_v"]:
        """Apply RoPE to Q/K and then run standard causal multi-head attention."""
        seq_len = in_features.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool).tril()

        Q, K, V = einx.dot(
            "... seq_len d_model, three (num_heads d_k) d_model -> three ... num_heads seq_len d_k",
            in_features,
            self.qkv_proj_weight,
            num_heads=self.num_heads,
        )

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        O = scaled_dot_product_attention(Q, K, V, mask)
        O = einx.rearrange(
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
            O,
        )

        out_features = einx.dot(
            "... [d_model], d_out [d_model] -> ... d_out",
            O,
            self.o_proj_weight,
        )
        return out_features
