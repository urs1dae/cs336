import torch
import torch.nn as nn
import einx
import math
from jaxtyping import Float, Int, Bool

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            )
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a = - 3 * std,
            b = 3 * std
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        return einx.dot("... [d_in], d_out [d_in -> ... d_out", x, self.weight)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
            )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = 1,
            a = - 3,
            b = 3
        )

    def forward(
        self,
        token_ids: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.weight[token_ids]


class RmsNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device = device, dtype=dtype))

    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_out = x * self.gain / rms
        return x_out.to(in_dtype)


def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        d_ff = d_ff if d_ff else (int(8 / 3 * d_model) + 63) & ~63
        self.weight1 = Linear(d_model, d_ff, device, dtype)
        self.weight2 = Linear(d_ff, d_model, device, dtype)
        self.weight3 = Linear(d_model, d_ff, device, dtype)

    def  forward(
        self,
        x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.weight2(silu(self.weight1(x)) * self.weight3(x))


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        pos = torch.arange(max_seq_len, device=device)
        thetas = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        freqs = torch.outer(pos, thetas)

        self.register_buffer('cos', torch.cos(freqs), persistent=False)
        self.register_buffer('sin', torch.sin(freqs), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x0 = x[..., ::2]
        x1 = x[..., 1::2]

        x0_rot = cos * x0 - sin * x1
        x1_rot = sin * x0 + cos * x1

        x_out = torch.stack((x0_rot, x1_rot), dim=-1).flatten(-2)

        return x_out


def softmax(
    x: Float[torch.Tensor, "..."],
    dim: int
) -> Float[torch.Tensor, "..."]:
    max_element = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - max_element)
    x_out = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return x_out


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch_size ... seq_len_q d_k"],
    K: Float[torch.Tensor, "batch_size ... seq_len_k d_k"],
    V: Float[torch.Tensor, "batch_size ... seq_len_k d_v"],
    mask: Bool[torch.Tensor, "seq_len_q seq_len_k"]
) -> Float[torch.Tensor, "batch_size ... seq_len_q d_v"]:
    d_k = K.shape[-1]
    S = einx.dot(
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k",
        Q,
        K
    ) / math.sqrt(d_k)
    A = softmax(
        S.masked_fill(~mask, float('-inf')),
        dim=-1
    )
    O = einx.dot(
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
        A,
        V
    )
    return O
