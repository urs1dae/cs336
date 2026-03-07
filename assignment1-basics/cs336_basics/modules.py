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
        in_features: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        return einx.dot("... [d_in], d_out [d_in] -> ... d_out", in_features, self.weight)


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
        in_features: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = in_features.dtype
        in_features = in_features.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + self.eps)
        out_features = in_features * self.gain / rms
        return out_features.to(in_dtype)


def silu(in_features: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return in_features * torch.sigmoid(in_features)


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
        in_features: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.weight2(silu(self.weight1(in_features)) * self.weight3(in_features))


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
        in_features: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x0 = in_features[..., ::2]
        x1 = in_features[..., 1::2]

        x0_rot = cos * x0 - sin * x1
        x1_rot = sin * x0 + cos * x1

        out_features = torch.stack((x0_rot, x1_rot), dim=-1).flatten(-2)

        return out_features


def softmax(
    in_features: Float[torch.Tensor, "..."],
    dim: int
) -> Float[torch.Tensor, "..."]:
    max_element = torch.max(in_features, dim=dim, keepdim=True).values
    x_exp = torch.exp(in_features - max_element)
    out_features = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return out_features


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


class CausalMaskMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj_weight = nn.Parameter(
            torch.empty((3, d_model, d_model), device=device, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        seq_len = in_features.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool).tril()

        Q, K, V = einx.dot(
            "... seq_len d_model, three (num_heads d_k) d_model -> three ... num_heads seq_len d_k",
            in_features,
            self.qkv_proj_weight,
            num_heads=self.num_heads
        )

        O = scaled_dot_product_attention(Q, K, V, mask)
        O = einx.rearrange(
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
            O
        )

        out_features = einx.dot(
            "... [d_model], d_out [d_model] -> ... d_out",
            O,
            self.o_proj_weight
        )
        return out_features



class CausalMaskMultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj_weight = nn.Parameter(
            torch.empty((3, d_model, d_model), device=device, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.rope = RoPE(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "batch_size ... seq_len_q d_v"]:
        seq_len = in_features.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool).tril()

        Q, K, V = einx.dot(
            "... seq_len d_model, three (num_heads d_k) d_model -> three ... num_heads seq_len d_k",
            in_features,
            self.qkv_proj_weight,
            num_heads=self.num_heads
        )

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        O = scaled_dot_product_attention(Q, K, V, mask)
        O = einx.rearrange(
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
            O
        )

        out_features = einx.dot(
            "... [d_model], d_out [d_model] -> ... d_out",
            O,
            self.o_proj_weight
        )
        return out_features


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        d_ff: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.rms1 = RmsNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.mha = CausalMaskMultiHeadSelfAttentionWithRope(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        self.rms2 = RmsNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... d_model"]:
        x_rms1 = self.rms1(in_features)
        atten_features = self.mha(x_rms1, token_positions) + in_features
        x_rms2 = self.rms2(atten_features)
        out_features = self.ffn(x_rms2) + atten_features
        return out_features


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        d_ff: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    theta=theta,
                    max_seq_len=max_seq_len,
                    d_ff=d_ff,
                    eps=eps,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )

        self.final_rms = RmsNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype
        )

        self.final_layer = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        tokens: Int[torch.Tensor, "... seq_len"],
    ):
        _, seq_len = tokens.shape
        token_positions = torch.arange(seq_len, device=tokens.device, dtype=torch.long)

        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.final_rms(x)
        logits = self.final_layer(x)

        return logits
