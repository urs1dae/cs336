import os
import torch
from torch import nn
import einx
import math
import numpy as np
import numpy.typing as npt

from jaxtyping import Float, Int, Bool
from typing import Optional, Callable, Iterable, BinaryIO, IO

class Linear(nn.Module):
    """A bias-free linear map ``y = x W^T``.

    Inputs:
        ``in_features`` with shape ``(..., d_in)``.
    Outputs:
        Tensor with shape ``(..., d_out)``.
    Formula:
        ``y_{...,j} = sum_i x_{...,i} W_{j,i}``.
    """

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
        """Apply the linear projection to the last dimension of the input."""
        return einx.dot("... [d_in], d_out [d_in] -> ... d_out", in_features, self.weight)


class Embedding(nn.Module):
    """Embedding lookup table that maps token ids to dense vectors.

    Inputs:
        ``token_ids`` with shape ``(...)`` and integer dtype.
    Outputs:
        Tensor with shape ``(..., d_model)``.
    Formula:
        ``y_{...,k} = E[token_ids_{...}, k]``.
    """

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
        """Return the embedding vector for each token index."""
        return self.weight[token_ids]


class RmsNorm(nn.Module):
    """Root Mean Square LayerNorm without mean-centering.

    Inputs:
        ``in_features`` with shape ``(..., d_model)``.
    Outputs:
        Tensor with shape ``(..., d_model)``.
    Formula:
        ``RMS(x)=sqrt((1/d_model) * sum_i x_i^2 + eps)``,
        ``y_i = gain_i * x_i / RMS(x)``.
    """

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
        """Normalize each token vector by its RMS and scale by learnable gain."""
        in_dtype = in_features.dtype
        in_features = in_features.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + self.eps)
        out_features = in_features * self.gain / rms
        return out_features.to(in_dtype)


def silu(in_features: torch.Tensor) -> torch.Tensor:
    # Apply SiLU elementwise: x * sigmoid(x).
    return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block used in modern Transformers.

    Inputs:
        ``in_features`` with shape ``(..., d_model)``.
    Outputs:
        Tensor with shape ``(..., d_model)``.
    Formula:
        ``y = W2( SiLU(W1 x) odot (W3 x) )``.
    """

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
        """Project up, gate with SiLU, then project back to model dimension."""
        return self.weight2(silu(self.weight1(in_features)) * self.weight3(in_features))


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
        start = 2 ** (-2 ** (- (math.log2(n) - 3)))
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
        scores: Float[torch.Tensor, "..., num_heads, seq_len_q, seq_len_k"]
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
        device: torch.device | None = None
    ):
        super().__init__()

        inv_freq = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))

        self.original_max_seq_len = float(original_max_seq_len)
        self.scale = scale
        self.mscale = mscale
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(
        self,
        in_features: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
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


def softmax(
    in_features: Float[torch.Tensor, "..."],
    dim: int
) -> Float[torch.Tensor, "..."]:
    # Compute numerically stable softmax along the specified dimension.
    max_element = torch.amax(in_features, dim=dim, keepdim=True)
    x_exp = torch.exp(in_features - max_element)
    out_features = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return out_features


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... seq_len_q d_k"],
    K: Float[torch.Tensor, "... seq_len_k d_k"],
    V: Float[torch.Tensor, "... seq_len_k d_v"],
    mask: Bool[torch.Tensor, "seq_len_q seq_len_k"] | None = None
) -> Float[torch.Tensor, "... seq_len_q d_v"]:
    # Compute masked attention: softmax(QK^T / sqrt(d_k)) V.
    d_k = K.shape[-1]
    S = einx.dot(
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k",
        Q,
        K
    ) / math.sqrt(d_k)
    if mask is not None:
        A = softmax(
            S.masked_fill(~mask, float('-inf')),
            dim=-1
        )
    else:
        A = softmax(S, dim=-1)
    O = einx.dot(
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
        A,
        V
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
            device=device
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len_q d_v"]:
        """Apply RoPE to Q/K and then run standard causal multi-head attention."""
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
    """Pre-norm Transformer block with RoPE attention and SwiGLU FFN.

    Inputs:
        ``in_features`` with shape ``(..., seq_len, d_model)`` and
        ``token_positions`` with shape ``(..., seq_len)``.
    Outputs:
        Tensor with shape ``(..., seq_len, d_model)``.
    Formula:
        ``h1 = x + MHA(RMSNorm_1(x))``,
        ``y = h1 + FFN(RMSNorm_2(h1))``.
    """

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
        """Apply two residual sublayers: attention then feed-forward."""
        x_rms1 = self.rms1(in_features)
        atten_features = self.mha(x_rms1, token_positions) + in_features
        x_rms2 = self.rms2(atten_features)
        out_features = self.ffn(x_rms2) + atten_features
        return out_features


class TransformerLanguageModel(nn.Module):
    """Autoregressive Transformer language model over token sequences.

    Inputs:
        ``tokens`` with shape ``(batch_size, seq_len)``.
    Outputs:
        Logits with shape ``(batch_size, seq_len, vocab_size)``.
    Formula:
        ``x_0 = Embedding(tokens)``,
        ``x_{l+1} = Block_l(x_l)``,
        ``logits = W_vocab * RMSNorm(x_L)``.
    """

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
        """Map token ids to per-position next-token logits."""
        _, seq_len = tokens.shape
        token_positions = torch.arange(seq_len, device=tokens.device, dtype=torch.long)

        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.final_rms(x)
        logits = self.final_layer(x)

        return logits


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"],
    targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, "..."]:
    target_logit = einx.get_at(
        "... [vocab_size], ... [1] -> ...",
        logits,
        targets.unsqueeze(-1)
    )
    max_elements = torch.amax(logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_elements), dim=-1))

    losses = max_elements - target_logit + log_sum_exp

    return losses


def cross_entropy_loss(
    logits: Float[torch.Tensor, "... vocab_size"],
    targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    losses = cross_entropy(logits, targets)
    return losses.mean()


def sequence_perplexity(
    losses: Float[torch.Tensor, "... seq_len"]
) -> Float[torch.Tensor, "..."]:
    return losses.mean(dim=-1).exp()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        # hyparameters
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None,
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0) + 1
                exp_avg: torch.Tensor = state.get("exp_avg", torch.zeros_like(p, memory_format=torch.preserve_format))
                exp_avg_sq: torch.Tensor = state.get("exp_avg_sq", torch.zeros_like(p, memory_format=torch.preserve_format))
                grad = p.grad.data

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                lr_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)
                p.addcdiv_(exp_avg, (exp_avg_sq.sqrt().add_(eps)), value=-lr_t)
                if weight_decay != 0.0:
                    p.add_(p, alpha = - lr * weight_decay)

                state["t"] = t
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss


def learing_rate_scheduler_with_warmup(
    alpha_max: float,
    alpha_min: float,
    T_warmpup: int,
    T_c: int
):
    def scheduler(t: int) -> float:
        if t < T_warmpup:
            return t / T_warmpup * alpha_max
        elif T_warmpup <= t <= T_c:
            return alpha_min + (1 + math.cos((t - T_warmpup) / (T_c - T_warmpup) * math.pi)) * (alpha_max - alpha_min) / 2
        else:
            return alpha_min
    return scheduler


def gradient_clipping(
    params: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    total_l2_norm = 0.0
    for p in params:
        if p.grad is None:
            continue

        grad = p.grad
        total_l2_norm += (grad ** 2).sum()

    total_l2_norm.sqrt_()
    clip_coff = max_l2_norm / (total_l2_norm + eps)

    if clip_coff < 1.0:
        for p in params:
            if p.grad is None:
                continue

            p.grad.mul_(clip_coff)

    return


def batch_loading(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    max_start = len(dataset) - 1 - context_length
    starts = np.random.randint(low=0, high=max_start+1, size=(batch_size,))
    offsets = np.arange(context_length+1)

    indicies = starts[:, None] + offsets[None, :]
    blocks = dataset[indicies]

    inputs = torch.Tensor(blocks[:, :-1], device=device)
    targets = torch.Tensor(blocks[:, 1:], device=device)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    checkpoint = {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src, map_location="cpu")

    iteration = checkpoint["iteration"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    return iteration
