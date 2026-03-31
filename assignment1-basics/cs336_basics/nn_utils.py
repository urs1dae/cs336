import math
from collections.abc import Iterable

import einx
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int
from torch import nn


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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
        )
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_in"],
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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3,
            b=3,
        )

    def forward(
        self,
        token_ids: Int[torch.Tensor, "..."],
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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        """Normalize each token vector by its RMS and scale by learnable gain."""
        in_dtype = in_features.dtype
        in_features = in_features.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(in_features**2, dim=-1, keepdim=True) + self.eps)
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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_ff = d_ff if d_ff else (int(8 / 3 * d_model) + 63) & ~63
        self.weight1 = Linear(d_model, d_ff, device, dtype)
        self.weight2 = Linear(d_ff, d_model, device, dtype)
        self.weight3 = Linear(d_model, d_ff, device, dtype)

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
    ) -> Float[torch.Tensor, "... d_model"]:
        """Project up, gate with SiLU, then project back to model dimension."""
        return self.weight2(silu(self.weight1(in_features)) * self.weight3(in_features))


def softmax(
    in_features: Float[torch.Tensor, "..."],
    dim: int,
) -> Float[torch.Tensor, "..."]:
    # Compute numerically stable softmax along the specified dimension.
    max_element = torch.amax(in_features, dim=dim, keepdim=True)
    x_exp = torch.exp(in_features - max_element)
    out_features = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return out_features


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"],
    targets: Int[torch.Tensor, "..."],
) -> Float[torch.Tensor, "..."]:
    target_logit = einx.get_at(
        "... [vocab_size], ... [1] -> ...",
        logits,
        targets.unsqueeze(-1),
    )
    max_elements = torch.amax(logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_elements), dim=-1))

    losses = max_elements - target_logit + log_sum_exp

    return losses


def cross_entropy_loss(
    logits: Float[torch.Tensor, "... vocab_size"],
    targets: Int[torch.Tensor, "..."],
) -> Float[torch.Tensor, ""]:
    losses = cross_entropy(logits, targets)
    return losses.mean()


def sequence_perplexity(
    losses: Float[torch.Tensor, "... seq_len"],
) -> Float[torch.Tensor, "..."]:
    return losses.mean(dim=-1).exp()


def gradient_clipping(
    params: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    total_l2_norm = 0.0
    for p in params:
        if p.grad is None:
            continue

        grad = p.grad
        total_l2_norm += (grad**2).sum()

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
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - 1 - context_length
    starts = np.random.randint(low=0, high=max_start + 1, size=(batch_size,))
    offsets = np.arange(context_length + 1)

    indicies = starts[:, None] + offsets[None, :]
    blocks = dataset[indicies]

    inputs = torch.Tensor(blocks[:, :-1], device=device)
    targets = torch.Tensor(blocks[:, 1:], device=device)

    return inputs, targets
