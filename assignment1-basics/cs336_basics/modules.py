import math
import os
from typing import BinaryIO, Callable, IO, Optional

import torch

from .attention import (
    Alibi,
    CausalMaskMultiHeadSelfAttention,
    CausalMaskMultiHeadSelfAttentionWithRope,
    RoPE,
    Yarn,
    get_alibi_slopes,
    scaled_dot_product_attention,
)
from .nn_utils import (
    Embedding,
    Linear,
    RmsNorm,
    SwiGLU,
    batch_loading,
    cross_entropy,
    cross_entropy_loss,
    gradient_clipping,
    sequence_perplexity,
    silu,
    softmax,
)
from .transformer import TransformerBlock, TransformerLanguageModel


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
                exp_avg: torch.Tensor = state.get(
                    "exp_avg", torch.zeros_like(p, memory_format=torch.preserve_format)
                )
                exp_avg_sq: torch.Tensor = state.get(
                    "exp_avg_sq", torch.zeros_like(p, memory_format=torch.preserve_format)
                )
                grad = p.grad.data

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                lr_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
                p.addcdiv_(exp_avg, (exp_avg_sq.sqrt().add_(eps)), value=-lr_t)
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                state["t"] = t
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss


def learing_rate_scheduler_with_warmup(
    alpha_max: float,
    alpha_min: float,
    T_warmpup: int,
    T_c: int,
):
    def scheduler(t: int) -> float:
        if t < T_warmpup:
            return t / T_warmpup * alpha_max
        elif T_warmpup <= t <= T_c:
            return alpha_min + (1 + math.cos((t - T_warmpup) / (T_c - T_warmpup) * math.pi)) * (alpha_max - alpha_min) / 2
        else:
            return alpha_min

    return scheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")

    iteration = checkpoint["iteration"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    return iteration


__all__ = [
    "Alibi",
    "AdamW",
    "CausalMaskMultiHeadSelfAttention",
    "CausalMaskMultiHeadSelfAttentionWithRope",
    "Embedding",
    "Linear",
    "RmsNorm",
    "RoPE",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLanguageModel",
    "Yarn",
    "batch_loading",
    "cross_entropy",
    "cross_entropy_loss",
    "get_alibi_slopes",
    "gradient_clipping",
    "learing_rate_scheduler_with_warmup",
    "load_checkpoint",
    "save_checkpoint",
    "scaled_dot_product_attention",
    "sequence_perplexity",
    "silu",
    "softmax",
]
