import torch
from jaxtyping import Float, Int
from torch import nn

from .attention import CausalMaskMultiHeadSelfAttentionWithRope
from .nn_utils import Embedding, Linear, RmsNorm, SwiGLU


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
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rms1 = RmsNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.mha = CausalMaskMultiHeadSelfAttentionWithRope(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.rms2 = RmsNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_features: Float[torch.Tensor, "... d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"],
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
        dtype: torch.dtype | None = None,
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
            dtype=dtype,
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
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_rms = RmsNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.final_layer = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
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
