from __future__ import annotations

import os

from cs336_basics.tokenizer import Tokenizer


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    tokenizer = Tokenizer.train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
    return tokenizer.vocab, tokenizer.merges
