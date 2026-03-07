"""Compatibility module re-exporting tokenizer utilities.

This file keeps existing imports stable while implementation lives in
`pretokenize.py` and `bpe_ops.py`.
"""

from cs336_basics.bpe_ops import (
    BYTE_VOCAB_SIZE,
    BytePair,
    ByteSeq,
    MaxCountPair,
    MergeLru,
    PairCounter,
    PairToSeqMap,
    SeqCounter,
    add_bytes_seq,
    apply_merge,
    best_pair_by_rank,
    count_byte_pair,
    gpt2_bytes_to_unicode,
    initialize_vocab,
    pop_valid_best_pair,
    remove_bytes_seq,
    seq_to_pair_counter,
)
from cs336_basics.pretokenize import (
    DEFAULT_SPLIT_SPECIAL_TOKEN,
    MINI_CHUNK_SIZE,
    PRETOKEN_PATTERN,
    count_bytes_seq,
    find_chunk_boundaries,
    pre_tokenization_encode,
    pre_tokenization_parallel,
    pre_tokenization_serial,
    pre_tokenization_train,
    pre_tokenization_worker,
    split_special_tokens,
)


BPE_OP_EXPORTS = (
    "BYTE_VOCAB_SIZE",
    "BytePair",
    "ByteSeq",
    "MaxCountPair",
    "MergeLru",
    "PairCounter",
    "PairToSeqMap",
    "SeqCounter",
    "add_bytes_seq",
    "apply_merge",
    "best_pair_by_rank",
    "count_byte_pair",
    "gpt2_bytes_to_unicode",
    "initialize_vocab",
    "pop_valid_best_pair",
    "remove_bytes_seq",
    "seq_to_pair_counter",
)

PRETOKENIZE_EXPORTS = (
    "DEFAULT_SPLIT_SPECIAL_TOKEN",
    "MINI_CHUNK_SIZE",
    "PRETOKEN_PATTERN",
    "count_bytes_seq",
    "find_chunk_boundaries",
    "pre_tokenization_encode",
    "pre_tokenization_parallel",
    "pre_tokenization_serial",
    "pre_tokenization_train",
    "pre_tokenization_worker",
    "split_special_tokens",
)

__all__ = [*BPE_OP_EXPORTS, *PRETOKENIZE_EXPORTS]
