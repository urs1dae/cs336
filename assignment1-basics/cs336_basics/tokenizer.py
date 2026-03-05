from __future__ import annotations
import heapq
import json
from collections.abc import Iterable, Sequence

from cs336_basics.bpe_ops import (
    BytePair,
    PairCounter,
    PairToSeqMap,
    ByteSeq,
    add_bytes_seq,
    apply_merge,
    best_pair_by_rank,
    gpt2_bytes_to_unicode,
    initialize_vocab,
    remove_bytes_seq,
    MaxCountPair,
    MergeLru,
    SeqCounter,
    pop_valid_best_pair,
)
from cs336_basics.pretokenize import (
    pre_tokenization_encode,
    pre_tokenization_train,
)


class Tokenizer:
    """Byte-level BPE tokenizer with train/load interfaces."""

    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    gpt2_byte_decoder = {token: byte for byte, token in gpt2_byte_encoder.items()}

    @classmethod
    def _parse_gpt2_vocab(cls, gpt2_vocab: dict[str, int]) -> dict[int, bytes]:
        """Build a byte-token vocabulary from a GPT-2 vocab mapping."""
        return {
            token_id: bytes([cls.gpt2_byte_decoder[token] for token in token_string])
            for token_string, token_id in gpt2_vocab.items()
        }

    @classmethod
    def _parse_gpt2_merges(cls, merge_lines: list[str]) -> list[BytePair]:
        """Build byte-pair merge rules from GPT-2 merge lines."""
        merge_tokens = [tuple(line.rstrip().split(" ")) for line in merge_lines]
        return [
            (
                bytes([cls.gpt2_byte_decoder[token] for token in left]),
                bytes([cls.gpt2_byte_decoder[token] for token in right]),
            )
            for left, right in merge_tokens
        ]

    @staticmethod
    def _initialize_pair_max_heap(pair_counts: PairCounter) -> list[MaxCountPair]:
        """Initialize a max-heap view over pair counts."""
        max_heap = [MaxCountPair(count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(max_heap)
        return max_heap

    @staticmethod
    def _update_training_stats_after_merge(
        merge_pair: BytePair,
        new_token: bytes,
        sequence_counts: SeqCounter,
        pair_counts: PairCounter,
        pair_to_sequences: PairToSeqMap,
        pair_max_heap: list[MaxCountPair],
    ) -> None:
        """Apply one merge and refresh sequence and pair statistics."""
        affected_sequences = list(pair_to_sequences[merge_pair])

        merged_sequence_counts: SeqCounter = {}
        for sequence in affected_sequences:
            sequence_count = sequence_counts.pop(sequence)

            remove_bytes_seq(sequence, sequence_count, pair_counts, pair_to_sequences, pair_max_heap)

            merged_sequence = apply_merge(sequence, merge_pair, new_token)
            merged_sequence_counts[merged_sequence] = sequence_count

        for merged_sequence, sequence_count in merged_sequence_counts.items():
            assert merged_sequence not in sequence_counts
            sequence_counts[merged_sequence] = sequence_count

            add_bytes_seq(merged_sequence, sequence_count, pair_counts, pair_to_sequences, pair_max_heap)

    def __init__(self, vocab: dict[int, bytes], merges: list[BytePair], special_tokens: Sequence[str] = ()):
        """Store tokenizer vocabulary, merges, and optional special tokens."""
        self.vocab = vocab
        self.bytes_to_tokenid = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_rank = {m: i for i, m in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.lru = MergeLru(1000)

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Sequence[str] = ()):
        """Build a tokenizer from GPT-2 style vocab/merge files."""
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab = cls._parse_gpt2_vocab(gpt2_vocab)

        with open(merges_filepath, encoding="utf-8") as f:
            merges = cls._parse_gpt2_merges(f.readlines())

        return cls(vocab, merges, special_tokens)

    @classmethod
    def train(
        cls,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ):
        """Alias of `train_bpe` for a shorter training entrypoint."""
        return cls.train_bpe(input_path, vocab_size, special_tokens, **kwargs)

    @classmethod
    def train_bpe(
        cls,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ):
        """Train a byte-level BPE tokenizer and return a tokenizer instance."""
        vocab, next_token_id = initialize_vocab(special_tokens)
        merges: list[BytePair] = []
        pretokenization_mode = kwargs.get("pretokenization_mode", "parallel")

        sequence_counts, pair_counts, pair_to_sequences = pre_tokenization_train(
            input_path,
            special_tokens,
            mode=pretokenization_mode,
        )

        pair_max_heap = cls._initialize_pair_max_heap(pair_counts)

        while next_token_id < vocab_size:
            # Pick the best valid pair and create its merged token.
            merge_pair = pop_valid_best_pair(pair_max_heap, pair_counts)
            new_token = merge_pair[0] + merge_pair[1]

            merges.append(merge_pair)
            vocab[next_token_id] = new_token
            next_token_id += 1

            # Update sequence and pair statistics after applying this merge.
            cls._update_training_stats_after_merge(
                merge_pair,
                new_token,
                sequence_counts,
                pair_counts,
                pair_to_sequences,
                pair_max_heap,
            )

        return cls(vocab, merges, special_tokens)

    def _encode_seq(self, sequence: ByteSeq) -> tuple[int, ...]:
        """Encode a single pre-tokenized byte sequence into token ids."""
        if len(sequence) == 1:
            return (self.bytes_to_tokenid[sequence[0]],)

        lru_tokens = self.lru.get(sequence)
        if lru_tokens is not None:
            return lru_tokens

        merged_sequence = sequence
        while True:
            merge_pair, _ = best_pair_by_rank(merged_sequence, self.merge_rank)

            if merge_pair is None:
                break

            new_token = merge_pair[0] + merge_pair[1]
            merged_sequence = apply_merge(merged_sequence, merge_pair, new_token)

        sequence_tokens = tuple(self.bytes_to_tokenid[token_bytes] for token_bytes in merged_sequence)
        self.lru.put(sequence, sequence_tokens)

        return sequence_tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        byte_sequences, _ = pre_tokenization_encode(text, self.special_tokens)

        tokens: list[int] = []

        for sequence in byte_sequences:
            tokens.extend(self._encode_seq(sequence))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Yield token ids for each text chunk in an iterable."""
        pending_text = ""

        for chunk in iterable:
            if not chunk:
                continue

            pending_text += chunk
            byte_sequences, pending_text = pre_tokenization_encode(pending_text, self.special_tokens)

            # Keep the final sequence buffered to avoid splitting across chunks.
            for sequence in byte_sequences[:-1]:
                for token in self._encode_seq(sequence):
                    yield token

        if pending_text:
            # Flush whatever remains after the iterable is exhausted.
            byte_sequences, _ = pre_tokenization_encode(pending_text, self.special_tokens)
            for sequence in byte_sequences:
                for token in self._encode_seq(sequence):
                    yield token

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        raw = b"".join(self.vocab[i] for i in ids)

        return raw.decode("utf-8", errors="ignore")
