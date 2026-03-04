from __future__ import annotations
import heapq
import json
from typing import Iterable

from cs336_basics.utils import (
    BytePair,
    ByteSeq,
    PairCounter,
    PairToSeqMap,
    SeqCounter,
    add_bytes_seq,
    apply_merge,
    gpt2_bytes_to_unicode,
    initialize_vocab,
    pop_valid_best_pair,
    pre_tokenization_parallel,
    remove_bytes_seq,
    MaxCountPair,
    pre_tokenization_encode,
    MergeLru,
    best_pair_by_rank
)

class Tokenizer:
    """Byte-level BPE tokenizer with train/load interfaces."""
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}


    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[BytePair],
        special_tokens: list[str] | None = None
    ):
        """Store tokenizer vocabulary, merges, and optional special tokens."""
        self.vocab = vocab
        self.bytes_to_tokenid = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_rank = {m: i for i, m in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None
        self.lru = MergeLru(1000)

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        """Build a tokenizer from GPT-2 style vocab/merge files."""
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([cls.gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
            }

        with open(merges_filepath, encoding="utf-8") as f:
            gpt2_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([cls.gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([cls.gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in gpt2_merges
            ]

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

        bytes_seq_counter, bytes_pair_counter, pair_to_seq = pre_tokenization_parallel(
            input_path,
            special_tokens,
        )

        bytes_pair_heap = [MaxCountPair(count, pair) for pair, count in bytes_pair_counter.items()]
        heapq.heapify(bytes_pair_heap)

        while next_token_id < vocab_size:
            merge_pair = pop_valid_best_pair(bytes_pair_heap, bytes_pair_counter)
            new_token = merge_pair[0] + merge_pair[1]

            merges.append(merge_pair)
            vocab[next_token_id] = new_token
            next_token_id += 1

            affected_bytes_seq = list(pair_to_seq[merge_pair])

            new_bytes_seqs: SeqCounter = {}
            for bytes_seq in affected_bytes_seq:
                count = bytes_seq_counter[bytes_seq]
                del bytes_seq_counter[bytes_seq]

                remove_bytes_seq(bytes_seq, count, bytes_pair_counter, pair_to_seq, bytes_pair_heap)

                new_seq = apply_merge(bytes_seq, merge_pair, new_token)
                new_bytes_seqs[new_seq] = count

            for new_seq, count in new_bytes_seqs.items():
                assert new_seq not in bytes_seq_counter
                bytes_seq_counter[new_seq] = count

                add_bytes_seq(new_seq, count, bytes_pair_counter, pair_to_seq, bytes_pair_heap)

        return cls(vocab, merges, special_tokens)

    def encode_seq(
        self,
        seq: ByteSeq
    ) -> tuple[int, ...]:
        if len(seq) == 1:
            return (self.bytes_to_tokenid[seq[0]],)

        lru_tokens = self.lru.get(seq)
        if lru_tokens is not None:
            return lru_tokens

        merge_seq = seq
        while True:
            merge_pair, _ = best_pair_by_rank(merge_seq, self.merge_rank)

            if merge_pair is None:
                break

            new_token = merge_pair[0] + merge_pair[1]
            merge_seq = apply_merge(merge_seq, merge_pair, new_token)

        seq_tokens = tuple(self.bytes_to_tokenid[b] for b in merge_seq)
        self.lru.put(seq, seq_tokens)

        return seq_tokens

    def encode(
        self,
        text: str
    ) -> list[int]:
        """Encode text into token ids."""
        bytes_seqs, _ = pre_tokenization_encode(text, self.special_tokens)

        tokens : list[int] = []

        for seq in bytes_seqs:
            tokens.extend(self.encode_seq(seq))

        return tokens

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterable[int]:
        """Yield token ids for each text chunk in an iterable."""
        buffer = ""
        if self.special_tokens is None:
            special_suffix = 0
        else:
            special_suffix = max((len(t) for t in self.special_tokens)) - 1

        for chunk in iterable:
            if not chunk:
                continue

            buffer += chunk
            # if special_suffix > 0:
            #     safe_text, buffer = buffer[:-special_suffix], buffer[-special_suffix:]
            # else:
            #     safe_text, buffer = buffer, ""

            bytes_seq, last = pre_tokenization_encode(buffer, self.special_tokens)

            buffer = last

            for seq in bytes_seq[:-1]:
                for token in self.encode_seq(seq):
                    yield token

        if buffer:
            bytes_seq, _ = pre_tokenization_encode(buffer, self.special_tokens)
            for seq in bytes_seq:
                for token in self.encode_seq(seq):
                    yield token

    def decode(
        self,
        ids: list[int]
    ) -> str:
        """Decode token ids back to text."""
        raw = b"".join(self.vocab[i] for i in ids)

        return raw.decode("utf-8", errors="ignore")
