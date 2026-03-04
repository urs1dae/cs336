import os
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, Literal

import regex as re

from cs336_basics.bpe_ops import (
    ByteSeq,
    PairCounter,
    PairToSeqMap,
    SeqCounter,
    count_byte_pair,
)


MINI_CHUNK_SIZE = 4096
DEFAULT_SPLIT_SPECIAL_TOKEN = b"<|endoftext|>"
PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _word_to_byte_sequence(word: str) -> ByteSeq:
    """Convert a pre-token word into a single-byte byte-sequence tuple."""
    return tuple(bytes([byte_value]) for byte_value in word.encode("utf-8"))


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # Read ahead in small blocks until a safe split token boundary is found.
    mini_chunk_size = MINI_CHUNK_SIZE

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        scan_position = chunk_boundaries[boundary_index]
        file.seek(scan_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            # Find the special token in the mini chunk
            match_offset = mini_chunk.find(split_special_token)
            if match_offset != -1:
                chunk_boundaries[boundary_index] = scan_position + match_offset
                break
            scan_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    """Split text by special tokens while keeping non-special spans."""
    escaped = [re.escape(token) for token in special_tokens]
    if not escaped:
        return [text]
    pattern = f"({'|'.join(escaped)})"
    return re.split(pattern, text)


def count_bytes_seq(
    chunks: list[str],
) -> SeqCounter:
    """Count UTF-8 byte sequences of GPT-2 style pre-tokens."""
    pretoken_counts = Counter()
    for chunk in chunks:
        pretoken_matches = re.findall(PRETOKEN_PATTERN, chunk)
        pretoken_counts.update(pretoken_matches)

    # Represent each pre-token as a tuple of single-byte bytes objects.
    bytes_seq_counter = {
        _word_to_byte_sequence(word): count for word, count in pretoken_counts.items()
    }

    return bytes_seq_counter


def _tokenize_text_to_byte_sequences(
    text: str,
    special_tokens: list[str],
    keep_special_tokens: bool,
) -> tuple[list[ByteSeq], str]:
    """Tokenize text into byte sequences and return the trailing fragment."""
    chunks = split_special_tokens(text, special_tokens)
    special_token_set = set(special_tokens)

    byte_sequences: list[ByteSeq] = []
    trailing_fragment = ""

    for part in chunks:
        if not part:
            continue

        if part in special_token_set:
            if keep_special_tokens:
                byte_sequences.append((part.encode("utf-8"),))
            trailing_fragment = ""
            continue

        part_matches = re.findall(PRETOKEN_PATTERN, part)
        if part_matches:
            trailing_fragment = part_matches[-1]
        for word in part_matches:
            byte_sequences.append(_word_to_byte_sequence(word))

    return byte_sequences, trailing_fragment


def _build_statistics_from_text(
    text: str,
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Build sequence and pair statistics from a text block."""
    # Exclude special-token spans from BPE statistics.
    byte_sequences, _ = _tokenize_text_to_byte_sequences(text, special_tokens, keep_special_tokens=False)
    bytes_seq_counter: SeqCounter = Counter(byte_sequences)
    bytes_pair_counter, pair_to_seq = count_byte_pair(bytes_seq_counter)
    return bytes_seq_counter, bytes_pair_counter, pair_to_seq


def _merge_worker_statistics(
    results: list[tuple[SeqCounter, PairCounter, PairToSeqMap]],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Merge worker-local statistics into global counters and indexes."""
    bytes_seq_counter: SeqCounter = Counter()
    bytes_pair_counter: PairCounter = Counter()
    pair_to_seq: PairToSeqMap = defaultdict(set)

    for sequence_counter, pair_counter, pair_map in results:
        bytes_seq_counter.update(sequence_counter)
        bytes_pair_counter.update(pair_counter)
        for pair, sequences in pair_map.items():
            pair_to_seq[pair].update(sequences)

    return bytes_seq_counter, bytes_pair_counter, pair_to_seq


def _build_training_intervals(corpus_path: str | os.PathLike) -> list[tuple[int, int]]:
    """Build byte intervals aligned to split-token-safe boundaries."""
    with open(corpus_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, cpu_count(), DEFAULT_SPLIT_SPECIAL_TOKEN)

    return [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]


def _build_worker_tasks(
    corpus_path: str | os.PathLike,
    byte_intervals: list[tuple[int, int]],
    special_tokens: list[str],
) -> list[tuple[str | os.PathLike, tuple[int, int], list[str]]]:
    """Build multiprocessing worker tasks from corpus intervals."""
    return [(corpus_path, interval, special_tokens) for interval in byte_intervals]


def _read_interval_texts(
    corpus_path: str | os.PathLike,
    byte_intervals: list[tuple[int, int]],
) -> list[str]:
    """Read and decode corpus byte intervals into UTF-8 text chunks."""
    text_chunks: list[str] = []
    with open(corpus_path, "rb") as f:
        for chunk_start, chunk_end in byte_intervals:
            f.seek(chunk_start)
            chunk_text = f.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")
            text_chunks.append(chunk_text)
    return text_chunks


def pre_tokenization_serial(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Pre-tokenize and count corpus statistics in one process.

    Contract: Input is corpus file path + special tokens; output is (seq counts, pair counts, pair->seq map).
    """
    return pre_tokenization_train(input_path, special_tokens, mode="serial")


def pre_tokenization_worker(
    input_path: str | os.PathLike,
    interval: tuple[int, int],
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Worker for counting one byte-range interval.

    Contract: Input is file path + byte interval + special tokens; output is local (seq, pair, reverse-index) stats.
    """
    chunk_start, chunk_end = interval
    with open(input_path, "rb") as file:
        file.seek(chunk_start)
        text = file.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")

    return _build_statistics_from_text(text, special_tokens)


def pre_tokenization_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Pre-tokenize and count corpus statistics with multiprocessing.

    Contract: Input is corpus file path + special tokens; output is merged global (seq, pair, reverse-index) stats.
    """
    return pre_tokenization_train(input_path, special_tokens, mode="parallel")


def pre_tokenization_train(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    mode: Literal["parallel", "serial"] = "parallel",
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Unified pre-tokenization entrypoint for training.

    Contract: Input is corpus file path + special tokens + mode; output is global (seq, pair, reverse-index) stats.
    """
    byte_intervals = _build_training_intervals(input_path)

    if mode == "serial":
        text_chunks = _read_interval_texts(input_path, byte_intervals)
        text = "".join(text_chunks)
        return _build_statistics_from_text(text, special_tokens)

    if mode != "parallel":
        raise ValueError("mode must be either 'parallel' or 'serial'")

    worker_tasks = _build_worker_tasks(input_path, byte_intervals, special_tokens)

    # Spawn at most one worker per interval.
    with Pool(min(len(byte_intervals), cpu_count())) as pool:
        results = pool.starmap(pre_tokenization_worker, worker_tasks)

    return _merge_worker_statistics(results)


def pre_tokenization_encode(
    text: str,
    special_tokens: list[str],
) -> tuple[list[ByteSeq], str]:
    """Pre-tokenization text for encoding, keeping special tokens.

    Contract: Input is text + special tokens; output is (byte-sequence list, trailing fragment for stream continuation).
    """
    return _tokenize_text_to_byte_sequences(text, special_tokens, keep_special_tokens=True)
