import os
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO


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

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_special_tokens(
    text: str,
    special_tokens: list[str]
) -> list[str]:
    
    escaped = [re.escape(token) for token in special_tokens]

    pattern = r"|".join(escaped)
    
    return re.split(pattern, text)


def count_bytes_seq(
    chunks: list[str]
) -> Counter:
    
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    words_counter = Counter()
    for chunk in chunks:
        words = re.findall(pattern, chunk)
        words_counter += Counter(words)
    
    bytes_seq_counter = {
        tuple(bytes([b]) for b in w.encode("utf-8")) : c 
        for w, c in words_counter.items()
    }
    
    return bytes_seq_counter


def seq_to_pair_counter(
    seq: tuple[bytes, ...]
) -> dict[tuple[bytes, bytes], int]:
    pair_counter = defaultdict(int)
    for b1, b2 in zip(seq[:-1], seq[1:]):
        byte_pair = tuple((b1, b2))
        pair_counter[byte_pair] += 1
    return pair_counter


def count_byte_pair(
    bytes_seq_counter: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    
    byte_pair_counter: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_seq: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    
    for bytes_seq, count in  bytes_seq_counter.items():
        
        pair_counter = seq_to_pair_counter(bytes_seq)
        for pair, occ in pair_counter.items():
            byte_pair_counter[pair] += occ * count
            pair_to_seq[pair].add(bytes_seq)
    
    return byte_pair_counter, pair_to_seq


def remove_bytes_seq(
    bytes_seq: tuple[bytes, ...],
    count: int,
    byte_pair_counter: dict[tuple[bytes, bytes], int],
    pair_to_seq: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    
    pair_counter = seq_to_pair_counter(bytes_seq)

    for pair, occ in pair_counter.items():
        byte_pair_counter[pair] -= occ * count
        if byte_pair_counter[pair] <= 0:
            assert byte_pair_counter[pair] == 0
            del byte_pair_counter[pair]
        
        s = pair_to_seq[pair]
        if s is not None:
            s.discard(bytes_seq)
            if not s:
                del pair_to_seq[pair]


def add_bytes_seq(
    bytes_seq: tuple[bytes, ...],
    count: int,
    byte_pair_counter: dict[tuple[bytes, bytes], int],
    pair_to_seq: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    
    pair_counter = seq_to_pair_counter(bytes_seq)
    
    for pair, occ in pair_counter.items():
        byte_pair_counter[pair] += occ * count
        pair_to_seq[pair].add(bytes_seq)


def apply_merge(
    seq: tuple[bytes, ...],
    merge_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[bytes, ...]:
    new_seq = []
    i = 0
    while i < len(seq):
        p = seq[i:i+2]
        if p == merge_pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return tuple(new_seq)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """


    # initialize vocab and merges
    vocab = {i: bytes([i]) for i in range(256)}
    new_id = 256
    for token in special_tokens:
        vocab[new_id] = token.encode("utf-8")
        new_id += 1
    merges = []


    # pre tokenization
    f = open(input_path, "rb")
    boundaries = find_chunk_boundaries(f, 5, b"<|endoftext|>")
    texts = ""
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        texts += f.read(end - start).decode("utf-8", errors="ignore")
    chunks = split_special_tokens(texts, special_tokens)
    bytes_seq_counter = count_bytes_seq(chunks)
    byte_pair_counter, pair_to_seq = count_byte_pair(bytes_seq_counter)


    # merges
    while new_id < vocab_size:
        
        merge_pair, count = max(byte_pair_counter.items(), key=lambda x: (x[1], x[0]))
        
        new_token = merge_pair[0] + merge_pair[1]

        merges.append(merge_pair)
        vocab[new_id] = new_token
        new_id += 1

        affected_bytes_seq = list(pair_to_seq[merge_pair])

        new_bytes_seqs = {}
        for bytes_seq in affected_bytes_seq:
            count = bytes_seq_counter[bytes_seq]
            del bytes_seq_counter[bytes_seq]

            remove_bytes_seq(bytes_seq, count, byte_pair_counter, pair_to_seq)
            
            new_seq = apply_merge(bytes_seq, merge_pair, new_token)
            new_bytes_seqs[new_seq] = count
        
        for new_seq, count in new_bytes_seqs.items():
            assert new_seq not in bytes_seq_counter
            bytes_seq_counter[new_seq] = count
            
            add_bytes_seq(new_seq, count, byte_pair_counter, pair_to_seq)
        
    
    return vocab, merges

