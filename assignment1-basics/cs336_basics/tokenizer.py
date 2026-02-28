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


class Tokenizer:
    def __init__():
        pass


def bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    unicode_values = byte_values[:]
    n = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            unicode_values.append(256 + n)
            n += 1
    return dict(zip(byte_values, [chr(c) for c in unicode_values]))

def pre_tokenization(
    chunk: bytes, 
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:
        # bytes_to_unicode
        b2u = bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}
        
        mapped_special_tokens = [
            "".join(b2u[b] for b in token.encode("utf-8"))
            for token in special_tokens
        ]

        # split text by special_tokens
        text = "".join(b2u[b] for b in chunk)
        if mapped_special_tokens:
            special_tokens_pattern = "|".join(map(re.escape, mapped_special_tokens))
            pattern = re.compile(f"({special_tokens_pattern})")
            split_text = [c for c in re.split(pattern, text) if c]
        else:
            split_text = [text]

        PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\s)"""
        words = []
        for t in split_text:
            if t in mapped_special_tokens:
                words.append(t)
            else:
                words.extend(re.findall(PAT, t))
        word_counter = Counter(words)
        bytes_counter = Counter()
        for w, count in word_counter.items():
            if w in mapped_special_tokens:
                continue
            byte_seq = tuple(u2b[c] for c in w)
            bytes_counter[byte_seq] = count

        return bytes_counter
    

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
    special_tokens_id = set()
    current_id = 256
    for token in special_tokens:
        bs = token.encode("utf-8")
        special_tokens_id.add(bs)
        vocab[current_id] = bs
        current_id += 1
    merges = []

    # pre tokenization
    f = open(input_path, "rb")

    assert "<|endoftext|>" in special_tokens

    boundaries = find_chunk_boundaries(f, 5, b"<|endoftext|>")

    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
    bytes_counter = Counter(chunks)

    # merge
    current_id = len(vocab)
    while current_id < vocab_size:
        
        # caculate count of all updated byte_pair_count
        byte_pair_counter = defaultdict(int)
        for bs, count in bytes_counter.items():
            if len(bs) < 2:
                continue
            for byte1, byte2 in zip(bs[:-1], bs[1:]):
                byte_pair_counter[(byte1, byte2)] += count
        
        # get lexicographically greater pair with max count
        if not byte_pair_counter:
            break
        else:
            max_count = max(byte_pair_counter.values())
        best_pairs = []
        for byte_pair, count in byte_pair_counter.items():
            if count != max_count:
                continue
            best_pairs.append(byte_pair)
        best_pair = max(best_pairs)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        
        # update vocab and bytes_count
        vocab[current_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        keys = list(bytes_counter.keys())
        for key in keys:
            new_key = []
            i = 0
            updated = False
            while i < len(key):
                if key[i:i+2] == best_pair:
                    new_key.append(current_id)
                    i += 2
                    updated = True
                else:
                    new_key.append(key[i])
                    i+= 1
            if updated:
                bytes_counter[tuple(new_key)] += bytes_counter[key]
                bytes_counter.pop(key, None)
        current_id += 1

    return vocab, merges

