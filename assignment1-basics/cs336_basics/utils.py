import heapq
import os
from collections import Counter, defaultdict, OrderedDict
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re


BYTE_VOCAB_SIZE = 256
MINI_CHUNK_SIZE = 4096
DEFAULT_SPLIT_SPECIAL_TOKEN = b"<|endoftext|>"
PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

ByteSeq = tuple[bytes, ...]
BytePair = tuple[bytes, bytes]
SeqCounter = dict[ByteSeq, int]
PairCounter = dict[BytePair, int]
PairToSeqMap = dict[BytePair, set[ByteSeq]]


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


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

    mini_chunk_size = MINI_CHUNK_SIZE  # Read ahead by 4k bytes at a time

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
    special_tokens: list[str] | None,
) -> list[str]:
    """Split text by special tokens while keeping non-special spans."""
    if special_tokens is None: return [text]
    escaped = [re.escape(token) for token in special_tokens]
    pattern = f"({"|".join(escaped)})"
    return re.split(pattern, text)


def count_bytes_seq(
    chunks: list[str],
) -> SeqCounter:
    """Count UTF-8 byte sequences of GPT-2 style pre-tokens."""
    words_counter = Counter()
    for chunk in chunks:
        words = re.findall(PRETOKEN_PATTERN, chunk)
        words_counter.update(words)

    bytes_seq_counter = {
        tuple(bytes([b]) for b in word.encode("utf-8")): count
        for word, count in words_counter.items()
    }

    return bytes_seq_counter


def seq_to_pair_counter(
    seq: ByteSeq,
) -> PairCounter:
    """Count adjacent byte-pair occurrences within one byte sequence."""
    pair_counter: PairCounter = defaultdict(int)
    for b1, b2 in zip(seq[:-1], seq[1:]):
        pair_counter[(b1, b2)] += 1
    return pair_counter


def count_byte_pair(
    bytes_seq_counter: SeqCounter,
) -> tuple[PairCounter, PairToSeqMap]:
    """Aggregate global pair counts and pair-to-sequence reverse index."""
    bytes_pair_counter: PairCounter = defaultdict(int)
    pair_to_seq: PairToSeqMap = defaultdict(set)

    for bytes_seq, count in bytes_seq_counter.items():
        pair_counter = seq_to_pair_counter(bytes_seq)
        for pair, occurrences in pair_counter.items():
            bytes_pair_counter[pair] += occurrences * count
            pair_to_seq[pair].add(bytes_seq)

    return bytes_pair_counter, pair_to_seq


def _count_from_text(
    text: str,
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Run split and counting pipeline for one text block."""
    chunks = split_special_tokens(text, special_tokens)
    special_token_set = set(special_tokens)
    chunks = [chunk for chunk in chunks if chunk and chunk not in special_token_set]

    bytes_seq_counter = count_bytes_seq(chunks)
    bytes_pair_counter, pair_to_seq = count_byte_pair(bytes_seq_counter)
    return bytes_seq_counter, bytes_pair_counter, pair_to_seq


def _merge_pretokenization_results(
    results: list[tuple[SeqCounter, PairCounter, PairToSeqMap]],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Merge worker-local counters/maps into global structures."""
    bytes_seq_counter: SeqCounter = Counter()
    bytes_pair_counter: PairCounter = Counter()
    pair_to_seq: PairToSeqMap = defaultdict(set)

    for seq_counter, pair_counter, pair_map in results:
        bytes_seq_counter.update(seq_counter)
        bytes_pair_counter.update(pair_counter)
        for pair, sequences in pair_map.items():
            pair_to_seq[pair].update(sequences)

    return bytes_seq_counter, bytes_pair_counter, pair_to_seq


def pre_tokenization_serial(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Pre-tokenize and count corpus statistics in one process."""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, cpu_count(), DEFAULT_SPLIT_SPECIAL_TOKEN)
        text_chunks: list[str] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            text_chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

    text = "".join(text_chunks)
    return _count_from_text(text, special_tokens)


def pre_tokenization_worker(
    input_path: str | os.PathLike,
    interval: tuple[int, int],
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Worker for counting one byte-range interval."""
    start, end = interval
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors="ignore")

    return _count_from_text(text, special_tokens)


def pre_tokenization_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> tuple[SeqCounter, PairCounter, PairToSeqMap]:
    """Pre-tokenize and count corpus statistics with multiprocessing."""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, cpu_count(), DEFAULT_SPLIT_SPECIAL_TOKEN)

    intervals = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    tasks = [(input_path, interval, special_tokens) for interval in intervals]

    with Pool(min(len(intervals), cpu_count())) as pool:
        results = pool.starmap(pre_tokenization_worker, tasks)

    return _merge_pretokenization_results(results)


class MaxCountPair:
    """Heap item that pops the pair with highest count first."""
    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: BytePair):
        self.count = count
        self.pair = pair

    def __lt__(self, other: "MaxCountPair"):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair

    def __iter__(self):
        yield self.count
        yield self.pair


def initialize_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    """Create initial byte vocab plus appended special tokens."""
    vocab = {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}
    next_token_id = BYTE_VOCAB_SIZE
    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1
    return vocab, next_token_id


def pop_valid_best_pair(
    bytes_pair_heap: list[MaxCountPair],
    bytes_pair_counter: PairCounter,
) -> BytePair:
    """Pop the best non-stale pair from a lazy-update max heap."""
    while True:
        candidate_count, candidate_pair = heapq.heappop(bytes_pair_heap)
        real_count = bytes_pair_counter.get(candidate_pair)
        if real_count is None:
            continue
        if candidate_count == real_count:
            return candidate_pair


def remove_bytes_seq(
    bytes_seq: ByteSeq,
    count: int,
    bytes_pair_counter: PairCounter,
    pair_to_seq: PairToSeqMap,
    bytes_pair_heap: list[MaxCountPair],
) -> None:
    """Remove one sequence's pair contributions from global counters."""
    pair_counter = seq_to_pair_counter(bytes_seq)

    for pair, occurrences in pair_counter.items():
        bytes_pair_counter[pair] -= occurrences * count
        heapq.heappush(bytes_pair_heap, MaxCountPair(bytes_pair_counter[pair], pair))
        if bytes_pair_counter[pair] <= 0:
            assert bytes_pair_counter[pair] == 0
            del bytes_pair_counter[pair]

        related_sequences = pair_to_seq[pair]
        related_sequences.discard(bytes_seq)
        if not related_sequences:
            del pair_to_seq[pair]


def add_bytes_seq(
    bytes_seq: ByteSeq,
    count: int,
    bytes_pair_counter: PairCounter,
    pair_to_seq: PairToSeqMap,
    bytes_pair_heap: list[MaxCountPair],
) -> None:
    """Add one sequence's pair contributions to global counters."""
    pair_counter = seq_to_pair_counter(bytes_seq)

    for pair, occurrences in pair_counter.items():
        bytes_pair_counter[pair] += occurrences * count
        heapq.heappush(bytes_pair_heap, MaxCountPair(bytes_pair_counter[pair], pair))
        pair_to_seq[pair].add(bytes_seq)


def apply_merge(
    seq: ByteSeq,
    merge_pair: BytePair,
    new_token: bytes,
) -> ByteSeq:
    """Apply one BPE merge pair greedily over a byte sequence."""
    new_seq: list[bytes] = []
    i = 0
    while i < len(seq):
        if seq[i:i + 2] == merge_pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return tuple(new_seq)


def pre_tokenization_encode(
    text: str,
    special_tokens: list[str] | None,
) -> tuple[list[ByteSeq], str]:
    """Pre-tokenization text for encoding, keeping special tokens"""
    chunks = split_special_tokens(text, special_tokens)

    special_token_set = set(special_tokens) if special_tokens else None
    output: list[ByteSeq] = []

    for part in chunks:
        if not part: continue

        if special_token_set is not None and part in special_token_set:
            output.append((part.encode("utf-8"),))
            continue

        words = re.findall(PRETOKEN_PATTERN, part)
        for word in words:
            output.append(tuple(bytes([b]) for b in word.encode("utf-8")))

    return output, words[-1]


class MergeLru:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: ByteSeq) -> tuple[int, ...] | None:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: ByteSeq, val: tuple[int, ...]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = val

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def best_pair_by_rank(
    seq : ByteSeq,
    merge_rank : dict[BytePair, int]
) -> tuple[BytePair | None, int]:
    best_pair = None
    best_rank = len(merge_rank)
    if len(seq) == 1:
        return best_pair, best_rank
    for b1, b2 in zip(seq[:-1], seq[1:]):
        pair = (b1, b2)
        rank = merge_rank.get(pair)
        if rank is not None and rank < best_rank:
            best_rank = rank
            best_pair = pair
    return best_pair, best_rank
