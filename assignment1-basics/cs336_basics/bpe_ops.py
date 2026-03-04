import heapq
from collections import defaultdict, OrderedDict
from functools import lru_cache


BYTE_VOCAB_SIZE = 256

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


def seq_to_pair_counter(
    seq: ByteSeq,
) -> PairCounter:
    """Count adjacent byte-pair occurrences within one byte sequence."""
    sequence_pair_counts: PairCounter = defaultdict(int)
    for left_byte, right_byte in zip(seq[:-1], seq[1:]):
        sequence_pair_counts[(left_byte, right_byte)] += 1
    return sequence_pair_counts


def count_byte_pair(
    sequence_counts: SeqCounter,
) -> tuple[PairCounter, PairToSeqMap]:
    """Aggregate global pair counts and pair-to-sequence reverse index."""
    pair_counts: PairCounter = defaultdict(int)
    pair_to_sequences: PairToSeqMap = defaultdict(set)

    for sequence, sequence_count in sequence_counts.items():
        sequence_pair_counts = seq_to_pair_counter(sequence)
        for pair, pair_occurrences in sequence_pair_counts.items():
            pair_counts[pair] += pair_occurrences * sequence_count
            pair_to_sequences[pair].add(sequence)

    return pair_counts, pair_to_sequences


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
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    return vocab, next_token_id


def pop_valid_best_pair(
    pair_max_heap: list[MaxCountPair],
    pair_counts: PairCounter,
) -> BytePair:
    """Pop the best non-stale pair from a lazy-update max heap."""
    while True:
        candidate_count, candidate_pair = heapq.heappop(pair_max_heap)
        # Skip stale heap entries whose count no longer matches the latest counter.
        current_count = pair_counts.get(candidate_pair)
        if current_count is None:
            continue
        if candidate_count == current_count:
            return candidate_pair


def remove_bytes_seq(
    sequence: ByteSeq,
    sequence_count: int,
    pair_counts: PairCounter,
    pair_to_sequences: PairToSeqMap,
    pair_max_heap: list[MaxCountPair],
) -> None:
    """Remove one sequence's pair contributions from global counters."""
    sequence_pair_counts = seq_to_pair_counter(sequence)

    for pair, pair_occurrences in sequence_pair_counts.items():
        pair_counts[pair] -= pair_occurrences * sequence_count
        heapq.heappush(pair_max_heap, MaxCountPair(pair_counts[pair], pair))
        if pair_counts[pair] <= 0:
            assert pair_counts[pair] == 0
            del pair_counts[pair]

        related_sequences = pair_to_sequences[pair]
        related_sequences.discard(sequence)
        if not related_sequences:
            del pair_to_sequences[pair]


def add_bytes_seq(
    sequence: ByteSeq,
    sequence_count: int,
    pair_counts: PairCounter,
    pair_to_sequences: PairToSeqMap,
    pair_max_heap: list[MaxCountPair],
) -> None:
    """Add one sequence's pair contributions to global counters."""
    sequence_pair_counts = seq_to_pair_counter(sequence)

    for pair, pair_occurrences in sequence_pair_counts.items():
        pair_counts[pair] += pair_occurrences * sequence_count
        # Push updated counts and lazily discard stale entries when popping.
        heapq.heappush(pair_max_heap, MaxCountPair(pair_counts[pair], pair))
        pair_to_sequences[pair].add(sequence)


def apply_merge(
    sequence: ByteSeq,
    merge_pair: BytePair,
    new_token: bytes,
) -> ByteSeq:
    """Apply one BPE merge pair greedily over a byte sequence."""
    merged_sequence: list[bytes] = []
    index = 0
    while index < len(sequence):
        # Prefer the leftmost valid pair each step.
        if sequence[index : index + 2] == merge_pair:
            merged_sequence.append(new_token)
            index += 2
        else:
            merged_sequence.append(sequence[index])
            index += 1
    return tuple(merged_sequence)


class MergeLru:
    """Small LRU cache for memoizing sequence-level merge results."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[ByteSeq, tuple[int, ...]] = OrderedDict()

    def get(self, key: ByteSeq) -> tuple[int, ...] | None:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: ByteSeq, value: tuple[int, ...]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        # Evict the least-recently-used item when exceeding capacity.
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def best_pair_by_rank(sequence: ByteSeq, merge_rank: dict[BytePair, int]) -> tuple[BytePair | None, int]:
    """Return the adjacent pair with best (lowest) merge rank for a sequence."""
    best_pair = None
    best_rank = len(merge_rank)
    if len(sequence) == 1:
        return best_pair, best_rank
    for left_byte, right_byte in zip(sequence[:-1], sequence[1:]):
        pair = (left_byte, right_byte)
        rank = merge_rank.get(pair)
        if rank is not None and rank < best_rank:
            best_rank = rank
            best_pair = pair
    return best_pair, best_rank
