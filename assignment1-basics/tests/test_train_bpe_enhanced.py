import json
import cProfile
import pstats

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_tinystories():
    input_path = "assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"

    pr = cProfile.Profile()
    pr.enable()

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )

    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(20)

    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    vocab_serializable = {"".join(gpt2_byte_encoder[b] for b in v): str(k) for k, v in vocab.items()}
    with open(FIXTURES_PATH / "TinyStoriesV2-GPT4-vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=4)

    with open(FIXTURES_PATH / "TinyStoriesV2-GPT4-merges.txt", "w", encoding="utf-8") as f:
        for merge1, merge2 in merges:
            f.write(f"{''.join(gpt2_byte_encoder[b] for b in merge1)} {''.join(gpt2_byte_encoder[b] for b in merge2)}\n")


def test_train_bpe_expts_owt():
    input_path = "assignment1-basics/data/owt_train.txt"

    pr = cProfile.Profile()
    pr.enable()

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )

    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(20)

    gpt2_bytes_encoder = gpt2_bytes_to_unicode()
    vocab_serializable = {"".join(gpt2_bytes_encoder[b] for b in v): str(k) for k, v in vocab.items()}
    with open(FIXTURES_PATH / "owt-vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=4)

    with open(FIXTURES_PATH / "owt-merges.txt", "w", encoding="utf-8") as f:
        for merge1, merge2 in merges:
            f.write(f"{''.join(gpt2_bytes_encoder[b] for b in merge1)} {''.join(gpt2_bytes_encoder[b] for b in merge2)}\n")
