import random

from matchers import ALGORITHMS, benchmark, detect


def reference(text, pat):
    return [i for i in range(len(text) - len(pat) + 1) if text.startswith(pat, i)] if pat else []


def test_matchers_agree_with_reference():
    rng = random.Random(1)
    for _ in range(2000):
        text = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        pat = "".join(rng.choice("ab") for _ in range(rng.randint(0, 6)))
        for name, fn in ALGORITHMS.items():
            assert fn(text, pat) == reference(text, pat), (name, text, pat)


def test_detect():
    src = "The quick brown fox jumps over the lazy dog. Nothing else here."
    sus = "Hello there. The quick brown fox jumps over a cat."
    for name in ALGORITHMS:
        r = detect(src, sus, n=3, algo=name)
        assert [m["phrase"] for m in r["matches"]] == ["the quick brown", "quick brown fox", "brown fox jumps", "fox jumps over"]
        assert all(m["suspect_sentence"] == 1 and m["source_sentences"] == [0] for m in r["matches"])
        assert round(r["similarity"]) == 60  # 6 of 10 suspect words copied
    assert detect(src, "fox", n=3)["similarity"] == 0  # shorter than n


def test_benchmark_rows():
    assert len(benchmark([200], [5, 10], "worst")) == 6


if __name__ == "__main__":
    for t in (test_matchers_agree_with_reference, test_detect, test_benchmark_rows):
        t()
    print("all tests passed")
