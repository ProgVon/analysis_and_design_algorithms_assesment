"""String matching algorithms + plagiarism detection + benchmarks.

All three matchers compare character by character in pure Python so the
timing comparison is fair (no C-speed slicing in the inner loop).
"""
import bisect
import random
import re
import time


def naive(text, pat):
    """O(n*m): try every alignment."""
    n, m = len(text), len(pat)
    if m == 0:
        return []
    hits = []
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pat[j]:
            j += 1
        if j == m:
            hits.append(i)
    return hits


def rabin_karp(text, pat, base=256, mod=1_000_000_007):
    """O(n+m) average: rolling hash, verify on hash hit."""
    n, m = len(text), len(pat)
    if m == 0 or m > n:
        return []
    high = pow(base, m - 1, mod)
    ph = th = 0
    for i in range(m):
        ph = (ph * base + ord(pat[i])) % mod
        th = (th * base + ord(text[i])) % mod
    hits = []
    for i in range(n - m + 1):
        if ph == th:
            j = 0  # verify char by char to rule out hash collisions
            while j < m and text[i + j] == pat[j]:
                j += 1
            if j == m:
                hits.append(i)
        if i < n - m:
            th = ((th - ord(text[i]) * high) * base + ord(text[i + m])) % mod
    return hits


def kmp(text, pat):
    """O(n+m) worst case: prefix table avoids re-scanning text."""
    m = len(pat)
    if m == 0:
        return []
    lps, k = [0] * m, 0
    for i in range(1, m):
        while k and pat[i] != pat[k]:
            k = lps[k - 1]
        if pat[i] == pat[k]:
            k += 1
        lps[i] = k
    hits, k = [], 0
    for i, c in enumerate(text):
        while k and c != pat[k]:
            k = lps[k - 1]
        if c == pat[k]:
            k += 1
        if k == m:
            hits.append(i - m + 1)
            k = lps[k - 1]
    return hits


ALGORITHMS = {"Naive": naive, "Rabin-Karp": rabin_karp, "KMP": kmp}


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if words(s)]


def detect(source, suspect, n=5, algo="KMP"):
    """Search every n-word phrase of `suspect` inside `source`.

    Returns similarity (% of suspect words inside a copied phrase), the matches
    with sentence indices on both sides, and the search time.
    """
    find = ALGORITHMS[algo]
    src_sents = sentences(source)
    parts, starts, pos = [], [], 1
    for s in src_sents:
        p = " ".join(words(s))
        starts.append(pos)
        parts.append(p)
        pos += len(p) + 1
    text = " " + " ".join(parts) + " "  # padding so matches land on word boundaries

    sus_sents = sentences(suspect)
    sus_words, sus_sent = [], []
    for k, s in enumerate(sus_sents):
        for w in words(s):
            sus_words.append(w)
            sus_sent.append(k)

    covered = [False] * len(sus_words)
    matches = []
    t0 = time.perf_counter()
    for i in range(len(sus_words) - n + 1):
        phrase = " ".join(sus_words[i:i + n])
        hits = find(text, " " + phrase + " ")
        if hits:
            covered[i:i + n] = [True] * n
            src = sorted({bisect.bisect_right(starts, h + 1) - 1 for h in hits})
            matches.append({"phrase": phrase, "suspect_sentence": sus_sent[i], "source_sentences": src})
    elapsed = time.perf_counter() - t0

    return {
        "similarity": 100 * sum(covered) / len(sus_words) if sus_words else 0.0,
        "matches": matches,
        "time": elapsed,
        "words": sus_words,
        "covered": covered,
        "source_sentences": src_sents,
        "suspect_sentences": sus_sents,
    }


VOCAB = ("data algorithm pattern string text match search hash prefix table "
         "document copy source report student result method analysis value").split()


def make_text(size, kind="random", seed=0):
    """random = English-like words; worst = 'aaa...' which is Naive's worst case."""
    if kind == "worst":
        return "a" * size
    rng = random.Random(seed)
    out, total = [], 0
    while total < size:
        w = rng.choice(VOCAB)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)[:size]


def make_pattern(text, length, kind="random"):
    # worst case: 'aa...ab' never matches but forces Naive to scan m chars per alignment
    return "a" * (length - 1) + "b" if kind == "worst" else text[-length:]


def time_algo(fn, text, pat, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(text, pat)
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark(sizes, lengths, kind="random"):
    """Rows of (algorithm, doc size, pattern length, ms) over the size x length grid."""
    rows = []
    for size in sizes:
        text = make_text(size, kind)
        for length in lengths:
            pat = make_pattern(text, length, kind)
            for name, fn in ALGORITHMS.items():
                rows.append({"Algorithm": name, "Document size": size, "Pattern length": length,
                             "Time (ms)": 1000 * time_algo(fn, text, pat)})
    return rows
