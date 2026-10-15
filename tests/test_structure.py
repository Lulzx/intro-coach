"""Run every labelled intro in corpus.py through the spaCy structure checks.

    uv run pytest tests            # pass/fail
    uv run python tests/test_structure.py   # per-check accuracy and every miss
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpus import CORPUS  # noqa: E402
from holdout import FRESH, HOLDOUT  # noqa: E402

from intro_coach import nlp  # noqa: E402


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def run(item):
    hits = nlp.detect(sentences(item["text"]), list(item["expect"]))
    return {k: hits[k]["found"] is not None for k in item["expect"]}


def report(items=CORPUS + HOLDOUT + FRESH):
    tally = defaultdict(lambda: [0, 0])
    misses = []
    for item in items:
        got = run(item)
        for k, want in item["expect"].items():
            tally[k][0] += got[k] == want
            tally[k][1] += 1
            if got[k] != want:
                misses.append((item["scenario"], k, want, item["note"], item["text"]))
    total = sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values())
    return tally, misses, total


def test_corpus():
    _, misses, _ = report()
    assert not misses, "\n".join(f"{s}/{k}: expected {w} ({n})" for s, k, w, n, _ in misses)


if __name__ == "__main__":
    tally, misses, (ok, n) = report()
    for k, (good, count) in sorted(tally.items()):
        print(f"{k:13s} {good}/{count}")
    print(f"overall {ok}/{n} = {ok / n:.0%}\n")
    for s, k, want, note, text in misses:
        print(f"MISS {s}/{k}: expected {'present' if want else 'absent'} ({note})\n     {text}")
