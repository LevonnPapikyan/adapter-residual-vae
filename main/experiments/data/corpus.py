"""The corpus: 20 Newsgroups, cleaned, with the training posts that leak into the test set removed.

What belongs here
-----------------
- ``load_corpus``, ``load_train_texts``, and the code that builds and checks ``corpus.json``.

What does NOT belong here
-------------------------
- The translated test set (``translations.py``) or the teacher (``teacher.py``).

Run ``python -m experiments.data.corpus --data-dir ../data`` to rebuild the corpus
from the raw download and compare it with the saved file. Nothing is written unless
the data folder has no ``corpus.json`` yet.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Sequence

import numpy as np

from core.runs.io import read_json, refuse_protected_path, write_json
from experiments.data.paths import PROTECTED_DIRS, DataPaths

MIN_CHARS = 50
SHINGLE_WORDS = 8
LEAK_SHARE = 0.90


@dataclass(frozen=True)
class Corpus:
    """Training and test posts with their newsgroup labels."""

    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    label_names: list[str]


@dataclass(frozen=True)
class PurgeResult:
    """The training posts kept, and how many were dropped for each reason."""

    texts: list[str]
    labels: np.ndarray
    num_identical: int
    num_contained: int


def load_corpus(*, paths: DataPaths) -> Corpus:
    """Read ``corpus.json``."""

    data = read_json(paths.corpus)
    return Corpus(
        train_texts=data["train"]["text"],
        train_labels=np.array(data["train"]["label"]),
        test_texts=data["test"]["text"],
        test_labels=np.array(data["test"]["label"]),
        label_names=data["label_names"],
    )


def load_train_texts(*, paths: DataPaths, lang: str = "en") -> list[str]:
    """The training posts in English, or their German translation (same posts, same order)."""

    if lang == "en":
        return load_corpus(paths=paths).train_texts
    return read_json(paths.data_dir / f"train_{lang}.json")


def clean_post(text: str) -> str:
    """Remove separator lines, e-mail addresses, and PGP blocks, and collapse whitespace."""

    text = re.sub(r"[-=_]{5,}", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"BEGIN PGP SIGNATURE.*|END PGP SIGNATURE.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_split(raw: Any) -> tuple[list[str], np.ndarray]:
    texts, labels = [], []
    for text, label in zip(raw.data, raw.target):
        text = clean_post(text)
        if len(text) > MIN_CHARS:
            texts.append(text)
            labels.append(int(label))
    return texts, np.array(labels)


def shingles(text: str, *, num_words: int = SHINGLE_WORDS) -> set[str]:
    """The word n-grams of a post, lowercase letters only."""

    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < num_words:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + num_words]) for i in range(len(words) - num_words + 1)}


def purge_leaks(*, train_texts: Sequence[str], train_labels: np.ndarray, test_texts: Sequence[str]) -> PurgeResult:
    """Drop training posts identical to a test post or at least ``LEAK_SHARE`` inside one."""

    test_hashes = {hashlib.sha1(text.encode()).hexdigest() for text in test_texts}
    postings: dict[str, list[int]] = {}
    for j, text in enumerate(test_texts):
        for shingle in shingles(text):
            postings.setdefault(shingle, []).append(j)
    identical, contained = [], []
    for i, text in enumerate(train_texts):
        if hashlib.sha1(text.encode()).hexdigest() in test_hashes:
            identical.append(i)
            continue
        post_shingles = shingles(text)
        hits = Counter(j for shingle in post_shingles for j in postings.get(shingle, ()))
        if post_shingles and hits and max(hits.values()) / len(post_shingles) >= LEAK_SHARE:
            contained.append(i)
    leaked = set(identical) | set(contained)
    keep = [i for i in range(len(train_texts)) if i not in leaked]
    return PurgeResult(
        texts=[train_texts[i] for i in keep],
        labels=train_labels[keep],
        num_identical=len(identical),
        num_contained=len(contained),
    )


def build_corpus(*, download_if_missing: bool = False) -> dict[str, Any]:
    """Rebuild the contents of ``corpus.json`` from raw 20 Newsgroups."""

    from sklearn.datasets import fetch_20newsgroups

    remove = ("headers", "footers", "quotes")
    raw_train = fetch_20newsgroups(subset="train", remove=remove, download_if_missing=download_if_missing)
    raw_test = fetch_20newsgroups(subset="test", remove=remove, download_if_missing=download_if_missing)
    example = next(text for text in raw_train.data if "@" in text and "----" in text)
    print("one post before cleaning:", repr(example[:200]))
    print("and after:               ", repr(clean_post(example)[:200]))
    train_texts, train_labels = _clean_split(raw_train)
    test_texts, test_labels = _clean_split(raw_test)
    print(f"train: {len(raw_train.data)} posts -> {len(train_texts)} with more than {MIN_CHARS} characters")
    print(f"test:  {len(raw_test.data)} posts -> {len(test_texts)}")
    purged = purge_leaks(train_texts=train_texts, train_labels=train_labels, test_texts=test_texts)
    print(
        f"leak purge: {purged.num_identical} training posts identical to a test post, "
        f"{purged.num_contained} at least {LEAK_SHARE:.0%} inside one; {len(purged.texts)} training posts left"
    )
    return {
        "label_names": list(raw_train.target_names),
        "train": {"text": purged.texts, "label": purged.labels.tolist()},
        "test": {"text": test_texts, "label": test_labels.tolist()},
    }


def check_or_write_corpus(*, paths: DataPaths, download_if_missing: bool = False) -> bool | None:
    """Rebuild the corpus. Compare with the saved file, or write it if there is none yet.

    Returns True when the rebuild matches the saved file, None when the raw download
    is not available and so nothing could be checked. A rebuild that differs raises.
    """

    try:
        rebuilt = build_corpus(download_if_missing=download_if_missing)
    except OSError as exc:
        print(
            f"could not rebuild the corpus ({type(exc).__name__}: {exc}).\n"
            "Download 20 Newsgroups once on a node with internet: "
            "python -c 'from sklearn.datasets import fetch_20newsgroups; fetch_20newsgroups()'"
        )
        return None
    if not paths.corpus.exists():
        refuse_protected_path(paths.corpus, protected_dirs=PROTECTED_DIRS)
        write_json(paths.corpus, rebuilt, indent=None)
        print("saved", paths.corpus)
        return True
    # A JSON round trip so that both sides hold plain lists and strings.
    if json.loads(json.dumps(rebuilt)) != read_json(paths.corpus):
        raise ValueError(f"The rebuilt corpus differs from {paths.corpus}, the one the models were trained on.")
    print("the rebuilt corpus is identical to", paths.corpus)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--download-if-missing", action="store_true")
    args = parser.parse_args(argv)
    check_or_write_corpus(paths=DataPaths(data_dir=args.data_dir), download_if_missing=args.download_if_missing)


if __name__ == "__main__":
    main()
