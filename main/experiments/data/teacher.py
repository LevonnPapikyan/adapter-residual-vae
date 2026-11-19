"""The teacher: an LDA on OCTIS's preprocessing, and that preprocessing.

What belongs here
-----------------
- ``LDATeacher``, the fitted model as training and evaluation use it.
- OCTIS's recipe (lowercase, spaCy lemmas, spaCy's stop words dropped, punctuation
  and digits removed) with its word cache, and the fit that made ``lda_octis.joblib``.

What does NOT belong here
-------------------------
- The German baselines (``experiments/baselines``).

Run ``python -m experiments.data.teacher --config configs/paper/teacher.json`` to
refit the teacher into ``outputs/teacher/``. The saved teacher is never overwritten.
"""

from __future__ import annotations

import argparse
from collections import ChainMap
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import string
import time
from typing import Any, Sequence

import joblib
import numpy as np

from core.runs.io import parse_args_with_config, read_json, refuse_protected_path, write_json
from experiments.data.corpus import load_corpus
from experiments.data.paths import PROTECTED_DIRS, DataPaths

SPACY_MODELS = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
_STRIP = str.maketrans(string.punctuation + "0123456789", " " * (len(string.punctuation) + 10))
_loaded_spacy: dict[str, Any] = {}


def octis_words(texts: Sequence[str], *, lang: str = "en") -> list[str]:
    """OCTIS's preprocessing. German lemmas are lowercased, since German nouns are capitalized."""

    if lang not in SPACY_MODELS:
        raise ValueError(f"lang must be one of {sorted(SPACY_MODELS)}, got {lang!r}.")
    if lang not in _loaded_spacy:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                f"{len(texts)} texts are missing from the word cache, and preprocessing them needs spaCy with {SPACY_MODELS[lang]}."
            ) from exc

        _loaded_spacy[lang] = spacy.load(SPACY_MODELS[lang], disable=["parser", "ner"])  # the lemmas need only the tagger
    lemma = (lambda token: token.lemma_) if lang == "en" else (lambda token: token.lemma_.lower())
    out = []
    cleaned = (text.replace("\n", "").replace("\t", "").lower() for text in texts)
    for doc in _loaded_spacy[lang].pipe(cleaned, batch_size=64):
        out.append(" ".join(" ".join(lemma(token) for token in doc if not token.is_stop).translate(_STRIP).split()))
    return out


@lru_cache(maxsize=4)
def _read_word_file(path: Path, modified: float) -> dict[str, str]:
    return read_json(path)


def _word_file(path: Path) -> dict[str, str]:
    return _read_word_file(path, path.stat().st_mtime) if path.exists() else {}


def octis_words_cached(
    texts: Sequence[str],
    *,
    paths: DataPaths,
    lang: str = "en",
    save_to: str | Path | None = None,
) -> list[str]:
    """``octis_words``, read from the data folder's cache where possible.

    ``save_to`` keeps words for texts the cache did not know in a separate file,
    which later calls with the same ``save_to`` also read. It is never the shared
    cache in a protected folder.
    """

    extra_path = None if save_to is None else refuse_protected_path(save_to, protected_dirs=PROTECTED_DIRS)
    prefix = "" if lang == "en" else f"{lang}:"  # English keys have no prefix: the cache predates German

    def key(text: str) -> str:
        return prefix + hashlib.sha1(text.encode()).hexdigest()[:16]

    known = ChainMap({}, _word_file(extra_path) if extra_path else {}, _word_file(paths.octis_words))
    missing = [text for text in dict.fromkeys(texts) if key(text) not in known]
    if missing:
        known.maps[0].update(zip(map(key, missing), octis_words(missing, lang=lang)))
        if extra_path is not None:
            write_json(extra_path, {**known.maps[1], **known.maps[0]}, indent=None)
    return [known[key(text)] for text in texts]


class LDATeacher:
    """A fitted teacher. ``theta(texts)`` is LDA's topic mixture for each text.

    Parameters
    ----------
    paths:
        The data folder that holds the fitted model and the word cache.
    name:
        File name of the fitted model. ``lda_octis.joblib`` is the teacher the
        models were trained to match; ``lda_german.joblib`` is the German one.
    """

    def __init__(self, *, paths: DataPaths, name: str = "lda_octis.joblib") -> None:
        self.paths = paths
        saved = joblib.load(paths.teacher(name))
        self.vectorizer, self.lda = saved["vectorizer"], saved["lda"]
        self.prep = saved.get("prep")  # "octis" / "octis_de": that preprocessing first; None: raw text
        self.num_topics = int(self.lda.n_components)

    def words(self, texts: Sequence[str]) -> Sequence[str]:
        """What the vectorizer reads."""

        if self.prep is None:
            return texts
        return octis_words_cached(texts, paths=self.paths, lang="de" if self.prep == "octis_de" else "en")

    def counts(self, texts: Sequence[str]) -> Any:
        """``[texts, vocabulary]`` word counts."""

        return self.vectorizer.transform(self.words(texts))

    def theta(self, texts: Sequence[str]) -> np.ndarray:
        """``[texts, num_topics]`` mixtures."""

        return self.lda.transform(self.counts(texts))

    def top_words(self, num_words: int = 6) -> list[str]:
        """The most likely words of each topic."""

        vocabulary = np.array(self.vectorizer.get_feature_names_out())
        return [" ".join(vocabulary[np.argsort(-row)[:num_words]]) for row in self.lda.components_]


@dataclass(frozen=True)
class TeacherConfig:
    """Settings of the OCTIS teacher's fit."""

    data_dir: Path
    output_dir: Path
    num_topics: int = 20
    num_passes: int = 100
    lda_seed: int = 42
    min_words: int = 5
    vocabulary_size: int = 1612

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.num_topics <= 0 or self.num_passes <= 0 or self.min_words < 0:
            raise ValueError(f"Invalid teacher settings: {self}.")


@dataclass(frozen=True)
class OctisCounts:
    """What LDA sees of the training posts.

    Attributes
    ----------
    words:
        OCTIS's preprocessing of each post.
    vocabulary, vectorizer:
        OCTIS's word list and the counter restricted to it.
    counts:
        ``[posts, vocabulary]`` word counts.
    words_per_post, fit_rows:
        Vocabulary words per post, and which posts have enough to go into the fit.
    exact_octis_documents, long_enough_posts:
        How many long-enough posts are word for word an OCTIS document, out of how many.
    """

    words: list[str]
    vocabulary: list[str]
    vectorizer: Any
    counts: Any
    words_per_post: np.ndarray
    fit_rows: np.ndarray
    exact_octis_documents: int
    long_enough_posts: int


def build_octis_counts(
    *,
    paths: DataPaths,
    train_texts: Sequence[str],
    min_words: int = 5,
    vocabulary_size: int = 1612,
    save_to: str | Path | None = None,
) -> OctisCounts:
    """Preprocess the training posts and count them over OCTIS's vocabulary."""

    from sklearn.feature_extraction.text import CountVectorizer

    words = octis_words_cached(train_texts, paths=paths, save_to=save_to)
    vocabulary = (paths.octis_dir / "vocabulary.txt").read_text().split()
    if len(vocabulary) != vocabulary_size:
        raise ValueError(f"Expected {vocabulary_size} OCTIS words, found {len(vocabulary)}.")
    keep = set(vocabulary)
    octis_documents = {line.split("\t")[0] for line in (paths.octis_dir / "corpus.tsv").read_text().splitlines()}
    in_vocabulary = [" ".join(word for word in post.split() if word in keep) for post in words]
    long_enough = [post for post in in_vocabulary if len(post.split()) >= min_words]
    vectorizer = CountVectorizer(vocabulary=vocabulary, token_pattern=r"(?u)\S+", lowercase=False)
    counts = vectorizer.transform(words)
    words_per_post = np.asarray(counts.sum(1)).ravel()
    return OctisCounts(
        words=words,
        vocabulary=vocabulary,
        vectorizer=vectorizer,
        counts=counts,
        words_per_post=words_per_post,
        fit_rows=words_per_post >= min_words,
        exact_octis_documents=sum(post in octis_documents for post in long_enough),
        long_enough_posts=len(long_enough),
    )


def fit_octis_teacher(*, config: TeacherConfig) -> Path:
    """Fit LDA on the training posts over OCTIS's vocabulary and save it in ``output_dir``."""

    from sklearn.decomposition import LatentDirichletAllocation

    output_path = refuse_protected_path(config.output_dir / "lda_octis.joblib", protected_dirs=PROTECTED_DIRS)
    if output_path.exists():
        raise FileExistsError(f"{output_path} exists; choose another output_dir.")
    paths = DataPaths(data_dir=config.data_dir)
    train_texts = load_corpus(paths=paths).train_texts
    octis = build_octis_counts(
        paths=paths,
        train_texts=train_texts,
        min_words=config.min_words,
        vocabulary_size=config.vocabulary_size,
        save_to=config.output_dir / "octis_words_new.json",
    )
    print(
        f"counts {octis.counts.shape}, {octis.counts.nnz:,} non-zero; median {np.median(octis.words_per_post):.0f} vocabulary words per post; "
        f"{octis.fit_rows.sum()} of {len(train_texts)} posts have >= {config.min_words} and go into the fit"
    )
    started = time.time()
    lda = LatentDirichletAllocation(
        n_components=config.num_topics, learning_method="batch", max_iter=config.num_passes, random_state=config.lda_seed
    )
    lda.fit(octis.counts[octis.fit_rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": octis.vectorizer, "lda": lda, "prep": "octis"}, output_path)
    print(f"fit {config.num_topics} topics, {config.num_passes} passes, in {time.time() - started:.0f} s; saved {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the OCTIS teacher into outputs/.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="outputs/teacher")
    parser.add_argument("--num-topics", type=int, default=20)
    parser.add_argument("--num-passes", type=int, default=100)
    parser.add_argument("--lda-seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=5)
    fit_octis_teacher(config=TeacherConfig(**{k: v for k, v in vars(parse_args_with_config(parser, argv)).items() if k != "config"}))


if __name__ == "__main__":
    main()
