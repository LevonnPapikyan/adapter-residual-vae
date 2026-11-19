"""Two LDAs with exactly the same setup, one on English text and one on German text.

    python -m experiments.baselines.lda_pair --data-dir ../data --output-dir outputs/baselines

Both read the whole training posts (German being their machine translation) with
OCTIS's recipe in that language, a vocabulary of the words in at least 0.5% of the
posts, K = 20, batch variational Bayes, 100 passes, seed 42. The German topics are
matched to the English ones on the parallel posts; the test set is never used to fit
or to match. Also compares an English refit with the teacher, for scale.

What belongs here
-----------------
- This comparison's steps and its report. Needs spaCy.

What does NOT belong here
-------------------------
- Writing into the data folder; this baseline only reads it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import string
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from core.analysis.topic_matching import agreement_counts, map_onto, match_topics
from core.runs.io import parse_args_with_config, read_json, refuse_protected_path, write_json
from experiments.data.corpus import load_corpus
from experiments.data.paths import PROTECTED_DIRS, DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import load_translations

NUM_TOPICS, MIN_WORDS, MIN_DF, NUM_PASSES, LDA_SEED = 20, 5, 0.005, 100, 42
_STRIP = str.maketrans(string.punctuation + "0123456789", " " * (len(string.punctuation) + 10))


@dataclass(frozen=True)
class FittedLDA:
    """One LDA and its mixtures on the training posts and the test documents."""

    theta_train: np.ndarray
    theta_test: np.ndarray
    vocabulary: np.ndarray
    lda: LatentDirichletAllocation
    fit_rows: np.ndarray


def _words(texts: Sequence[str], *, nlp: Any) -> list[str]:
    out = []
    for doc in nlp.pipe((text.replace("\n", "").replace("\t", "").lower() for text in texts), batch_size=64):
        out.append(" ".join(" ".join(token.lemma_.lower() for token in doc if not token.is_stop).translate(_STRIP).split()))
    return out


def fit_lda(*, train_texts: Sequence[str], test_texts: Sequence[str], spacy_model: str) -> FittedLDA:
    """One LDA, the same recipe for every language."""

    import spacy

    nlp = spacy.load(spacy_model, disable=["parser", "ner"])
    words_train, words_test = _words(train_texts, nlp=nlp), _words(test_texts, nlp=nlp)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b", lowercase=False, min_df=MIN_DF)
    counts, counts_test = vectorizer.fit_transform(words_train), vectorizer.transform(words_test)
    words_per_post = np.asarray(counts.sum(1)).ravel()
    rows = words_per_post >= MIN_WORDS
    lda = LatentDirichletAllocation(
        n_components=NUM_TOPICS, learning_method="batch", max_iter=NUM_PASSES, random_state=LDA_SEED
    ).fit(counts[rows])
    print(
        f"{spacy_model}: vocabulary {counts.shape[1]} words; training posts median {np.median(words_per_post):.0f} words, "
        f"{rows.sum()} fit; test median {np.median(np.asarray(counts_test.sum(1))):.0f} words"
    )
    return FittedLDA(
        theta_train=lda.transform(counts),
        theta_test=lda.transform(counts_test),
        vocabulary=vectorizer.get_feature_names_out(),
        lda=lda,
        fit_rows=rows,
    )


def agreement_with_hellinger(*, theta_a: np.ndarray, theta_b_mapped: np.ndarray) -> dict[str, Any]:
    """Agreement counts, plus the median Hellinger distance between the mixtures."""

    hellinger = np.sqrt(((np.sqrt(theta_a) - np.sqrt(theta_b_mapped)) ** 2).sum(1) / 2)
    return {
        "n": len(theta_a),
        **agreement_counts(theta_a=theta_a, theta_b_mapped=theta_b_mapped),
        "hellinger_median": float(np.median(hellinger)),
    }


def run_lda_pair(*, data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Fit both LDAs and write ``report_lda_pair.json`` into ``output_dir``."""

    output_path = refuse_protected_path(Path(output_dir) / "report_lda_pair.json", protected_dirs=PROTECTED_DIRS)
    paths = DataPaths(data_dir=data_dir)
    train_en = load_corpus(paths=paths).train_texts
    train_de = read_json(paths.train_de)
    translated = load_translations(paths=paths)

    en = fit_lda(train_texts=train_en, test_texts=translated.texts["en"], spacy_model="en_core_web_sm")
    de = fit_lda(train_texts=train_de, test_texts=translated.texts["de"], spacy_model="de_core_news_sm")
    both = en.fit_rows & de.fit_rows  # posts both LDAs could read

    match = match_topics(theta_a=en.theta_train[both], theta_b=de.theta_train[both])
    pair = agreement_with_hellinger(theta_a=en.theta_test, theta_b_mapped=map_onto(theta_b=de.theta_test, to_a=match.to_a))

    # The same on whole posts: match on a random half of the training posts, compare on the other half.
    shuffled = np.random.default_rng(LDA_SEED).permutation(np.flatnonzero(both))
    fit_half, eval_half = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]
    match_half = match_topics(theta_a=en.theta_train[fit_half], theta_b=de.theta_train[fit_half])
    pair_whole = agreement_with_hellinger(
        theta_a=en.theta_train[eval_half],
        theta_b_mapped=map_onto(theta_b=de.theta_train[eval_half], to_a=match_half.to_a),
    )

    # For reference: how close the English refit is to the teacher (same text, same language).
    teacher = LDATeacher(paths=paths)
    match_teacher = match_topics(theta_a=teacher.theta(train_en)[en.fit_rows], theta_b=en.theta_train[en.fit_rows])
    vs_teacher = agreement_with_hellinger(
        theta_a=teacher.theta(translated.texts["en"]),
        theta_b_mapped=map_onto(theta_b=en.theta_test, to_a=match_teacher.to_a),
    )

    majority = int(np.bincount(en.theta_test.argmax(1)).max())
    report = {
        "english_lda_vs_german_lda": pair,
        "english_lda_vs_german_lda_whole_posts": pair_whole,
        "english_refit_vs_teacher": vs_teacher,
        "majority_english_lda": majority,
        "matching_correlation": match.correlation.tolist(),
        "matching_correlation_teacher": match_teacher.correlation.tolist(),
    }
    write_json(output_path, report, indent=1)

    print("\nOn the 1,000 test documents (English LDA on the English text vs German LDA on its translation), of 1,000:")
    print(f"  always the English LDA's most common topic   {majority}")
    print(f"  same top topic                              {pair['top_topic']}")
    print(f"  same top-2 pair                             {pair['same_pair']}")
    print(f"  at least one of the top 2 shared            {pair['one_shared']}")
    print(f"  English top topic in German top 1..5        {pair['topk'][:5]}")
    print(f"  Hellinger distance, median                  {pair['hellinger_median']:.3f}")
    num_whole = pair_whole["n"]
    print(f"\nOn {num_whole} whole training posts the matching did not see (English LDA on the post vs German LDA on its translation):")
    print(f"  same top topic                              {pair_whole['top_topic']} ({pair_whole['top_topic'] / num_whole:.0%})")
    print(f"  same top-2 pair                             {pair_whole['same_pair']} ({pair_whole['same_pair'] / num_whole:.0%})")
    print(f"  at least one of the top 2 shared            {pair_whole['one_shared']} ({pair_whole['one_shared'] / num_whole:.0%})")
    print(f"  Hellinger distance, median                  {pair_whole['hellinger_median']:.3f}")
    print(f"\nFor reference, the English refit vs the OCTIS teacher on the same English text: same top topic {vs_teacher['top_topic']}, top 1..5 {vs_teacher['topk'][:5]}")
    print("written to", output_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="An English and a German LDA with the same setup.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="outputs/baselines")
    args = parse_args_with_config(parser, argv)
    run_lda_pair(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
