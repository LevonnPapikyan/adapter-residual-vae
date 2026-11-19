"""Baseline: an LDA trained on German text.

    python -m experiments.baselines.lda_german --data-dir ../data --output-dir outputs/baselines

The training posts, whole, machine-translated into German the way the test set was;
OCTIS's recipe in German; K = 20. Its topics are matched one to one to the teacher's
on the parallel training posts, then it is scored on the 1,000 German test documents
exactly like the models.

What belongs here
-----------------
- This baseline's steps. Saved files in the data folder are reused; anything missing
  is made into ``output_dir``, never into the data folder.

What does NOT belong here
-------------------------
- The two-LDA comparison (``lda_pair.py``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from core.analysis.topic_matching import agreement_counts, map_onto, match_topics
from core.runs.io import parse_args_with_config, read_json, refuse_protected_path, write_json
from experiments.data.corpus import load_corpus
from experiments.data.paths import PROTECTED_DIRS, DataPaths
from experiments.data.teacher import LDATeacher, octis_words_cached
from experiments.data.translations import load_translations, translate

NUM_TOPICS, MIN_WORDS, MIN_DF, NUM_PASSES, LDA_SEED = 20, 5, 0.005, 100, 42


@dataclass(frozen=True)
class GermanLDAConfig:
    """Folders of this baseline. ``reference_results``, when given, adds the models' German scores to the printout."""

    data_dir: Path
    output_dir: Path
    reference_results: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "output_dir", refuse_protected_path(self.output_dir, protected_dirs=PROTECTED_DIRS))


def run_german_lda(*, config: GermanLDAConfig) -> dict:
    """Run the baseline and write ``report_german_lda.json`` into ``output_dir``."""

    paths = DataPaths(data_dir=config.data_dir)
    extra_words = config.output_dir / "octis_words_new.json"
    train_en = load_corpus(paths=paths).train_texts
    if paths.train_de.exists():
        train_de = read_json(paths.train_de)
    else:
        started = time.time()
        train_de = translate(texts=train_en, lang="de")
        write_json(config.output_dir / "train_de.json", train_de, indent=None)
        print(f"1. translated {len(train_de)} training posts to German in {time.time() - started:.0f} s")
    translated = load_translations(paths=paths)
    print("   e.g.", repr(train_de[0][:160]))

    started = time.time()
    words_train = octis_words_cached(train_de, paths=paths, lang="de", save_to=extra_words)
    words_test = octis_words_cached(translated.texts["de"], paths=paths, lang="de", save_to=extra_words)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b", lowercase=False, min_df=MIN_DF)
    counts = vectorizer.fit_transform(words_train)
    words_per_post = np.asarray(counts.sum(1)).ravel()
    fit_rows = words_per_post >= MIN_WORDS
    counts_test = vectorizer.transform(words_test)
    print(
        f"2. German vocabulary {counts.shape[1]} words ({time.time() - started:.0f} s); training posts: median "
        f"{np.median(words_per_post):.0f} words, {fit_rows.sum()} with >= {MIN_WORDS}; test documents: median "
        f"{np.median(np.asarray(counts_test.sum(1))):.0f} words"
    )

    saved = paths.teacher("lda_german.joblib")
    if saved.exists():  # fit once, seeded; later runs reuse it
        lda = joblib.load(saved)["lda"]
        print("3. loaded the German LDA from", saved)
    else:
        lda = LatentDirichletAllocation(
            n_components=NUM_TOPICS, learning_method="batch", max_iter=NUM_PASSES, random_state=LDA_SEED
        ).fit(counts[fit_rows])
        config.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vectorizer, "lda": lda, "prep": "octis_de"}, config.output_dir / "lda_german.joblib")
        print(f"3. fit {NUM_TOPICS} German topics, {NUM_PASSES} passes; saved {config.output_dir / 'lda_german.joblib'}")
    theta_de_train, theta_de_test = lda.transform(counts), lda.transform(counts_test)

    teacher = LDATeacher(paths=paths)
    theta_en_train = teacher.theta(train_en)
    match = match_topics(theta_a=theta_en_train[fit_rows], theta_b=theta_de_train[fit_rows])

    target = teacher.theta(translated.texts["en"])
    mapped = map_onto(theta_b=theta_de_test, to_a=match.to_a)
    report = {
        **agreement_counts(theta_a=target, theta_b_mapped=mapped),
        "teacher_on_german_text": int((teacher.theta(translated.texts["de"]).argmax(1) == target.argmax(1)).sum()),
        "majority": int(np.bincount(target.argmax(1)).max()),
        "matching_correlation": [float(value) for value in match.correlation],
    }
    output_path = config.output_dir / "report_german_lda.json"
    write_json(output_path, report, indent=1)

    print("\n5. German test documents (of 1,000): top topic vs LDA's top topic on the English original")
    print(f"   always the most common topic          {report['majority']}")
    print(f"   the English teacher on German text    {report['teacher_on_german_text']}")
    print(f"   LDA trained on German (this)          {report['top_topic']}   same pair {report['same_pair']}, one topic shared {report['one_shared']}")
    if config.reference_results is not None and Path(config.reference_results).exists():
        for name, scores in read_json(config.reference_results)["topic"].items():
            print(f"   {name:37s} {scores['de']}")
    print(f"   top-k for k = 1..5: {report['topk'][:5]}")
    print("   written to", output_path)

    def top_words(components: np.ndarray, vocabulary: np.ndarray, topic: int) -> str:
        return " ".join(vocabulary[np.argsort(-components[topic])[:6]])

    de_vocabulary, en_vocabulary = vectorizer.get_feature_names_out(), teacher.vectorizer.get_feature_names_out()
    print("\nGerman topic -> teacher topic (correlation of their shares over the training posts)")
    for j in np.argsort(-match.correlation):
        print(
            f"  {match.correlation[j]:+.2f}  {top_words(lda.components_, de_vocabulary, j):48s} -> {match.to_a[j]:2d}: "
            f"{top_words(teacher.lda.components_, en_vocabulary, match.to_a[j])}"
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="An LDA trained on German text.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--reference-results", default="../figures/paper/results.json")
    args = parse_args_with_config(parser, argv)
    run_german_lda(config=GermanLDAConfig(data_dir=args.data_dir, output_dir=args.output_dir, reference_results=args.reference_results))


if __name__ == "__main__":
    main()
