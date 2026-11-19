"""Score a trained pair: every number and figure the paper reports.

    python -m experiments.factorization.evaluate --config configs/paper/evaluation.json

Nothing is trained except the linear maps of the frozen baseline and the layer scan.
Writes ``results.json`` (same keys as the shipped file), ``comparison.json`` (this
run minus a reference ``results.json``), ``environment.json`` (the GPU and versions) and
the figures under ``output_dir``. An ``output_dir`` that already holds a ``results.json``
is refused unless ``--overwrite`` asks for it.

What belongs here
-----------------
- Loading what the scores need, and one function per score, in the order
  ``inference.ipynb`` calls them.

What does NOT belong here
-------------------------
- The definition of a score (``core/analysis``) and the drawing (``figures.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any

import numpy as np
import torch

from core.analysis.encoding import backbone_states, topic_mixtures
from core.analysis.generation import generate_by_topic, generation_score
from core.analysis.language_probe import language_score
from core.analysis.layers import layer_states, score_layers
from core.analysis.representation import TextIndex, finds_original, variance_split
from core.analysis.topic_score import majority_count, most_common_floor, topic_score, topk_counts
from core.factorization.checkpointing import LoadedRun, load_trained_run, recipe_to_saved_config
from core.factorization.config import TopicVAEConfig
from core.factorization.model import TopicVAE
from core.runs.io import read_json
from core.train.loop import resolve_device
from experiments.baselines.frozen_linear_map import fit_frozen_linear_map
from experiments.data.corpus import Corpus, load_corpus
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import LANG_NAMES, LANGS, NUM_DOCS, TranslatedTestSet, check_alignment, load_translations
from experiments.factorization import figures
from experiments.factorization.config import EvaluationConfig, parse_evaluation_config
from experiments.factorization.output import (
    LABEL_DELTA,
    LABEL_FROZEN,
    LABEL_H,
    LABEL_LDA,
    LABEL_STATE_DELTA,
    LABEL_STATE_H,
    LABEL_STATE_OFF,
    gpu_name,
    print_row,
    refuse_existing_results,
    write_comparison,
    write_environment,
    write_results,
)

FOLLOWED_DOC = 0  # the one document followed through the figures


@dataclass(frozen=True)
class EvaluationBundle:
    """Everything the scores read.

    Attributes
    ----------
    corpus, translated, teacher:
        The data and the LDA teacher.
    train_texts:
        Training posts for the linear maps.
    target:
        The teacher on the English text: the answer in every language, ``[docs, topics]``.
    theta_train:
        The teacher on ``train_texts``.
    majority:
        Always answering the most common topic.
    index:
        Language and document of each row of the stacked ``[5 * docs, ...]`` arrays.
    all_texts, topic:
        The stacked texts, and the teacher's top topic of each text's English original.
    runs:
        The two trained runs by their label in ``results.json``.
    frozen:
        Plain language model, no encoder LoRA: what it already knows.
    device:
        Where the models live.
    """

    config: EvaluationConfig
    corpus: Corpus
    translated: TranslatedTestSet
    teacher: LDATeacher
    train_texts: list[str]
    target: np.ndarray
    theta_train: np.ndarray
    majority: int
    index: TextIndex
    all_texts: list[str]
    topic: np.ndarray
    runs: dict[str, LoadedRun]
    frozen: TopicVAE
    device: str

    @property
    def models(self) -> dict[str, TopicVAE]:
        """The two trained models by label."""

        return {name: run.model for name, run in self.runs.items()}

    @property
    def num_docs(self) -> int:
        """Documents per language."""

        return self.translated.num_docs


@dataclass(frozen=True)
class TopicScores:
    """Score 1: mixtures per predictor and language, the counts, and two reference numbers."""

    thetas: dict[str, dict[str, np.ndarray]]
    scores: dict[str, dict[str, int]]
    german_lda: int
    refit_vs_teacher: int


def load_evaluation_bundle(*, config: EvaluationConfig) -> EvaluationBundle:
    """Load the data, the teacher, and the three models, with the checks everything rests on."""

    paths = DataPaths(data_dir=config.data_dir)
    device = resolve_device(config.device)
    corpus = load_corpus(paths=paths)
    translated = load_translations(paths=paths)
    check_alignment(translated=translated, corpus=corpus)
    translated = translated.first(config.max_docs)
    train_texts = corpus.train_texts if config.max_train_texts is None else corpus.train_texts[: config.max_train_texts]
    print(f"{len(corpus.train_texts):,} training posts, {len(corpus.test_texts):,} test posts, {len(corpus.label_names)} newsgroups")
    print(
        f"translated test set: {translated.num_docs:,} documents x {len(LANGS)} languages; empty texts per language: "
        f"{ {lang: sum(not text.strip() for text in translated.texts[lang]) for lang in LANGS} }"
    )

    teacher = LDATeacher(paths=paths)
    target = teacher.theta(translated.texts["en"])
    theta_train = teacher.theta(train_texts)
    majority = majority_count(target=target)
    print(
        f"teacher: K = {teacher.num_topics}, vocabulary {len(teacher.vectorizer.get_feature_names_out()):,} words | "
        f"target {target.shape}, theta_train {theta_train.shape} | majority {majority}"
    )

    index = TextIndex(num_langs=len(LANGS), num_docs=translated.num_docs)
    all_texts = translated.stacked()
    topic = np.tile(target.argmax(1), len(LANGS))

    run_h = load_trained_run(run_dir=config.runs_dir / "h", device=device, encoder_max_tokens=config.encoder_max_tokens)
    run_delta = load_trained_run(run_dir=config.runs_dir / "delta", device=device, encoder_max_tokens=config.encoder_max_tokens)
    saved_h, saved_delta = recipe_to_saved_config(run_h.recipe), recipe_to_saved_config(run_delta.recipe)
    differ = {key for key, value in saved_h.items() if value != saved_delta[key]}
    if differ != {"mu_from"}:
        raise ValueError(f"The two models should differ in mu_from only, they differ in {differ}.")
    if run_h.recipe.model.base_model != config.base_model:
        raise ValueError(f"base_model is {config.base_model} but the models were trained on {run_h.recipe.model.base_model}.")
    frozen_config = TopicVAEConfig(
        base_model=config.base_model,
        train_encoder_lora=False,
        encoder_max_tokens=config.encoder_max_tokens or TopicVAEConfig().encoder_max_tokens,
    )
    frozen = TopicVAE(config=frozen_config).to(device).eval()
    print("settings shared by the two models:", {key: value for key, value in saved_h.items() if key != "mu_from"})
    return EvaluationBundle(
        config=config,
        corpus=corpus,
        translated=translated,
        teacher=teacher,
        train_texts=train_texts,
        target=target,
        theta_train=theta_train,
        majority=majority,
        index=index,
        all_texts=all_texts,
        topic=topic,
        runs={LABEL_H: run_h, LABEL_DELTA: run_delta},
        frozen=frozen,
        device=device,
    )


def score_topics(*, bundle: EvaluationBundle) -> TopicScores:
    """Score 1 for every predictor. The frozen map starts from random weights, so torch is seeded first."""

    config = bundle.config
    torch.manual_seed(config.seed)
    frozen_map = fit_frozen_linear_map(
        model=bundle.frozen,
        train_texts=bundle.train_texts,
        theta_train=bundle.theta_train,
        device=bundle.device,
        steps=config.linear_map_steps,
    )
    thetas: dict[str, dict[str, np.ndarray]] = {LABEL_LDA: {}, LABEL_FROZEN: {}, LABEL_H: {}, LABEL_DELTA: {}}
    for lang in LANGS:
        texts = bundle.translated.texts[lang]
        thetas[LABEL_LDA][lang] = bundle.teacher.theta(texts)
        thetas[LABEL_FROZEN][lang] = frozen_map.predict_texts(texts=texts)
        thetas[LABEL_H][lang] = topic_mixtures(model=bundle.runs[LABEL_H].model, texts=texts)
        thetas[LABEL_DELTA][lang] = topic_mixtures(model=bundle.runs[LABEL_DELTA].model, texts=texts)
        print(f"{LANG_NAMES[lang]:8s} " + ", ".join(f"{name} {per_lang[lang].shape}" for name, per_lang in thetas.items()), flush=True)
    for name, per_lang in thetas.items():
        for lang in LANGS:
            if per_lang[lang].shape != bundle.target.shape or not np.allclose(per_lang[lang].sum(1), 1, atol=1e-3):
                raise ValueError(f"{name} on {lang} is not a set of mixtures of shape {bundle.target.shape}.")

    scores = {
        name: {lang: topic_score(theta_pred=per_lang[lang], target=bundle.target) for lang in LANGS}
        for name, per_lang in thetas.items()
    }
    german_lda = read_json(config.baselines_dir / "report_german_lda.json")["top_topic"]
    refit = read_json(config.baselines_dir / "report_lda_pair.json")["english_refit_vs_teacher"]["top_topic"]

    print()
    print_row(f"top topic right (of {bundle.num_docs:,})", [LANG_NAMES[lang] for lang in LANGS])
    print_row("always the most common topic", [bundle.majority] * len(LANGS))
    print_row("LDA trained on German", [german_lda if lang == "de" else "" for lang in LANGS])
    for name, per_lang in scores.items():
        print_row(name, [per_lang[lang] for lang in LANGS])
    print_row("Δh - h", [f"{scores[LABEL_DELTA][lang] - scores[LABEL_H][lang]:+d}" for lang in LANGS])
    print(f"\nFor scale: a second English LDA, refit on the same training posts and matched topic to topic to the teacher, has the teacher's top topic on {refit} of the 1,000 English documents.")
    return TopicScores(thetas=thetas, scores=scores, german_lda=german_lda, refit_vs_teacher=refit)


def topk_tables(*, bundle: EvaluationBundle, topic_scores: TopicScores) -> dict[str, dict[str, list[int]]]:
    """The teacher's top topic within the top k, for the three learned predictors."""

    print(f"the teacher's θ puts a median {np.median(bundle.target.max(1)):.0%} of its mass on its top topic")
    return {
        name: {lang: topk_counts(theta_pred=topic_scores.thetas[name][lang], target=bundle.target) for lang in LANGS}
        for name in (LABEL_FROZEN, LABEL_H, LABEL_DELTA)
    }


def score_language(*, topic_scores: TopicScores) -> dict[str, float]:
    """Score 2 for the three learned predictors. Chance is 0.20; lower is better."""

    scores = {name: language_score(thetas=topic_scores.thetas[name]) for name in (LABEL_FROZEN, LABEL_H, LABEL_DELTA)}
    for name, value in scores.items():
        print(f"{name:26s} {value:.2f}")
    return scores


def look_inside(*, bundle: EvaluationBundle) -> dict[str, dict[str, Any]]:
    """Finds-original and the variance split, on plain Qwen, the h model's state, and Δh."""

    states_delta = backbone_states(model=bundle.runs[LABEL_DELTA].model, texts=bundle.all_texts)
    states_h = backbone_states(model=bundle.runs[LABEL_H].model, texts=bundle.all_texts)
    print(f"states {states_delta.on.shape}; both models' h_off are plain Qwen, max |difference| {np.abs(states_h.off - states_delta.off).max():.2e}")
    representations = {
        LABEL_STATE_OFF: states_delta.off,
        LABEL_STATE_H: states_h.on,
        LABEL_STATE_DELTA: states_delta.on - states_delta.off,
    }
    print()
    print_row("finds its English original", [LANG_NAMES[lang] for lang in LANGS[1:]] + ["language", "document"])
    inside = {}
    for name, features in representations.items():
        found = [finds_original(features=features, lang_index=j, index=bundle.index) for j in range(1, len(LANGS))]
        split = variance_split(features=features, index=bundle.index)
        inside[name] = {"finds_original": found, "language": split.language, "document": split.document}
        print_row(name, found + [f"{split.language:.0%}", f"{split.document:.0%}"])
    size = np.mean(np.linalg.norm(states_delta.on - states_delta.off, axis=1) / np.linalg.norm(states_delta.off, axis=1))
    print(f"\nsize of Δh relative to h_off: {size:.2f}")
    return inside


def score_generation(*, bundle: EvaluationBundle) -> tuple[dict[str, dict[int, list[str]]], dict[str, dict[int, int]]]:
    """Score 3: the written texts and how many the teacher puts back in their topic."""

    config = bundle.config
    test_texts = bundle.corpus.test_texts
    if config.max_generation_texts is not None:
        test_texts = test_texts[: config.max_generation_texts]
    torch.manual_seed(config.seed)  # sampling z, and the generator's top-k sampling
    generated = {
        name: generate_by_topic(
            model=model,
            teacher=bundle.teacher,
            texts=test_texts,
            per_topic=config.generation_per_topic,
            max_new_tokens=config.generation_max_new_tokens,
        )
        for name, model in bundle.models.items()
    }
    gen_scores = {name: generation_score(teacher=bundle.teacher, generated=texts) for name, texts in generated.items()}
    topics = sorted(gen_scores[LABEL_H])
    words = bundle.teacher.top_words(4)
    per_topic = config.generation_per_topic
    print(f"{len(topics)} topics with at least one test post; {sum(len(texts) for texts in generated[LABEL_H].values())} texts written per model\n")
    print_row("topic", list(bundle.models))
    for k in topics:
        print_row(f"{k:2d} {words[k]}", [f"{gen_scores[name][k]}/{per_topic}" for name in bundle.models])
    print_row("total", [f"{sum(gen_scores[name].values())}/{per_topic * len(topics)}" for name in bundle.models])
    return generated, gen_scores


def scan_layers(*, bundle: EvaluationBundle) -> list[dict[str, Any]]:
    """The three scores of plain Qwen after every block. The slow part."""

    train_states = layer_states(model=bundle.frozen, texts=bundle.train_texts)
    test_states = layer_states(model=bundle.frozen, texts=bundle.all_texts)
    print(f"states after every block: training posts {train_states.shape}, test texts {test_states.shape}")
    torch.manual_seed(bundle.config.seed)  # the linear maps start from random weights
    return score_layers(
        train_states=train_states,
        test_states=test_states,
        theta_train=bundle.theta_train,
        target=bundle.target,
        langs=LANGS,
        index=bundle.index,
        device=bundle.device,
        linear_map_steps=bundle.config.linear_map_steps,
    )


def assemble_results(
    *,
    bundle: EvaluationBundle,
    topic_scores: TopicScores,
    lang_scores: dict[str, float],
    gen_scores: dict[str, dict[int, int]],
    inside: dict[str, dict[str, Any]],
    layers: list[dict[str, Any]],
) -> dict[str, Any]:
    """``results.json``, with the shipped file's keys in the shipped file's order."""

    return {
        "topic": topic_scores.scores,
        "language": lang_scores,
        "generation": gen_scores,
        "inside": inside,
        "majority": bundle.majority,
        "layers": layers,
    }


def summarize_training(*, bundle: EvaluationBundle) -> None:
    """Print the per-epoch table of the two runs and draw ``training.pdf`` from their saved files."""

    runs = {name: (run.history, read_json(run.run_dir / "test.json")) for name, run in bundle.runs.items()}
    for name, (history, test_rows) in runs.items():
        epochs = test_rows[1:]  # one row per epoch; row 0 is before training
        kept = min(epochs, key=lambda row: row["val"])["epoch"]
        print(f"{name}: {len(history)} logged steps, kept epoch {kept}")
        print_row("  epoch", ["held-out", "rec/token", "align", "en", "de"])
        for row in epochs:
            print_row(f"  {row['epoch']}", [f"{row['val']:.3f}", f"{row['rec']:.3f}", f"{row['align']:.3f}", row["en"], row["de"]])
    figures.plot_training(runs=runs, num_docs=NUM_DOCS, output_dir=bundle.config.output_dir)


def run_evaluation(*, config: EvaluationConfig) -> dict[str, Any]:
    """Every score and figure, in the notebook's order. Returns the results."""

    output_dir = config.output_dir
    refuse_existing_results(output_dir=output_dir, overwrite=config.overwrite)
    bundle = load_evaluation_bundle(config=config)
    teacher, target = bundle.teacher, bundle.target
    figures.plot_theta_example(theta=target[FOLLOWED_DOC], top_words=teacher.top_words(4), doc=FOLLOWED_DOC, output_dir=output_dir)
    summarize_training(bundle=bundle)

    topic_scores = score_topics(bundle=bundle)
    figures.plot_topic_score(scores=topic_scores.scores, majority=bundle.majority, num_docs=bundle.num_docs, output_dir=output_dir)
    figures.plot_topk(
        topk=topk_tables(bundle=bundle, topic_scores=topic_scores),
        floor=most_common_floor(target=target),
        num_docs=bundle.num_docs,
        output_dir=output_dir,
    )
    trained_thetas = {name: topic_scores.thetas[name] for name in (LABEL_H, LABEL_DELTA)}
    figures.plot_one_document(thetas=trained_thetas, top_topic=int(target[FOLLOWED_DOC].argmax()), doc=FOLLOWED_DOC, output_dir=output_dir)

    lang_scores = score_language(topic_scores=topic_scores)
    figures.plot_language_score(lang_scores=lang_scores, output_dir=output_dir)
    figures.plot_latent_space(
        thetas=trained_thetas,
        topic=bundle.topic,
        text_lang=bundle.index.text_lang,
        top_words_two=teacher.top_words(2),
        output_dir=output_dir,
    )
    inside = look_inside(bundle=bundle)

    gen_scores: dict[str, dict[int, int]] = {}
    if not config.skip_generation:
        _, gen_scores = score_generation(bundle=bundle)
        figures.plot_generation_score(gen_scores=gen_scores, num_topics=teacher.num_topics, per_topic=config.generation_per_topic, output_dir=output_dir)
    layers: list[dict[str, Any]] = []
    if not config.skip_layers:
        layers = scan_layers(bundle=bundle)
        figures.plot_layers(layers=layers, majority=bundle.majority, num_docs=bundle.num_docs, output_dir=output_dir)

    results = assemble_results(
        bundle=bundle, topic_scores=topic_scores, lang_scores=lang_scores, gen_scores=gen_scores, inside=inside, layers=layers
    )
    write_results(output_dir=output_dir, results=results)
    write_comparison(
        output_dir=output_dir,
        results=json.loads(json.dumps(results, ensure_ascii=False)),  # the same string keys the reference has
        reference_path=config.reference_results,
        is_partial=config.is_partial,
    )
    write_environment(output_dir=output_dir, device=bundle.device)
    return results


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers.utils import logging

    logging.disable_progress_bar()
    config = parse_evaluation_config(argv)
    print(f"device {resolve_device(config.device)} | gpu {gpu_name(resolve_device(config.device))} | torch {torch.__version__} | models from {config.runs_dir} | base {config.base_model} | outputs to {config.output_dir}")
    print(asdict(config))
    run_evaluation(config=config)


if __name__ == "__main__":
    main()
