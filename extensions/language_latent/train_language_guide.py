"""The mirror experiment: the guide is the language of the text, not LDA's topics.

    cd main && python -u ../extensions/language_latent/train_language_guide.py --config ../extensions/language_latent/delta.json [--held-out-topics]

The paper aligns the latent to LDA's topic mixture and finds that reading mu from Delta h = h_on - h_off keeps the
language out of the latent. Here the same model (Qwen2.5-0.5B, LoRA, mu from h or from Delta h, same loss) is aligned to
the language instead: the latent has 5 coordinates, one per language, and the alignment is the cross-entropy between
softmax(mu) and the one-hot language of the text. The question is the mirror one: does the Delta h readout hold less
topic than the h readout?

Data. The training posts in English, German, French, Spanish and Chinese, the first 150 words in every language
(so length cannot give the language away). All five versions of a post are trained on or held out together.
With --held-out-topics only the posts of every second newsgroup (label 0, 2, ..., 18) are trained on, which mirrors
the paper training its guide on English only; the other ten newsgroups are then unseen topics.

Scores, on the 1,000 test documents in five languages (never trained on):
- language right: argmax mu is the text's language; per language, and the worst newsgroup;
- topic in the latent: 5-fold probe from mu (and from softmax(mu)) to the newsgroup label and to LDA's top topic,
  all versions of a document in the same fold;
- the same probes, and a language probe, on the 896-number readouts: plain Qwen's h_off, this run's readout,
  the paper's seed 0 readout of the same kind, and Delta h of a random LoRA (this run's LoRA tensors replaced by
  noise of the same size per tensor);
- writing: from each document's sampled z, and from each language's mean mu, which language is written.

This folder imports main/ and writes nothing under it.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.factorization.checkpointing import load_trained_run
from core.factorization.model import TopicVAE
from core.factorization.posterior import sample_z
from core.runs.io import read_json
from core.train.loop import resolve_device, split_held_out
from experiments.data.corpus import load_corpus
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import LANG_NAMES, LANGS, NUM_WORDS, load_translations
import experiments.factorization.train as train_module

HERE = Path(__file__).resolve().parent
EXP9 = HERE.parent / "multilingual_decode"
sys.path.insert(0, str(EXP9))
from train_multilingual import detect_language  # noqa: E402  (the language detector of multilingual_decode)

BATCH = 32
GEN_LEN = 96
HELD_OUT_TOPICS = "--held-out-topics" in sys.argv


def trained_newsgroup(label: int) -> bool:
    """With --held-out-topics, the newsgroups trained on: every second one."""

    return label % 2 == 0


def one_hot(lang: str, count: int) -> torch.Tensor:
    out = torch.zeros(count, len(LANGS))
    out[:, LANGS.index(lang)] = 1.0
    return out


def load_language_training_data(*, config, rng: np.random.Generator):
    """Every kept training post in five languages, first 150 words, with its language as the target."""

    recipe = config.recipe
    paths = DataPaths(data_dir=config.data_dir)
    corpus = load_corpus(paths=paths)
    versions = {
        "en": [" ".join(text.split()[:NUM_WORDS]) for text in corpus.train_texts],
        "de": [" ".join(text.split()[:NUM_WORDS]) for text in read_json(paths.train_de)],
        **{lang: read_json(EXP9 / "data" / f"train_{lang}.json") for lang in ("fr", "es", "zh")},
    }
    if any(len(texts) != len(corpus.train_texts) for texts in versions.values()):
        raise ValueError("Every language must hold every training post.")
    index = np.arange(len(corpus.train_texts) if config.max_train_texts is None else min(len(corpus.train_texts), config.max_train_texts))
    teacher = LDATeacher(paths=paths, name=recipe.teacher)  # only to keep the posts the paper keeps
    words_per_post = np.asarray(teacher.counts([corpus.train_texts[i] for i in index]).sum(1)).ravel()
    index = index[words_per_post >= recipe.min_words]
    index = np.array([i for i in index if all(versions[lang][i].strip() for lang in LANGS)])
    if HELD_OUT_TOPICS:
        index = np.array([i for i in index if trained_newsgroup(int(corpus.train_labels[i]))])
        names = [name for label, name in enumerate(corpus.label_names) if trained_newsgroup(label)]
        print(f"held-out topics: training on {len(names)} newsgroups only: {names}", flush=True)
    split = split_held_out(rng=rng, num_texts=len(index), val_frac=recipe.val_frac)

    def stack(posts: np.ndarray) -> tuple[list[str], torch.Tensor]:
        texts = [versions[lang][i] for lang in LANGS for i in posts]
        return texts, torch.cat([one_hot(lang, len(posts)) for lang in LANGS])

    texts, theta = stack(index[split.kept])
    val_texts, val_theta = stack(index[split.held_out])
    print(f"{len(split.kept)} posts x {len(LANGS)} languages trained on, {len(split.held_out)} posts x {len(LANGS)} held out", flush=True)
    test_docs = load_translations(paths=paths).first(config.max_test_docs)
    return train_module.TrainingData(
        texts=texts,
        theta=theta,
        val_texts=val_texts,
        val_theta=val_theta,
        test_docs=test_docs,
        theta_test=np.zeros((test_docs.num_docs, len(LANGS)), dtype=np.float32),  # unused by the watch below
    )


@torch.no_grad()
def watch_language_scores(*, model: TopicVAE, docs: Mapping[str, Sequence[str]], theta_target: np.ndarray, batch_size: int = BATCH) -> dict:
    """After every epoch: language right per language on the test documents, and English reconstruction at z = mu."""

    model.eval()
    out: dict = {}
    rec = 0.0
    for k, lang in enumerate(LANGS):
        right = 0
        for i in range(0, len(docs[lang]), batch_size):
            batch = docs[lang][i : i + batch_size]
            mu = model.posterior(texts=batch).mu
            right += int((mu.argmax(1) == k).sum())
            if lang == "en":
                reconstruction = model.reconstruction_nll(z=mu, texts=batch)
                rec += (reconstruction.nll / reconstruction.num_tokens).sum().item()
        out[lang] = right
    model.train()
    return {"rec": rec / len(docs["en"]), **out}


@torch.no_grad()
def state(model: TopicVAE, texts: Sequence[str], *, adapter: bool) -> np.ndarray:
    """The backbone's last-token state of every text, with the LoRA on or off, float32."""

    out = []
    for i in range(0, len(texts), BATCH):
        ids, mask = model.tokenize(texts=texts[i : i + BATCH], max_tokens=model.config.encoder_max_tokens)
        if adapter:
            out.append(model.backbone_state(input_ids=ids, attention_mask=mask).cpu())
        else:
            with model.posterior_backbone.disable_adapter():
                out.append(model.backbone_state(input_ids=ids, attention_mask=mask).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def moments(model: TopicVAE, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``mu`` and one sampled ``z`` per text."""

    mus, zs = [], []
    for i in range(0, len(texts), BATCH):
        posterior = model.posterior(texts=texts[i : i + BATCH])
        mus.append(posterior.mu.float().cpu())
        zs.append(sample_z(mu=posterior.mu, logvar=posterior.logvar).float().cpu())
    return torch.cat(mus).numpy(), torch.cat(zs).numpy()


def probe(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """5-fold accuracy of a linear probe, all versions of a document in the same fold."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        return round(float(cross_val_score(model, features, labels, groups=groups, cv=GroupKFold(5), n_jobs=4).mean()), 3)


def three_probes(features: np.ndarray, *, language: np.ndarray, newsgroup: np.ndarray, lda_top: np.ndarray, groups: np.ndarray) -> dict:
    return {
        "language": probe(features, language, groups),
        "newsgroup": probe(features, newsgroup, groups),
        "lda_top_topic": probe(features, lda_top, groups),
    }


def randomize_lora(model: TopicVAE, *, seed: int = 0) -> None:
    """Replace every LoRA tensor of the posterior backbone by Gaussian noise with that tensor's own standard deviation."""

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.posterior_backbone.named_parameters():
            if "lora_" in name:
                noise = torch.randn(parameter.shape, generator=generator) * parameter.float().std().item()
                parameter.copy_(noise.to(parameter.dtype).to(parameter.device))


def main() -> None:
    argv = [arg for arg in sys.argv[1:] if arg != "--held-out-topics"]
    train_module.load_training_data = load_language_training_data
    train_module.watch_test_scores = watch_language_scores
    train_module.main(argv)

    config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
    smoke = config.get("max_steps") is not None
    mu_from = config["mu_from"]
    device = resolve_device(None)
    paths = DataPaths(data_dir=Path(config["data_dir"]))
    corpus = load_corpus(paths=paths)
    translated = load_translations(paths=paths).first(60 if smoke else None)
    num_docs = translated.num_docs
    texts = translated.stacked()
    language = np.repeat(np.arange(len(LANGS)), num_docs)
    groups = np.tile(np.arange(num_docs), len(LANGS))
    doc_newsgroup = corpus.test_labels[np.array(translated.test_index)]
    newsgroup = np.tile(doc_newsgroup, len(LANGS))
    lda_top = np.tile(LDATeacher(paths=paths).theta(translated.texts["en"]).argmax(1), len(LANGS))
    labels = {"language": language, "newsgroup": newsgroup, "lda_top": lda_top, "groups": groups}
    chance = {
        "language": 1 / len(LANGS),
        "newsgroup": round(float(np.bincount(doc_newsgroup).max() / num_docs), 3),
        "lda_top_topic": round(float(np.bincount(lda_top[:num_docs]).max() / num_docs), 3),
    }
    results: dict = {"mu_from": mu_from, "held_out_topics": HELD_OUT_TOPICS, "docs_per_language": num_docs, "majority_baseline": chance}
    saved: dict = dict(labels)

    model = load_trained_run(run_dir=Path(config["output_dir"]), device=device).model
    model.eval()
    torch.manual_seed(0)
    mu, z = moments(model, texts)
    right = mu.argmax(1) == language
    by_group = {corpus.label_names[g]: float(right[newsgroup == g].mean()) for g in np.unique(newsgroup)}
    worst = min(by_group, key=by_group.get)
    results["language_right_of_1000"] = {LANG_NAMES[lang]: int(right[language == k].sum()) for k, lang in enumerate(LANGS)}
    results["language_right_worst_newsgroup"] = {worst: round(by_group[worst], 3)}
    if HELD_OUT_TOPICS:
        seen = np.array([trained_newsgroup(int(g)) for g in newsgroup])
        results["language_right_share"] = {"trained newsgroups": round(float(right[seen].mean()), 4), "held-out newsgroups": round(float(right[~seen].mean()), 4)}
    softmax_mu = torch.softmax(torch.tensor(mu), -1).numpy()
    results["latent_probes"] = {
        "mu": three_probes(mu, language=language, newsgroup=newsgroup, lda_top=lda_top, groups=groups),
        "softmax(mu)": three_probes(softmax_mu, language=language, newsgroup=newsgroup, lda_top=lda_top, groups=groups),
        "sampled z": three_probes(z, language=language, newsgroup=newsgroup, lda_top=lda_top, groups=groups),
    }
    if HELD_OUT_TOPICS:
        results["latent_probes"]["mu, held-out newsgroups only"] = {
            "newsgroup": probe(mu[~seen], newsgroup[~seen], groups[~seen]),
            "lda_top_topic": probe(mu[~seen], lda_top[~seen], groups[~seen]),
        }
    print("latent", json.dumps({k: results[k] for k in ("language_right_of_1000", "language_right_worst_newsgroup", "latent_probes")}), flush=True)
    saved.update(mu=mu, z=z)

    # Writing: from each document's own z, and from each language's mean mu.
    writing: dict = {}
    samples: dict = {}
    for k, lang in enumerate(LANGS):
        own = torch.tensor(z[language == k])
        written: list[str] = []
        for i in range(0, len(own), BATCH):
            written += model.write_from_z(z=own[i : i + BATCH].to(device), max_new_tokens=GEN_LEN)
        detected = [detect_language(text) for text in written]
        center = torch.tensor(mu[language == k].mean(0))
        count = 40 if smoke else 200
        from_center: list[str] = []
        for i in range(0, count, BATCH):
            from_center += model.write_from_z(z=center.repeat(min(BATCH, count - i), 1).to(device), max_new_tokens=GEN_LEN)
        detected_center = [detect_language(text) for text in from_center]
        writing[LANG_NAMES[lang]] = {
            "from each document's z": {"written_language": dict(Counter(detected).most_common()), "in the input language": int(sum(d == lang for d in detected)), "out_of": len(detected)},
            "from the language's mean mu": {"written_language": dict(Counter(detected_center).most_common()), "in that language": int(sum(d == lang for d in detected_center)), "out_of": count},
        }
        samples[lang] = {
            "from each document's z": [{"input": translated.texts[lang][i][:200], "written": written[i]} for i in range(8)],
            "from the language's mean mu": from_center[:8],
        }
    results["writing"] = writing
    print("writing", json.dumps(writing, ensure_ascii=False), flush=True)

    # The 896-number readouts.
    h_off = state(model, texts, adapter=False)
    h_on = state(model, texts, adapter=True)
    readouts = {"plain Qwen h_off": h_off, f"this run, {mu_from} readout": h_on - h_off if mu_from == "delta" else h_on}
    randomize_lora(model)
    readouts["random LoRA, Delta h"] = state(model, texts, adapter=True) - h_off
    del model
    torch.cuda.empty_cache()
    paper = load_trained_run(run_dir=Path(f"../models/final_0.5b/{mu_from}"), device=device).model
    paper.eval()
    paper_on = state(paper, texts, adapter=True)
    readouts[f"paper (topic guide), {mu_from} readout"] = paper_on - h_off if mu_from == "delta" else paper_on
    del paper
    torch.cuda.empty_cache()
    results["readout_probes"] = {}
    for name, features in readouts.items():
        results["readout_probes"][name] = three_probes(features, language=language, newsgroup=newsgroup, lda_top=lda_top, groups=groups)
        print("readout", name, results["readout_probes"][name], flush=True)
        saved["readout: " + name] = features.astype(np.float32)

    stem = Path(config["output_dir"])
    stem.with_name(stem.name + "_scores.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    stem.with_name(stem.name + "_samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    np.savez_compressed(stem.with_name(stem.name + "_features.npz"), **saved)
    print("wrote", stem.with_name(stem.name + "_scores.json"))


if __name__ == "__main__":
    main()
