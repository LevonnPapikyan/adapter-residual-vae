"""Train on the posts in all five languages, with the LDA guide on the English posts only. Then: what does the decoder write?

    cd main && python -u ../extensions/multilingual_decode/train_multilingual.py --config ../extensions/multilingual_decode/delta.json

The paper's models are trained on English posts only, and their decoder writes English whatever language z came
from (foreign_decode.py). Here everything is the paper's recipe (seed 0, same held-out posts) except the training set:

- every kept English training post, whole, with LDA's theta: reconstruction + rate + alignment, as in the paper;
- the same posts in German, French, Spanish and Chinese (first 150 words, Marian): reconstruction + rate only.
  Their theta row is zero, and the alignment is averaged over the English texts of the batch alone.

The epoch is picked on the held-out English posts, as in the paper. No test document is trained on.

After training, the saved weights and the paper's seed 0 model with the same readout are scored with the same code,
on the 1,000 test documents in each language:

- topic right from argmax mu, and the language score (how well a probe tells the language from softmax(mu));
- writing: one z is sampled per document, the decoder writes 96 tokens, and the language of each written text is detected
  (Chinese characters, or stopwords of English / German / French / Spanish);
- topic of the written text: LDA's top topic of it (only meaningful for English output), and the model's own
  argmax mu of it ("round trip", works in any language but uses the model to judge itself);
- per-token NLL of each document given its own z, next to plain Qwen.

This folder imports main/ and writes nothing under it.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np
import torch

from core.analysis.language_probe import language_score
from core.analysis.topic_score import majority_count, topic_score
from core.factorization.checkpointing import load_trained_run
from core.factorization.model import TopicVAE
from core.factorization.objective import ObjectiveOutput, alignment_to_teacher, standard_normal_rate
from core.factorization.posterior import sample_z
from core.runs.io import read_json
from core.train.loop import resolve_device, split_held_out
from experiments.data.corpus import load_train_texts
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import LANG_NAMES, LANGS, NUM_WORDS, load_translations
import experiments.factorization.train as train_module

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from foreign_decode import nll_given_z, nll_plain  # noqa: E402  (read-only the two NLL routines)

FOREIGN = ("de", "fr", "es", "zh")
BATCH = 32
GEN_LEN = 96
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
STOPWORDS = {
    "en": {"the", "and", "of", "to", "is", "that", "it", "for", "you", "was", "with", "this", "are", "have", "not", "be"},
    "de": {"der", "das", "und", "ist", "nicht", "ich", "zu", "den", "ein", "eine", "mit", "auf", "für", "sich", "von", "dem", "auch"},
    "fr": {"le", "les", "et", "est", "une", "pas", "pour", "dans", "je", "vous", "qui", "ce", "sur", "au", "avec", "sont", "il"},
    "es": {"el", "los", "las", "y", "una", "por", "para", "con", "lo", "del", "como", "más", "pero", "su", "al"},
}


def detect_language(text: str) -> str:
    """``zh`` when most letters are Chinese characters, else the language whose stopwords appear most; ``other`` if none do."""

    cjk, latin = len(CJK.findall(text)), len(LATIN.findall(text))
    if cjk + latin == 0:
        return "other"
    if cjk / (cjk + latin) > 0.5:
        return "zh"
    words = Counter(re.findall(r"[a-zà-ÿ]+", text.lower()))
    hits = {lang: sum(words[word] for word in stop) for lang, stop in STOPWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "other"


class EnglishAlignedTopicVAE(TopicVAE):
    """TopicVAE whose alignment term counts only the texts that have a teacher mixture (a non-zero theta row)."""

    def training_objective(self, *, texts: Sequence[str], theta: torch.Tensor) -> ObjectiveOutput:
        posterior = self.posterior(texts=texts)
        if theta.shape != posterior.mu.shape:
            raise ValueError(f"theta {tuple(theta.shape)} must have the shape of mu {tuple(posterior.mu.shape)}.")
        z = sample_z(mu=posterior.mu, logvar=posterior.logvar)
        reconstruction = self.reconstruction_nll(z=z, texts=texts)
        rate = standard_normal_rate(mu=posterior.mu, logvar=posterior.logvar)
        has_teacher = theta.sum(-1) > 0.5
        align = alignment_to_teacher(theta=theta, mu=posterior.mu)  # zero on the rows whose theta is zero
        align_mean = align[has_teacher].mean() if has_teacher.any() else align.sum() * 0.0
        objective = self.config.objective
        mean_tokens = reconstruction.num_tokens.mean()
        loss = (reconstruction.nll + objective.beta * rate).mean() / mean_tokens + objective.lam * align_mean
        return ObjectiveOutput(
            loss=loss,
            rec=reconstruction.nll.mean().item() / mean_tokens.item(),
            align=align_mean.item(),
            rate=rate.mean().item(),
        )


def load_multilingual_training_data(*, config, rng: np.random.Generator):
    """The paper's English training and held-out posts (same draws from ``rng``), plus the kept posts in four more languages."""

    recipe = config.recipe
    paths = DataPaths(data_dir=config.data_dir)
    english = load_train_texts(paths=paths)
    foreign = {"de": [" ".join(text.split()[:NUM_WORDS]) for text in read_json(paths.train_de)]}
    for lang in FOREIGN[1:]:
        foreign[lang] = read_json(HERE / "data" / f"train_{lang}.json")
    for lang, texts in foreign.items():
        if len(texts) != len(english):
            raise ValueError(f"{lang}: {len(texts)} translations for {len(english)} training posts.")
    index = np.arange(len(english) if config.max_train_texts is None else min(len(english), config.max_train_texts))
    teacher = LDATeacher(paths=paths, name=recipe.teacher)
    words_per_post = np.asarray(teacher.counts([english[i] for i in index]).sum(1)).ravel()
    index = index[words_per_post >= recipe.min_words]
    theta = torch.tensor(teacher.theta([english[i] for i in index]), dtype=torch.float32)
    split = split_held_out(rng=rng, num_texts=len(index), val_frac=recipe.val_frac)
    texts = [english[i] for i in index[split.kept]]
    thetas = [theta[split.kept]]
    print(f"English, with the LDA guide: {len(texts)} posts", flush=True)
    for lang in FOREIGN:
        added = [foreign[lang][i] for i in index[split.kept] if foreign[lang][i].strip()]
        texts += added
        thetas.append(torch.zeros(len(added), theta.shape[1]))
        print(f"{LANG_NAMES[lang]}, no guide: {len(added)} posts", flush=True)
    test_docs = load_translations(paths=paths).first(config.max_test_docs)
    return train_module.TrainingData(
        texts=texts,
        theta=torch.cat(thetas),
        val_texts=[english[i] for i in index[split.held_out]],
        val_theta=theta[split.held_out],
        test_docs=test_docs,
        theta_test=teacher.theta(test_docs.texts["en"]),
    )


@torch.no_grad()
def encode(model: TopicVAE, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """``mu`` and one sampled ``z`` per text, float32, on the CPU."""

    mus, zs = [], []
    for i in range(0, len(texts), BATCH):
        posterior = model.posterior(texts=texts[i : i + BATCH])
        mus.append(posterior.mu.float().cpu())
        zs.append(sample_z(mu=posterior.mu, logvar=posterior.logvar).float().cpu())
    return torch.cat(mus), torch.cat(zs)


@torch.no_grad()
def score(model: TopicVAE, *, texts: dict[str, list[str]], target: np.ndarray, teacher: LDATeacher) -> tuple[dict, dict]:
    """Every number of the module docstring for one model, and a few written samples per language."""

    model.eval()
    torch.manual_seed(0)
    top = target.argmax(1)
    encoded = {lang: encode(model, texts[lang]) for lang in LANGS}
    thetas = {lang: torch.softmax(encoded[lang][0], -1).numpy() for lang in LANGS}
    out: dict = {
        "topic_argmax_mu": {LANG_NAMES[lang]: int(topic_score(theta_pred=thetas[lang], target=target)) for lang in LANGS},
        "language_score": round(float(language_score(thetas=thetas)), 3),
        "writing": {},
    }
    samples: dict = {}
    for lang in LANGS:
        z = encoded[lang][1]
        written: list[str] = []
        for i in range(0, len(z), BATCH):
            written += model.write_from_z(z=z[i : i + BATCH].to(model.device), max_new_tokens=GEN_LEN)
        detected = [detect_language(text) for text in written]
        is_english = np.array([d == "en" for d in detected])
        lda_top = teacher.theta(written).argmax(1)
        round_trip = encode(model, written)[0].argmax(1).numpy()
        out["writing"][LANG_NAMES[lang]] = {
            "written_language": dict(Counter(detected).most_common()),
            "writes_the_input_language": int(sum(d == lang for d in detected)),
            "english_texts": int(is_english.sum()),
            "lda_topic_right_among_english_texts": int((lda_top[is_english] == top[is_english]).sum()),
            "round_trip_topic_right": int((round_trip == top).sum()),
            "nll_own_z": round(nll_given_z(model=model, z=z, texts=texts[lang]), 4),
            "nll_plain_qwen": round(nll_plain(model=model, texts=texts[lang]), 4),
            "out_of": len(z),
        }
        samples[lang] = [
            {"english_original": texts["en"][i][:300], "input": texts[lang][i][:300], "written": written[i], "detected": detected[i]}
            for i in range(min(12, len(written)))
        ]
    return out, samples


def main() -> None:
    train_module.TopicVAE = EnglishAlignedTopicVAE  # the unchanged loop builds this class
    train_module.load_training_data = load_multilingual_training_data  # and trains on these texts
    train_module.main(sys.argv[1:])

    config = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
    smoke = config.get("max_steps") is not None
    device = resolve_device(None)
    paths = DataPaths(data_dir=Path(config["data_dir"]))
    translated = load_translations(paths=paths).first(40 if smoke else None)
    teacher = LDATeacher(paths=paths)
    target = teacher.theta(translated.texts["en"])
    check = {LANG_NAMES[lang]: int(sum(detect_language(text) == lang for text in translated.texts[lang])) for lang in LANGS}
    print(f"language detector on the test documents themselves, right of {translated.num_docs}: {check}", flush=True)
    results = {"mu_from": config["mu_from"], "majority_baseline": int(majority_count(target=target)), "detector_check": check}
    all_samples = {}
    paper = f"paper {'Δh' if config['mu_from'] == 'delta' else 'h'}, seed 0 (English only)"
    for name, run_dir in (("this run (all languages)", config["output_dir"]), (paper, f"../models/final_0.5b/{config['mu_from']}")):
        model = load_trained_run(run_dir=Path(run_dir), device=device).model
        results[name], all_samples[name] = score(model, texts=translated.texts, target=target, teacher=teacher)
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    stem = Path(config["output_dir"])
    stem.with_name(stem.name + "_scores.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    stem.with_name(stem.name + "_samples.json").write_text(json.dumps(all_samples, indent=1, ensure_ascii=False))
    print("wrote", stem.with_name(stem.name + "_scores.json"))


if __name__ == "__main__":
    main()
