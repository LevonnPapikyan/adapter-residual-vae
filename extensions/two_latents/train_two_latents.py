"""Two latents at once: a topic latent aligned to LDA and a language latent aligned to the language label.

    cd main && python -u ../extensions/two_latents/train_two_latents.py --config ../extensions/two_latents/main.json

Model (Qwen2.5-0.5B, frozen, three encoder passes over the same text):
    h_topic     the base with the topic LoRA           Delta h_topic    = h_topic    - h_off  -> mu, log sigma^2 of z_topic    (20)
    h_language  the base with the language LoRA        Delta h_language = h_language - h_off  -> mu, log sigma^2 of z_language (5)
    h_off       the base with no LoRA
    decoder     the base with its own LoRA, reading two prefix positions: one from z_topic, one from z_language

Loss, per batch:
    (rec + beta (KL_topic + KL_language)) / T  +  lam * align_topic  +  lam * align_language
    align_topic    = -sum_k theta_k log softmax(mu_topic)_k,   theta = LDA on the ENGLISH version of the post, for every language
    align_language = -log softmax(mu_language)_[language of the text]

Data: the training posts in English, German, French, Spanish and Chinese, first 150 words each. All five versions of
a post are trained on or held out together. No test document is trained on.

After training, on the 1,000 test documents in five languages:
- topic right (argmax mu_topic against LDA's top topic of the English original) and language right (argmax mu_language);
- leakage: the language read from the topic latent, and the topic read from the language latent;
- the judge: a written text is translated back to English with Marian and read by the English LDA. The judge is first
  checked on the test documents themselves (translated back), so its ceiling is known;
- the grid: for each of 20 topics x 5 languages, texts written from (mean mu_topic of that topic, mean mu_language of
  that language): is the language right, is the topic right;
- the swap: a document's own mu_topic with another language's mean mu_language.

This folder imports main/ and writes nothing under it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

from core.analysis.language_probe import language_score
from core.factorization.adaptation import apply_lora
from core.factorization.model import LOGVAR_INIT, LOGVAR_MAX, LOGVAR_MIN, ReconstructionOutput, TopicVAE
from core.factorization.objective import ObjectiveOutput, alignment_to_teacher, standard_normal_rate
from core.factorization.posterior import last_token_state, sample_z
from core.runs.io import read_json, write_json
from core.train.loop import resolve_device, split_held_out
from experiments.data.corpus import load_corpus
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import LANG_NAMES, LANGS, NUM_WORDS, load_translations, split_sentences
from experiments.factorization.config import parse_train_config
import experiments.factorization.train as train_module

HERE = Path(__file__).resolve().parent
EXP9 = HERE.parent / "multilingual_decode"
sys.path.insert(0, str(EXP9))
from train_multilingual import detect_language  # noqa: E402  (the language detector of multilingual_decode)

NUM_LANGS = len(LANGS)
BATCH = 32
GEN_LEN = 96


@dataclass(frozen=True)
class TwoLatentObjective(ObjectiveOutput):
    """The loss parts of a batch; ``align`` is the topic alignment, ``rate`` the sum of the two KLs."""

    align_language: float = 0.0

    def stats(self) -> dict[str, float]:
        return {**super().stats(), "align_language": self.align_language}


class TwoLatentVAE(TopicVAE):
    """TopicVAE (its encoder, heads and prefix are the topic side) plus a language encoder, heads and a second prefix."""

    def __init__(self, *, config) -> None:
        if config.mu_from != "delta" or not config.train_encoder_lora:
            raise ValueError("Both latents are read from Delta h: the config needs mu_from = delta and a trained encoder LoRA.")
        super().__init__(config=config)
        backbone = AutoModel.from_pretrained(config.base_model, dtype=torch.bfloat16).requires_grad_(False)
        self.language_backbone = apply_lora(backbone, lora=config.lora)
        self.language_mu_head = nn.Linear(self.hidden_size, NUM_LANGS)
        self.language_logvar_head = nn.Linear(self.hidden_size, NUM_LANGS)
        self.language_to_prefix = nn.Linear(NUM_LANGS, self.hidden_size)
        nn.init.zeros_(self.language_logvar_head.weight)
        nn.init.constant_(self.language_logvar_head.bias, LOGVAR_INIT)
        token_std = self.generator.get_input_embeddings().weight.float().std().item()
        nn.init.normal_(self.language_to_prefix.weight, std=token_std / math.sqrt(NUM_LANGS))
        nn.init.zeros_(self.language_to_prefix.bias)

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {"encoder_lora": [], "decoder_lora": [], "linear_maps": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "lora_" not in name:
                groups["linear_maps"].append(parameter)
            elif name.startswith(("posterior_backbone.", "language_backbone.")):
                groups["encoder_lora"].append(parameter)
            else:
                groups["decoder_lora"].append(parameter)
        return groups

    def encode(self, *, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Moments of both posteriors: ``mu_topic``, ``logvar_topic``, ``mu_language``, ``logvar_language``."""

        input_ids, attention_mask = self.tokenize(texts=texts, max_tokens=self.config.encoder_max_tokens)
        with torch.set_grad_enabled(self.training):
            h_topic = self.backbone_state(input_ids=input_ids, attention_mask=attention_mask)
            hidden = self.language_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
            h_language = last_token_state(hidden_states=hidden, attention_mask=attention_mask)
        with torch.no_grad(), self.posterior_backbone.disable_adapter():
            h_off = self.backbone_state(input_ids=input_ids, attention_mask=attention_mask)
        delta_topic, delta_language = h_topic - h_off, h_language - h_off
        return {
            "mu_topic": self.mu_head(delta_topic),
            "logvar_topic": self.logvar_head(delta_topic).clamp(LOGVAR_MIN, LOGVAR_MAX),
            "mu_language": self.language_mu_head(delta_language),
            "logvar_language": self.language_logvar_head(delta_language).clamp(LOGVAR_MIN, LOGVAR_MAX),
        }

    def prefixes(self, *, z_topic: torch.Tensor, z_language: torch.Tensor) -> torch.Tensor:
        """The two prefix positions, ``[batch, 2, hidden]``: topic first, language second."""

        return torch.stack([self.z_to_prefix(z_topic), self.language_to_prefix(z_language)], 1)

    def decode_nll(self, *, z_topic: torch.Tensor, z_language: torch.Tensor, texts: Sequence[str]) -> ReconstructionOutput:
        """``-log p(y | z_topic, z_language)`` summed over each text's tokens."""

        input_ids, attention_mask = self.tokenize(texts=texts, max_tokens=self.config.decoder_max_tokens)
        embeddings = self.generator.get_input_embeddings()(input_ids)
        inputs = torch.cat([self.prefixes(z_topic=z_topic, z_language=z_language).to(embeddings.dtype), embeddings], 1)
        mask = torch.cat([attention_mask.new_ones(len(input_ids), 2), attention_mask], 1)
        logits = self.generator(inputs_embeds=inputs, attention_mask=mask).logits[:, 1:-1].float()  # position 1 predicts token 0
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        return ReconstructionOutput(nll=(token_nll * attention_mask).sum(1), num_tokens=attention_mask.sum(1).float())

    def training_objective(self, *, texts: Sequence[str], theta: torch.Tensor) -> TwoLatentObjective:
        """``theta`` is ``[batch, 20 + 5]``: LDA's mixture of the English version, then the one-hot language."""

        moments = self.encode(texts=texts)
        theta_topic, language = theta[:, : self.config.num_topics], theta[:, self.config.num_topics :]
        z_topic = sample_z(mu=moments["mu_topic"], logvar=moments["logvar_topic"])
        z_language = sample_z(mu=moments["mu_language"], logvar=moments["logvar_language"])
        reconstruction = self.decode_nll(z_topic=z_topic, z_language=z_language, texts=texts)
        rate = standard_normal_rate(mu=moments["mu_topic"], logvar=moments["logvar_topic"]) + standard_normal_rate(
            mu=moments["mu_language"], logvar=moments["logvar_language"]
        )
        align_topic = alignment_to_teacher(theta=theta_topic, mu=moments["mu_topic"]).mean()
        align_language = alignment_to_teacher(theta=language, mu=moments["mu_language"]).mean()
        objective = self.config.objective
        mean_tokens = reconstruction.num_tokens.mean()
        loss = (reconstruction.nll + objective.beta * rate).mean() / mean_tokens + objective.lam * (align_topic + align_language)
        return TwoLatentObjective(
            loss=loss,
            rec=reconstruction.nll.mean().item() / mean_tokens.item(),
            align=align_topic.item(),
            rate=rate.mean().item(),
            align_language=align_language.item(),
        )

    @torch.no_grad()
    def write(self, *, z_topic: torch.Tensor, z_language: torch.Tensor, max_new_tokens: int = GEN_LEN) -> list[str]:
        """Let the decoder write after the two prefix positions (sampling, top-k 50)."""

        prefix = self.prefixes(z_topic=z_topic.to(self.device), z_language=z_language.to(self.device)).to(torch.bfloat16)
        output = self.generator.generate(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def one_hot(lang: str, count: int) -> torch.Tensor:
    out = torch.zeros(count, NUM_LANGS)
    out[:, LANGS.index(lang)] = 1.0
    return out


def load_two_latent_training_data(*, config, rng: np.random.Generator):
    """Every kept training post in five languages; the target is [LDA of the English version, one-hot language]."""

    recipe = config.recipe
    paths = DataPaths(data_dir=config.data_dir)
    corpus = load_corpus(paths=paths)
    versions = {
        "en": [" ".join(text.split()[:NUM_WORDS]) for text in corpus.train_texts],
        "de": [" ".join(text.split()[:NUM_WORDS]) for text in read_json(paths.train_de)],
        **{lang: read_json(EXP9 / "data" / f"train_{lang}.json") for lang in ("fr", "es", "zh")},
    }
    index = np.arange(len(corpus.train_texts) if config.max_train_texts is None else min(len(corpus.train_texts), config.max_train_texts))
    teacher = LDATeacher(paths=paths, name=recipe.teacher)
    words_per_post = np.asarray(teacher.counts([versions["en"][i] for i in index]).sum(1)).ravel()
    index = index[words_per_post >= recipe.min_words]
    index = np.array([i for i in index if all(versions[lang][i].strip() for lang in LANGS)])
    theta = torch.tensor(teacher.theta([versions["en"][i] for i in index]), dtype=torch.float32)  # of the English version
    split = split_held_out(rng=rng, num_texts=len(index), val_frac=recipe.val_frac)

    def stack(rows: np.ndarray) -> tuple[list[str], torch.Tensor]:
        texts = [versions[lang][i] for lang in LANGS for i in index[rows]]
        target = torch.cat([torch.cat([theta[rows], one_hot(lang, len(rows))], 1) for lang in LANGS])
        return texts, target

    texts, target = stack(split.kept)
    val_texts, val_target = stack(split.held_out)
    print(f"{len(split.kept)} posts x {NUM_LANGS} languages trained on, {len(split.held_out)} posts x {NUM_LANGS} held out", flush=True)
    test_docs = load_translations(paths=paths).first(config.max_test_docs)
    return train_module.TrainingData(
        texts=texts, theta=target, val_texts=val_texts, val_theta=val_target,
        test_docs=test_docs, theta_test=teacher.theta(test_docs.texts["en"]),
    )


@torch.no_grad()
def held_out_two_latent_loss(*, model: TwoLatentVAE, texts: Sequence[str], theta: torch.Tensor, batch_size: int = BATCH) -> float:
    """Reconstruction per token at z = mu plus lam times both alignments, on the held-out posts."""

    model.eval()
    total = 0.0
    num_topics = model.config.num_topics
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        target = theta[i : i + batch_size].to(model.device)
        moments = model.encode(texts=batch)
        align = alignment_to_teacher(theta=target[:, :num_topics], mu=moments["mu_topic"]) + alignment_to_teacher(
            theta=target[:, num_topics:], mu=moments["mu_language"]
        )
        reconstruction = model.decode_nll(z_topic=moments["mu_topic"], z_language=moments["mu_language"], texts=batch)
        total += (model.config.objective.lam * align + reconstruction.nll / reconstruction.num_tokens).sum().item()
    model.train()
    return total / len(texts)


@torch.no_grad()
def watch_two_latent_scores(*, model: TwoLatentVAE, docs: Mapping[str, Sequence[str]], theta_target: np.ndarray, batch_size: int = BATCH) -> dict:
    """After every epoch, on the test documents: topic right per language, language right in all, English reconstruction."""

    model.eval()
    top = torch.tensor(theta_target.argmax(1))
    out: dict = {}
    rec, language_right = 0.0, 0
    for k, lang in enumerate(LANGS):
        right = 0
        for i in range(0, len(docs[lang]), batch_size):
            batch = docs[lang][i : i + batch_size]
            moments = model.encode(texts=batch)
            right += int((moments["mu_topic"].argmax(1).cpu() == top[i : i + batch_size]).sum())
            language_right += int((moments["mu_language"].argmax(1) == k).sum())
            if lang == "en":
                reconstruction = model.decode_nll(z_topic=moments["mu_topic"], z_language=moments["mu_language"], texts=batch)
                rec += (reconstruction.nll / reconstruction.num_tokens).sum().item()
        out[lang] = right
    model.train()
    return {"rec": rec / len(docs["en"]), **out, "language_right_of_all": language_right}


def save_two_latent_model(*, model: TwoLatentVAE, recipe, history, run_dir: Path) -> None:
    """``model.pt`` holds the trained tensors under their module names; the settings are the run's config file."""

    torch.save({"weights": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}}, Path(run_dir) / "model.pt")
    write_json(Path(run_dir) / "history.json", history, indent=None)


def load_two_latent_model(*, argv: list[str], device) -> TwoLatentVAE:
    config = parse_train_config(argv)
    model = TwoLatentVAE(config=config.recipe.model)
    weights = torch.load(Path(config.output_dir) / "model.pt", map_location="cpu")["weights"]
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unknown saved tensors: {result.unexpected_keys[:5]}")
    print(f"{config.output_dir}: {len(weights)} trained tensors loaded", flush=True)
    return model.to(device).eval()


# ---------------------------------------------------------------- scoring


def split_for_translation(text: str, lang: str) -> list[str]:
    if lang != "zh":
        return split_sentences(text)
    pieces = [piece for piece in re.split(r"(?<=[。！？!?；;])", text) if piece.strip()]
    return [piece[i : i + 120] for piece in pieces for i in range(0, len(piece), 120)]


@torch.no_grad()
def to_english(texts: Sequence[str], lang: str, *, batch_size: int = 48) -> list[str]:
    """Marian ``opus-mt-{lang}-en``, sentence by sentence, greedy: the routine that made the test set, in the other direction."""

    if lang == "en":
        return list(texts)
    from transformers import MarianMTModel, MarianTokenizer

    name = f"Helsinki-NLP/opus-mt-{lang}-en"
    tokenizer = MarianTokenizer.from_pretrained(name)
    marian = MarianMTModel.from_pretrained(name).cuda().half().eval()
    sentences, owner = [], []
    for i, text in enumerate(texts):
        for sentence in split_for_translation(text, lang):
            sentences.append(sentence)
            owner.append(i)
    order = np.argsort([-len(sentence) for sentence in sentences])
    written = [""] * len(sentences)
    for start in range(0, len(sentences), batch_size):
        indices = order[start : start + batch_size]
        batch = tokenizer([sentences[i] for i in indices], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        output = marian.generate(**batch, max_new_tokens=256, num_beams=1)
        for i, sentence in zip(indices, tokenizer.batch_decode(output, skip_special_tokens=True)):
            written[i] = sentence
    joined: list[list[str]] = [[] for _ in texts]
    for i, sentence in zip(owner, written):
        joined[i].append(sentence)
    del marian
    torch.cuda.empty_cache()
    return [" ".join(parts) for parts in joined]


@torch.no_grad()
def encode_all(model: TwoLatentVAE, texts: Sequence[str]) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {}
    for i in range(0, len(texts), BATCH):
        for key, value in model.encode(texts=texts[i : i + BATCH]).items():
            parts.setdefault(key, []).append(value.float().cpu())
    return {key: torch.cat(value).numpy() for key, value in parts.items()}


def write_all(model: TwoLatentVAE, z_topic: np.ndarray, z_language: np.ndarray) -> list[str]:
    out: list[str] = []
    for i in range(0, len(z_topic), 64):
        out += model.write(z_topic=torch.tensor(z_topic[i : i + 64]), z_language=torch.tensor(z_language[i : i + 64]))
    return out


def probe(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        return round(float(cross_val_score(model, features, labels, groups=groups, cv=GroupKFold(5), n_jobs=4).mean()), 3)


def judge(texts: Sequence[str], lang: str, teacher: LDATeacher) -> tuple[list[str], np.ndarray, list[str]]:
    """The detected language of each text, LDA's top topic of its English translation, and that translation."""

    english = to_english(texts, lang)
    return [detect_language(text) for text in texts], teacher.theta(english).argmax(1), english


def main() -> None:
    argv = sys.argv[1:]
    train_module.TopicVAE = TwoLatentVAE
    train_module.load_training_data = load_two_latent_training_data
    train_module.held_out_loss = held_out_two_latent_loss
    train_module.watch_test_scores = watch_two_latent_scores
    train_module.save_trained_model = save_two_latent_model
    train_module.main(argv)

    config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
    smoke = config.get("max_steps") is not None
    device = resolve_device(None)
    paths = DataPaths(data_dir=Path(config["data_dir"]))
    corpus = load_corpus(paths=paths)
    translated = load_translations(paths=paths).first(60 if smoke else None)
    num_docs = translated.num_docs
    teacher = LDATeacher(paths=paths)
    target = teacher.theta(translated.texts["en"])
    top = target.argmax(1)
    texts = translated.stacked()
    language = np.repeat(np.arange(NUM_LANGS), num_docs)
    groups = np.tile(np.arange(num_docs), NUM_LANGS)
    top_all = np.tile(top, NUM_LANGS)
    newsgroup = np.tile(corpus.test_labels[np.array(translated.test_index)], NUM_LANGS)
    results: dict = {"docs_per_language": num_docs, "majority_topic": int(np.bincount(top).max()), "topic_words": teacher.top_words(6)}

    model = load_two_latent_model(argv=argv, device=device)
    torch.manual_seed(0)
    moments = encode_all(model, texts)
    mu_topic, mu_language = moments["mu_topic"], moments["mu_language"]
    results["topic_right"] = {LANG_NAMES[lang]: int((mu_topic[language == k].argmax(1) == top).sum()) for k, lang in enumerate(LANGS)}
    results["language_right"] = {LANG_NAMES[lang]: int((mu_language[language == k].argmax(1) == k).sum()) for k, lang in enumerate(LANGS)}
    softmax_topic = torch.softmax(torch.tensor(mu_topic), -1).numpy()
    results["leakage"] = {
        "language score of the topic latent (paper's score; paper Δh 0.379, h 0.732, chance 0.2)": round(
            float(language_score(thetas={lang: softmax_topic[language == k] for k, lang in enumerate(LANGS)})), 3
        ),
        "language probe on mu_topic": probe(mu_topic, language, groups),
        "LDA top topic probe on mu_language": probe(mu_language, top_all, groups),
        "newsgroup probe on mu_language": probe(mu_language, newsgroup, groups),
        "LDA top topic probe on mu_topic": probe(mu_topic, top_all, groups),
        "newsgroup probe on mu_topic": probe(mu_topic, newsgroup, groups),
        "language probe on mu_language": probe(mu_language, language, groups),
        "chance": {"language": 0.2, "LDA top topic": round(float(np.bincount(top).max() / num_docs), 3), "newsgroup": round(float(np.bincount(newsgroup[:num_docs]).max() / num_docs), 3)},
    }
    print("latents", json.dumps({k: results[k] for k in ("topic_right", "language_right", "leakage")}, ensure_ascii=False), flush=True)

    # The judge's ceiling: the real test documents, translated back to English and read by LDA.
    results["judge_check"] = {
        LANG_NAMES[lang]: int((teacher.theta(to_english(translated.texts[lang], lang)).argmax(1) == top).sum()) for lang in LANGS
    }
    print("judge check (real test documents, translated back, LDA top topic right):", results["judge_check"], flush=True)

    topic_center = {k: mu_topic[top_all == k].mean(0) for k in range(target.shape[1]) if (top_all == k).any()}
    language_center = np.stack([mu_language[language == k].mean(0) for k in range(NUM_LANGS)])
    samples: dict = {"grid": {}, "swap": {}}

    # The grid: every topic's mean with every language's mean.
    per_cell = 4 if smoke else 50
    grid: dict = {}
    for k, lang in enumerate(LANGS):
        topics = np.repeat(np.array(sorted(topic_center)), per_cell)
        written = write_all(model, np.stack([topic_center[t] for t in topics]), np.repeat(language_center[k][None], len(topics), 0))
        detected, judged, english = judge(written, lang, teacher)
        language_ok = np.array([d == lang for d in detected])
        topic_ok = judged == topics
        grid[LANG_NAMES[lang]] = {
            "language_right": round(float(language_ok.mean()), 3),
            "topic_right": round(float(topic_ok.mean()), 3),
            "both_right": round(float((language_ok & topic_ok).mean()), 3),
            "topic_right_by_topic": {int(t): round(float(topic_ok[topics == t].mean()), 3) for t in sorted(topic_center)},
            "language_right_by_topic": {int(t): round(float(language_ok[topics == t].mean()), 3) for t in sorted(topic_center)},
            "texts": len(written),
        }
        samples["grid"][lang] = {
            int(t): [{"written": written[i], "in_english": english[i], "judged_topic": int(judged[i]), "detected": detected[i]} for i in np.flatnonzero(topics == t)[:3]]
            for t in sorted(topic_center)
        }
        print("grid", LANG_NAMES[lang], {key: grid[LANG_NAMES[lang]][key] for key in ("language_right", "topic_right", "both_right")}, flush=True)
    results["grid"] = grid

    # The swap: a document's own mu_topic (from its English version) with each language's mean mu_language.
    swap: dict = {}
    own_topic = mu_topic[language == 0]
    for k, lang in enumerate(LANGS):
        written = write_all(model, own_topic, np.repeat(language_center[k][None], num_docs, 0))
        detected, judged, english = judge(written, lang, teacher)
        language_ok = np.array([d == lang for d in detected])
        swap[f"English document -> written in {LANG_NAMES[lang]}"] = {
            "language_right": int(language_ok.sum()), "topic_right": int((judged == top).sum()), "both_right": int((language_ok & (judged == top)).sum()), "out_of": num_docs,
        }
        samples["swap"][f"en->{lang}"] = [{"document": translated.texts["en"][i][:300], "written": written[i], "in_english": english[i]} for i in range(8)]
    for k, lang in enumerate(LANGS[1:], start=1):
        written = write_all(model, mu_topic[language == k], np.repeat(language_center[0][None], num_docs, 0))
        detected, judged, _ = judge(written, "en", teacher)
        language_ok = np.array([d == "en" for d in detected])
        swap[f"{LANG_NAMES[lang]} document -> written in English"] = {
            "language_right": int(language_ok.sum()), "topic_right": int((judged == top).sum()), "both_right": int((language_ok & (judged == top)).sum()), "out_of": num_docs,
        }
        samples["swap"][f"{lang}->en"] = [{"document": translated.texts[lang][i][:300], "written": written[i]} for i in range(8)]
    results["swap"] = swap
    print("swap", json.dumps(swap, ensure_ascii=False), flush=True)

    stem = Path(config["output_dir"])
    stem.with_name(stem.name + "_scores.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    stem.with_name(stem.name + "_samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    np.savez_compressed(stem.with_name(stem.name + "_features.npz"), mu_topic=mu_topic, mu_language=mu_language, language=language, lda_top=top_all, newsgroup=newsgroup, groups=groups)
    print("wrote", stem.with_name(stem.name + "_scores.json"))


if __name__ == "__main__":
    main()
