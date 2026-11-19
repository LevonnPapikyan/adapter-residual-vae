"""What does the decoder do with the z of a Chinese text? Saved weights only, nothing is trained.

For each saved model, encode the 1,000 test documents in English and in Chinese, sample one ``z``
per document, and ask two things.

- Writing: what language does the decoder write in, and does LDA read the English original's
  topic in the text?
- Reconstruction: per-token NLL of a document's tokens given its own ``z``, given the ``z`` of another
  document, and given the ``z`` of the same document in the other language, next to plain Qwen with
  no LoRA and no prefix. All skip each text's first token, which plain Qwen cannot predict.

    python -u foreign_decode.py --out out/run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F

from core.factorization.checkpointing import load_trained_run
from core.factorization.posterior import sample_z
from experiments.data.corpus import load_corpus
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import check_alignment, load_translations

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "vae" / "data"
MODELS_DIR = ROOT / "vae" / "models" / "seeds"
LANGS = ("en", "zh")
BATCH = 32
GEN_LEN = 96
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def script_shares(text: str) -> tuple[float, float]:
    """Share of a text's letters that are Chinese characters, and that are Latin."""

    cjk, latin = len(CJK.findall(text)), len(LATIN.findall(text))
    total = cjk + latin
    return (cjk / total, latin / total) if total else (0.0, 0.0)


@torch.no_grad()
def encode(*, model, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """One sampled ``z`` per text and each text's ``mu``, on the CPU."""

    zs, mus = [], []
    for i in range(0, len(texts), BATCH):
        posterior = model.posterior(texts=texts[i : i + BATCH])
        zs.append(sample_z(mu=posterior.mu, logvar=posterior.logvar).cpu())
        mus.append(posterior.mu.cpu())
    return torch.cat(zs), torch.cat(mus)


@torch.no_grad()
def nll_given_z(*, model, z: torch.Tensor, texts: list[str]) -> float:
    """Per-token NLL of ``texts`` after the prefix from ``z``, from each text's second token on."""

    total, count = 0.0, 0.0
    for i in range(0, len(texts), BATCH):
        ids, mask = model.tokenize(texts=texts[i : i + BATCH], max_tokens=model.config.decoder_max_tokens)
        embeddings = model.generator.get_input_embeddings()(ids)
        prefix = model.z_to_prefix(z[i : i + BATCH].to(model.device))[:, None].to(embeddings.dtype)
        inputs = torch.cat([prefix, embeddings], 1)
        full_mask = torch.cat([mask.new_ones(len(ids), 1), mask], 1)
        logits = model.generator(inputs_embeds=inputs, attention_mask=full_mask).logits[:, 1:-1].float()
        nll = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none") * mask[:, 1:]
        total += nll.sum().item()
        count += mask[:, 1:].sum().item()
    return total / count


@torch.no_grad()
def nll_plain(*, model, texts: list[str]) -> float:
    """The same NLL under Qwen with no LoRA and no prefix."""

    total, count = 0.0, 0.0
    with model.generator.disable_adapter():
        for i in range(0, len(texts), BATCH):
            ids, mask = model.tokenize(texts=texts[i : i + BATCH], max_tokens=model.config.decoder_max_tokens)
            logits = model.generator(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            nll = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none") * mask[:, 1:]
            total += nll.sum().item()
            count += mask[:, 1:].sum().item()
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max_docs", type=int, default=None, help="for smoke tests")
    args = parser.parse_args()
    if (args.out / "results.json").exists():
        raise FileExistsError(f"{args.out}/results.json exists; choose another --out.")
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = DataPaths(data_dir=DATA_DIR)
    corpus = load_corpus(paths=paths)
    translated = load_translations(paths=paths)
    check_alignment(translated=translated, corpus=corpus)
    translated = translated.first(args.max_docs)
    teacher = LDATeacher(paths=paths)
    target = teacher.theta(translated.texts["en"]).argmax(1)
    other_doc = np.roll(np.random.default_rng(0).permutation(translated.num_docs), 1)

    results, samples = [], {}
    for seed in args.seeds:
        for name in ("h", "delta"):
            model = load_trained_run(run_dir=MODELS_DIR / f"seed_{seed}" / name, device=device).model
            torch.manual_seed(seed)
            encoded = {lang: encode(model=model, texts=translated.texts[lang]) for lang in LANGS}
            row: dict = {"seed": seed, "model": name}
            for lang in LANGS:
                z, mu = encoded[lang]
                written = []
                for i in range(0, len(z), BATCH):
                    written += model.write_from_z(z=z[i : i + BATCH].to(device), max_new_tokens=GEN_LEN)
                shares = np.array([script_shares(text) for text in written])
                top = teacher.theta(written).argmax(1)
                texts = translated.texts[lang]
                other = "zh" if lang == "en" else "en"
                row[lang] = {
                    "chinese_share_of_letters": float(shares[:, 0].mean()),
                    "latin_share_of_letters": float(shares[:, 1].mean()),
                    "texts_mostly_chinese": int((shares[:, 0] > 0.5).sum()),
                    "lda_reads_english_topic": int((top == target).sum()),
                    "lda_reads_z_topic": int((top == mu.argmax(1).numpy()).sum()),
                    "nll_own_z": nll_given_z(model=model, z=z, texts=texts),
                    "nll_other_doc_z": nll_given_z(model=model, z=z[other_doc], texts=texts),
                    "nll_same_doc_other_language_z": nll_given_z(model=model, z=encoded[other][0], texts=texts),
                    "nll_plain_qwen": nll_plain(model=model, texts=texts),
                    "out_of": len(z),
                }
                if seed == args.seeds[0]:
                    samples[f"{name}_{lang}"] = [
                        {"english_original": translated.texts["en"][i][:400], "input": texts[i][:400], "written": written[i]}
                        for i in range(min(12, len(written)))
                    ]
            row["same_topic_en_zh"] = int((encoded["en"][1].argmax(1) == encoded["zh"][1].argmax(1)).sum())
            row["z_cosine_en_zh"] = float(F.cosine_similarity(encoded["en"][0], encoded["zh"][0]).mean())
            results.append(row)
            print(json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
            (args.out / "partial.json").write_text(json.dumps(results, indent=1))

    (args.out / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    (args.out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
