"""The translated test set: 1,000 test posts, first 150 words, in five languages.

Document ``i`` is the same post in every language, so each translation has a known
answer: LDA's ``theta`` of its English original.

What belongs here
-----------------
- Loading and checking ``translations.json``, and the routine that made it.

What does NOT belong here
-------------------------
- Scoring. The file is loaded, never rebuilt in place: the sample is reproducible,
  the Marian output is not byte for byte, and the models were scored on exactly
  these texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from core.runs.io import read_json, write_json
from experiments.data.corpus import Corpus
from experiments.data.paths import DataPaths

LANGS: tuple[str, ...] = ("en", "de", "fr", "es", "zh")
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese"}
NUM_DOCS, NUM_WORDS, SAMPLE_SEED = 1000, 150, 0
MARIAN_MODELS = {lang: f"Helsinki-NLP/opus-mt-en-{lang}" for lang in LANGS[1:]}


@dataclass(frozen=True)
class TranslatedTestSet:
    """``texts[lang][i]`` is document ``i`` in that language; ``test_index[i]`` is its test post."""

    texts: dict[str, list[str]]
    test_index: list[int]

    @property
    def num_docs(self) -> int:
        """Documents per language."""

        return len(self.test_index)

    def first(self, num_docs: int | None) -> TranslatedTestSet:
        """The first documents, for smoke tests; ``None`` keeps all of them."""

        if num_docs is None:
            return self
        return TranslatedTestSet(
            texts={lang: texts[:num_docs] for lang, texts in self.texts.items()},
            test_index=self.test_index[:num_docs],
        )

    def stacked(self) -> list[str]:
        """All texts in one list, the English block first."""

        return [text for lang in LANGS for text in self.texts[lang]]


def load_translations(*, paths: DataPaths) -> TranslatedTestSet:
    """Read ``translations.json``."""

    data = read_json(paths.translations)
    return TranslatedTestSet(texts={lang: data[lang] for lang in LANGS}, test_index=data["test_index"])


def check_alignment(*, translated: TranslatedTestSet, corpus: Corpus) -> None:
    """Everything rests on this: document ``i`` is the first 150 words of test post ``test_index[i]``, in every language."""

    if any(len(translated.texts[lang]) != translated.num_docs for lang in LANGS) or len(set(translated.test_index)) != translated.num_docs:
        raise ValueError("Every language must hold the same documents.")
    for text, j in zip(translated.texts["en"], translated.test_index):
        if text != " ".join(corpus.test_texts[j].split()[:NUM_WORDS]):
            raise ValueError(f"The English text of test post {j} is not its first {NUM_WORDS} words.")


def stratified_sample(*, test_labels: np.ndarray) -> np.ndarray:
    """The 1,000 test posts, stratified by newsgroup, in sorted order."""

    sample, _ = train_test_split(
        np.arange(len(test_labels)), train_size=NUM_DOCS, stratify=test_labels, random_state=SAMPLE_SEED
    )
    return np.sort(sample)


def check_sample(*, translated: TranslatedTestSet, corpus: Corpus) -> None:
    """The saved ``test_index`` is the stratified sample."""

    if not np.array_equal(stratified_sample(test_labels=corpus.test_labels), np.array(translated.test_index)):
        raise ValueError("The saved test_index is not the stratified sample.")


def split_sentences(text: str, *, max_words: int = 60) -> list[str]:
    """Sentences, with long ones cut into pieces of at most ``max_words``."""

    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        out += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
    return [sentence for sentence in out if sentence]


@torch.no_grad()
def translate(*, texts: Sequence[str], lang: str, batch_size: int = 48) -> list[str]:
    """Marian, sentence by sentence, greedy. Needs a GPU."""

    from transformers import MarianMTModel, MarianTokenizer

    if lang not in MARIAN_MODELS:
        raise ValueError(f"lang must be one of {sorted(MARIAN_MODELS)}, got {lang!r}.")
    tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODELS[lang])
    marian = MarianMTModel.from_pretrained(MARIAN_MODELS[lang]).cuda().half().eval()
    sentences, owner = [], []
    for i, text in enumerate(texts):
        for sentence in split_sentences(text):
            sentences.append(sentence)
            owner.append(i)
    order = np.argsort([-len(sentence) for sentence in sentences])  # similar lengths together: less padding
    written = [""] * len(sentences)
    for start in range(0, len(sentences), batch_size):
        indices = order[start : start + batch_size]
        batch = tokenizer(
            [sentences[i] for i in indices], return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to("cuda")
        output = marian.generate(**batch, max_new_tokens=256, num_beams=1)
        for i, sentence in zip(indices, tokenizer.batch_decode(output, skip_special_tokens=True)):
            written[i] = sentence
    joined: list[list[str]] = [[] for _ in texts]
    for i, sentence in zip(owner, written):
        joined[i].append(sentence)
    return [" ".join(parts) for parts in joined]


def regenerate_translations(*, corpus: Corpus, output_path: str | Path) -> None:
    """Rebuild a translations file the way the saved one was made, for a lost file, not for checking one."""

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"{output_path} exists; the models were scored on it.")
    if not torch.cuda.is_available():
        raise ValueError("Translating 1,000 documents into four languages needs a GPU.")
    sample = stratified_sample(test_labels=corpus.test_labels)
    out = {"test_index": sample.tolist(), "en": [" ".join(corpus.test_texts[j].split()[:NUM_WORDS]) for j in sample]}
    for lang in MARIAN_MODELS:
        started = time.time()
        out[lang] = translate(texts=out["en"], lang=lang)
        print(f"{LANG_NAMES[lang]}: {len(out[lang])} documents in {time.time() - started:.0f} s")
    write_json(output_path, out, indent=None)
    print("saved", output_path)
