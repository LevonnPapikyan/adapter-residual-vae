"""What a representation is about: the document or the language?

What belongs here
-----------------
- ``TextIndex``: which language and which document each row of a stacked array is.
- ``finds_original`` and ``variance_split``.

What does NOT belong here
-------------------------
- Extracting the representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VarianceSplit:
    """Share of the total variance explained by the language means and by the document means."""

    language: float
    document: float


@dataclass(frozen=True)
class TextIndex:
    """Rows of a stacked ``[num_langs * num_docs, ...]`` array, the first language's block first."""

    num_langs: int
    num_docs: int

    def __post_init__(self) -> None:
        if self.num_langs <= 0 or self.num_docs <= 0:
            raise ValueError(f"num_langs and num_docs must be positive, got {self.num_langs}, {self.num_docs}.")

    @property
    def text_lang(self) -> np.ndarray:
        """The language number of each row."""

        return np.repeat(np.arange(self.num_langs), self.num_docs)

    @property
    def text_doc(self) -> np.ndarray:
        """The document number of each row."""

        return np.tile(np.arange(self.num_docs), self.num_langs)


def finds_original(*, features: np.ndarray, lang_index: int, index: TextIndex) -> int:
    """How many texts in language ``lang_index`` have their own original (language 0) as nearest by cosine."""

    if len(features) != index.num_langs * index.num_docs:
        raise ValueError(f"features has {len(features)} rows, expected {index.num_langs} x {index.num_docs}.")
    centered = features - features.mean(0)
    unit = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    text_lang = index.text_lang
    similarity = unit[text_lang == lang_index] @ unit[text_lang == 0].T
    return int((similarity.argmax(1) == np.arange(index.num_docs)).sum())


def variance_split(*, features: np.ndarray, index: TextIndex) -> VarianceSplit:
    """Split the variance of ``features`` into the language means and the document means."""

    if len(features) != index.num_langs * index.num_docs:
        raise ValueError(f"features has {len(features)} rows, expected {index.num_langs} x {index.num_docs}.")
    centered = features - features.mean(0)
    total = (centered**2).sum()
    text_lang, text_doc = index.text_lang, index.text_doc
    language = sum((centered[text_lang == j].mean(0) ** 2).sum() * index.num_docs for j in range(index.num_langs))
    document = sum((centered[text_doc == d].mean(0) ** 2).sum() * index.num_langs for d in range(index.num_docs))
    return VarianceSplit(language=float(language / total), document=float(document / total))
