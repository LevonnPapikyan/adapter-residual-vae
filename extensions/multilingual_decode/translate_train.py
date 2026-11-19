"""Translate the first 150 words of every English training post into French, Spanish and Chinese.

    python -u translate_train.py

The same Marian models and the same routine that made the translated test set (experiments/data/translations.py).
German is not translated here: data/train_de.json already holds the training posts in German.
Output: data/train_{fr,es,zh}.json in this folder, one text per training post, in the order of corpus.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from experiments.data.corpus import load_train_texts
from experiments.data.paths import DataPaths
from experiments.data.translations import NUM_WORDS, translate

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "data"
OUT_DIR = HERE / "data"
LANGS = ("zh", "fr", "es")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    english = [" ".join(text.split()[:NUM_WORDS]) for text in load_train_texts(paths=DataPaths(data_dir=DATA_DIR))]
    print(f"{len(english)} training posts, first {NUM_WORDS} words each", flush=True)
    for lang in LANGS:
        path = OUT_DIR / f"train_{lang}.json"
        if path.exists():
            print(f"{path} exists, skipped", flush=True)
            continue
        started = time.time()
        written = translate(texts=english, lang=lang)
        assert len(written) == len(english)
        path.write_text(json.dumps(written, ensure_ascii=False))
        empty = sum(not text.strip() for text in written)
        print(f"{lang}: {len(written)} posts in {(time.time() - started) / 60:.1f} min, {empty} empty; saved {path}", flush=True)
        print("   ", english[0][:120], "\n   ", written[0][:120], flush=True)


if __name__ == "__main__":
    main()
