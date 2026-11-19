"""Baseline: no fine-tuning, a linear map from plain Qwen's last-token state to LDA's ``theta``.

    python -m experiments.baselines.frozen_linear_map --data-dir ../data --output-dir outputs/baselines

It measures what the language model already knows.

What belongs here
-----------------
- Fitting the map on plain Qwen's features and scoring it alone.

What does NOT belong here
-------------------------
- The comparison with the trained models (``experiments/factorization/evaluate.py``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from core.analysis.encoding import readout_features
from core.analysis.language_probe import language_score
from core.analysis.linear_map import LinearTopicMap, fit_linear_map
from core.analysis.topic_score import topic_score
from core.factorization.config import TopicVAEConfig
from core.factorization.model import TopicVAE
from core.runs.io import parse_args_with_config, refuse_protected_path, write_json
from core.train.loop import resolve_device
from experiments.data.corpus import load_corpus
from experiments.data.paths import PROTECTED_DIRS, DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import LANGS, load_translations


@dataclass(frozen=True)
class FrozenLinearMap:
    """A fitted map together with the plain model it reads."""

    model: TopicVAE
    linear_map: LinearTopicMap

    def predict_texts(self, *, texts: Sequence[str]) -> np.ndarray:
        """Topic mixtures for texts."""

        return self.linear_map.predict(features=readout_features(model=self.model, texts=texts))


def fit_frozen_linear_map(
    *,
    model: TopicVAE,
    train_texts: Sequence[str],
    theta_train: np.ndarray,
    device: str,
    steps: int = 500,
) -> FrozenLinearMap:
    """Fit on the English training posts. The caller seeds torch first: the map starts from random weights."""

    if model.config.train_encoder_lora:
        raise ValueError("The frozen baseline reads a plain backbone: train_encoder_lora must be False.")
    features = readout_features(model=model, texts=train_texts)
    print(f"plain Qwen's features for {features.shape[0]:,} training posts: {features.shape}")
    return FrozenLinearMap(
        model=model,
        linear_map=fit_linear_map(features=features, theta=theta_train, device=device, steps=steps),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the frozen linear map alone.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-texts", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--linear-map-steps", type=int, default=500)
    args = parse_args_with_config(parser, argv)

    output_path = refuse_protected_path(Path(args.output_dir) / "report_frozen_linear_map.json", protected_dirs=PROTECTED_DIRS)
    device = resolve_device(args.device)
    paths = DataPaths(data_dir=args.data_dir)
    train_texts = load_corpus(paths=paths).train_texts[: args.max_train_texts]
    translated = load_translations(paths=paths).first(args.max_docs)
    teacher = LDATeacher(paths=paths)
    target = teacher.theta(translated.texts["en"])
    model = TopicVAE(config=TopicVAEConfig(base_model=args.base_model, train_encoder_lora=False)).to(device).eval()
    torch.manual_seed(args.seed)
    fitted = fit_frozen_linear_map(
        model=model,
        train_texts=train_texts,
        theta_train=teacher.theta(train_texts),
        device=device,
        steps=args.linear_map_steps,
    )
    thetas = {lang: fitted.predict_texts(texts=translated.texts[lang]) for lang in LANGS}
    report = {
        "topic": {lang: topic_score(theta_pred=thetas[lang], target=target) for lang in LANGS},
        "language": language_score(thetas=thetas),
    }
    write_json(output_path, report, indent=1)
    print(report)
    print("written to", output_path)


if __name__ == "__main__":
    main()
