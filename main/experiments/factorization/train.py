"""Train one topic-aligned VAE.

    python -m experiments.factorization.train --config configs/paper/factorization_delta.json

1. seed; load the English training posts with enough teacher words and hold 10% out,
2. build the model, with AdamW over three groups and the warm-up cosine schedule,
3. train, scoring the held-out loss and the test documents after every epoch,
4. keep the epoch with the lowest held-out loss and write ``model.pt``,
   ``history.json`` and ``test.json``.

With ``checkpoint_steps`` the trained tensors are also saved at those steps, at step 0
before any update, and at the last step, into ``checkpoints/``. A skipped step still
counts as a step. Saving draws no random numbers, so the path of the run does not change.

What belongs here
-----------------
- The order of these steps. It fixes the order of every random draw, so it should
  not be rearranged: torch and numpy are seeded first, the held-out split draws from
  numpy, the model's construction draws from torch, and then every epoch draws one
  permutation from numpy and every batch one noise tensor from torch.

What does NOT belong here
-------------------------
- The arithmetic of a step (``train_step.py``) or of the loss (``core/factorization``).

The test set never picks anything. A folder that already holds a ``model.pt`` is refused,
and so is one with checkpoints left over when this run would save its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from core.factorization.checkpointing import list_checkpoints, load_trained_run, save_checkpoint, save_trained_model
from core.factorization.model import TopicVAE
from core.train.loop import (
    build_adamw,
    count_steps_per_epoch,
    epoch_batches,
    planned_checkpoint_steps,
    resolve_device,
    split_held_out,
)
from core.train.lr_schedules import build_warmup_cosine_scheduler
from core.train.seed import seed_run
from experiments.data.corpus import load_train_texts
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import TranslatedTestSet, load_translations
from experiments.factorization.config import TrainConfig, parse_train_config
from experiments.factorization.epoch_eval import held_out_loss, watch_test_scores
from experiments.factorization.output import gpu_name, write_test_rows
from experiments.factorization.train_step import train_one_step


@dataclass(frozen=True)
class TrainingData:
    """The texts of one run and the teacher's mixtures for them.

    Attributes
    ----------
    texts, theta:
        Texts trained on, and ``[texts, num_topics]`` mixtures.
    val_texts, val_theta:
        Held-out texts that pick the epoch.
    test_docs, theta_test:
        Test documents, and the teacher on their English text: the target in every language.
    """

    texts: list[str]
    theta: torch.Tensor
    val_texts: list[str]
    val_theta: torch.Tensor
    test_docs: TranslatedTestSet
    theta_test: np.ndarray


@dataclass(frozen=True)
class TrainRunOutput:
    """What a finished run holds; the model is in eval mode with the kept epoch's weights."""

    model: TopicVAE
    history: list[dict[str, float]]
    test_rows: list[dict[str, Any]]
    best_epoch: int
    best_val: float
    num_steps: int


def load_training_data(*, config: TrainConfig, rng: np.random.Generator) -> TrainingData:
    """Keep posts with enough teacher words, hold some out, and load the test documents."""

    recipe = config.recipe
    paths = DataPaths(data_dir=config.data_dir)
    texts = load_train_texts(paths=paths)
    if config.max_train_texts is not None:
        texts = texts[: config.max_train_texts]
    teacher = LDATeacher(paths=paths, name=recipe.teacher)
    words_per_post = np.asarray(teacher.counts(texts).sum(1)).ravel()
    texts = [text for text, count in zip(texts, words_per_post) if count >= recipe.min_words]
    theta = torch.tensor(teacher.theta(texts), dtype=torch.float32)
    split = split_held_out(rng=rng, num_texts=len(texts), val_frac=recipe.val_frac)
    test_docs = load_translations(paths=paths).first(config.max_test_docs)
    return TrainingData(
        texts=[texts[i] for i in split.kept],
        theta=theta[split.kept],
        val_texts=[texts[i] for i in split.held_out],
        val_theta=theta[split.held_out],
        test_docs=test_docs,
        theta_test=teacher.theta(test_docs.texts["en"]),
    )


def _epoch_row(*, model: TopicVAE, data: TrainingData, epoch: int, step: int) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "step": step,
        "val": held_out_loss(model=model, texts=data.val_texts, theta=data.val_theta),
        **watch_test_scores(model=model, docs=data.test_docs.texts, theta_target=data.theta_test),
    }


def train_run(*, config: TrainConfig) -> TrainRunOutput:
    """Train one model into ``config.output_dir``. See the module docstring."""

    run_dir = Path(config.output_dir)
    if (run_dir / "model.pt").exists():
        raise FileExistsError(f"{run_dir} already holds a model.pt; a trained model is never overwritten.")
    if config.checkpoint_steps and list_checkpoints(run_dir=run_dir):
        raise FileExistsError(f"{run_dir} already holds checkpoints from another run; pick another output_dir.")
    recipe = config.recipe
    device = resolve_device(config.device)
    print(f"training on gpu {gpu_name(device)}", flush=True)  # rounding differs by GPU type, so the log must say which

    rng = seed_run(seed=recipe.seed)
    data = load_training_data(config=config, rng=rng)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = TopicVAE(config=recipe.model).to(device)
    groups = model.trainable_parameter_groups()
    parameters = [parameter for group in groups.values() for parameter in group]
    optimizer = build_adamw(
        parameter_groups=groups,
        learning_rates={
            "encoder_lora": recipe.lr_encoder_lora,
            "decoder_lora": recipe.lr_decoder_lora,
            "linear_maps": recipe.lr_linear_maps,
        },
    )
    steps_per_epoch = count_steps_per_epoch(num_texts=len(data.texts), batch_size=recipe.batch_size)
    total_steps = recipe.num_epochs * steps_per_epoch
    scheduler = build_warmup_cosine_scheduler(optimizer=optimizer, total_steps=total_steps)
    print(
        f"{sum(parameter.numel() for parameter in parameters) / 1e6:.2f}M trainable parameters, "
        f"{len(data.texts)} training texts, {len(data.val_texts)} held out, {total_steps} steps"
    )

    last_step = total_steps if config.max_steps is None else min(total_steps, config.max_steps)
    checkpoint_steps = planned_checkpoint_steps(requested=config.checkpoint_steps, last_step=last_step)
    if checkpoint_steps:
        print(f"checkpoints at steps {sorted(checkpoint_steps)}")
    history: list[dict[str, float]] = []
    step, started = 0, time.time()
    test_rows = [_epoch_row(model=model, data=data, epoch=0, step=0)]
    print("test", test_rows[-1], flush=True)
    best: dict[str, Any] | None = None
    model.train()
    if 0 in checkpoint_steps:
        save_checkpoint(model=model, step=0, run_dir=run_dir)
    for epoch in range(recipe.num_epochs):
        for indices in epoch_batches(rng=rng, num_texts=len(data.texts), num_steps=steps_per_epoch):
            result = train_one_step(
                model=model,
                texts=[data.texts[i] for i in indices],
                theta=data.theta[indices].to(device),
                parameters=parameters,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            step += 1
            if step in checkpoint_steps:
                save_checkpoint(model=model, step=step, run_dir=run_dir)
            if result.took_step:
                history.append({"step": step, **result.stats})
                if step % config.log_every == 0:
                    print(
                        f"epoch {epoch} step {step}/{total_steps} | rec/token {result.stats['rec']:.3f} "
                        f"align {result.stats['align']:.3f} rate {result.stats['rate']:.1f} | "
                        f"{(time.time() - started) / 60:.1f} min",
                        flush=True,
                    )
            if config.max_steps is not None and step >= config.max_steps:
                break
        test_rows.append(_epoch_row(model=model, data=data, epoch=epoch + 1, step=step))
        print("test", test_rows[-1], flush=True)
        write_test_rows(run_dir=run_dir, rows=test_rows)
        if best is None or test_rows[-1]["val"] < best["val"]:
            best = {
                "epoch": epoch + 1,
                "val": test_rows[-1]["val"],
                "weights": {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad},
            }
        if config.max_steps is not None and step >= config.max_steps:
            break

    print(f"skipped {step - len(history)} of {step} steps (non-finite loss or gradient)")
    model.load_state_dict(best["weights"], strict=False)
    print(f"saving epoch {best['epoch']}, the lowest held-out loss ({best['val']:.3f})")
    model.eval()
    save_trained_model(model=model, recipe=recipe, history=history, run_dir=run_dir)
    return TrainRunOutput(
        model=model,
        history=history,
        test_rows=test_rows,
        best_epoch=best["epoch"],
        best_val=best["val"],
        num_steps=step,
    )


def train_or_load(*, config: TrainConfig) -> tuple[TopicVAE, list[dict[str, float]]]:
    """For notebooks: load the model when the folder already holds one, after checking its recipe; else train it."""

    run_dir = Path(config.output_dir)
    if not (run_dir / "model.pt").exists():
        output = train_run(config=config)
        return output.model, output.history
    run = load_trained_run(run_dir=run_dir, device=resolve_device(config.device))
    if run.recipe != config.recipe:
        raise ValueError(
            f"{run_dir} holds a model trained with {run.recipe}, not {config.recipe}. "
            "To use it as it is, call load_trained_run; to train this config, pick another output_dir."
        )
    print(f"loaded the trained model from {run_dir}: training skipped")
    return run.model, run.history


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = parse_train_config(argv)
    print(f"device {resolve_device(config.device)}, torch {torch.__version__}; the model goes to {config.output_dir}")
    print(config.recipe)
    train_run(config=config)


if __name__ == "__main__":
    main()
