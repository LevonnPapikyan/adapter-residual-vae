"""A trained model on disk, and the one place that maps saved names to current names.

A trained run is a folder:

- ``model.pt``: ``{"config": {...}, "weights": {name: tensor}}`` with only the
  trained tensors (the frozen weights come from the hub),
- ``history.json``: the loss parts of every step taken,
- ``test.json``: the per-epoch scores, written while training,
- ``checkpoints/ckpt_{step}.pt`` when the run asked for them: ``{"step": int,
  "weights": {name: tensor}}`` with only the trained tensors at that step, under the
  same saved names as ``model.pt``.

The saved names are the ones the final models were written with (``enc.*``,
``dec.*``, ``prefix.*``, ``K``, ``lora_r``, ...). They are not changed, so old files
load as they are.

The format is not identical in the other direction. Next to the shared names, shapes and
key order, a new file adds the setting ``base``, drops the five settings the final runs
fixed (``rate_per_doc``, ``prior_var``, ``delta_norm``, ``no_norm``, ``lr_schedule``) and
drops the unused ``norm.weight`` and ``norm.bias``, so it holds 390 tensors, not 392. The
flat code's last revision reads a new file (``tests/test_flat_equivalence.py``); the
earlier revision in ``archive-2026-10-30`` refuses it.

What belongs here
-----------------
- ``SAVED_TO_MODULE_PREFIX`` and ``SAVED_TO_RECIPE_FIELD``: the mapping.
- Saving and loading a trained model, including refusing a model this code cannot
  run as it was trained.
- Saving and loading a checkpoint. Saving draws no random numbers, and asserts that.

What does NOT belong here
-------------------------
- The file format's outer validation (``core/checkpoints/io.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from core.checkpoints.io import load_model_file, save_model_file
from core.factorization.config import LoraConfig, ObjectiveConfig, TopicVAEConfig, TrainingRecipe
from core.factorization.model import TopicVAE
from core.runs.io import read_json, write_json

# Saved tensor names to module names. An unused LayerNorm is the only other thing the final files hold.
SAVED_TO_MODULE_PREFIX: tuple[tuple[str, str], ...] = (
    ("enc.", "posterior_backbone."),
    ("dec.", "generator."),
    ("mu_head.", "mu_head."),
    ("logvar_head.", "logvar_head."),
    ("prefix.", "z_to_prefix."),
)
UNUSED_SAVED_WEIGHTS = frozenset({"norm.weight", "norm.bias"})
CHECKPOINT_DIR = "checkpoints"

# Saved setting names, in the order they are written, to the settings they mean.
SAVED_TO_RECIPE_FIELD: tuple[tuple[str, str], ...] = (
    ("base", "base_model"),
    ("K", "num_topics"),
    ("teacher", "teacher"),
    ("min_words", "min_words"),
    ("enc_lora", "train_encoder_lora"),
    ("mu_from", "mu_from"),
    ("lora_r", "lora_rank"),
    ("lam", "lam"),
    ("beta", "beta"),
    ("epochs", "num_epochs"),
    ("val_frac", "val_frac"),
    ("batch_size", "batch_size"),
    ("enc_len", "encoder_max_tokens"),
    ("dec_len", "decoder_max_tokens"),
    ("lr_enc", "lr_encoder_lora"),
    ("lr_dec", "lr_decoder_lora"),
    ("lr_head", "lr_linear_maps"),
    ("seed", "seed"),
    ("use_gen", "use_generator"),
)


def saved_name_to_module_name(saved_name: str) -> str:
    """The module name a saved tensor belongs to."""

    for saved_prefix, module_prefix in SAVED_TO_MODULE_PREFIX:
        if saved_name.startswith(saved_prefix):
            return module_prefix + saved_name[len(saved_prefix) :]
    raise ValueError(f"Unknown saved tensor name: {saved_name!r}.")


def module_name_to_saved_name(module_name: str) -> str:
    """The name a module's tensor is saved under."""

    for saved_prefix, module_prefix in SAVED_TO_MODULE_PREFIX:
        if module_name.startswith(module_prefix):
            return saved_prefix + module_name[len(module_prefix) :]
    raise ValueError(f"Unknown module tensor name: {module_name!r}.")


def recipe_to_saved_config(recipe: TrainingRecipe) -> dict[str, Any]:
    """The ``config`` dictionary written into ``model.pt``."""

    model = recipe.model
    values = {
        "base_model": model.base_model,
        "num_topics": model.num_topics,
        "teacher": recipe.teacher,
        "min_words": recipe.min_words,
        "train_encoder_lora": model.train_encoder_lora,
        "mu_from": model.mu_from,
        "lora_rank": model.lora.rank,
        "lam": model.objective.lam,
        "beta": model.objective.beta,
        "num_epochs": recipe.num_epochs,
        "val_frac": recipe.val_frac,
        "batch_size": recipe.batch_size,
        "encoder_max_tokens": model.encoder_max_tokens,
        "decoder_max_tokens": model.decoder_max_tokens,
        "lr_encoder_lora": recipe.lr_encoder_lora,
        "lr_decoder_lora": recipe.lr_decoder_lora,
        "lr_linear_maps": recipe.lr_linear_maps,
        "seed": recipe.seed,
        "use_generator": model.use_generator,
    }
    saved = {saved: values[field] for saved, field in SAVED_TO_RECIPE_FIELD}
    if model.use_generator:  # written only for the ablation, so a full model's file is as before
        del saved["use_gen"]
    return saved


def recipe_from_values(values: Mapping[str, Any]) -> TrainingRecipe:
    """Build a recipe from current field names. A field left out takes the recipe's default."""

    defaults = TrainingRecipe()
    model = defaults.model
    get = values.get
    return TrainingRecipe(
        model=TopicVAEConfig(
            base_model=get("base_model", model.base_model),
            num_topics=get("num_topics", model.num_topics),
            train_encoder_lora=get("train_encoder_lora", model.train_encoder_lora),
            mu_from=get("mu_from", model.mu_from),
            lora=LoraConfig(rank=get("lora_rank", model.lora.rank)),
            objective=ObjectiveConfig(lam=get("lam", model.objective.lam), beta=get("beta", model.objective.beta)),
            encoder_max_tokens=get("encoder_max_tokens", model.encoder_max_tokens),
            decoder_max_tokens=get("decoder_max_tokens", model.decoder_max_tokens),
            use_generator=get("use_generator", model.use_generator),
        ),
        teacher=get("teacher", defaults.teacher),
        min_words=get("min_words", defaults.min_words),
        num_epochs=get("num_epochs", defaults.num_epochs),
        val_frac=get("val_frac", defaults.val_frac),
        batch_size=get("batch_size", defaults.batch_size),
        lr_encoder_lora=get("lr_encoder_lora", defaults.lr_encoder_lora),
        lr_decoder_lora=get("lr_decoder_lora", defaults.lr_decoder_lora),
        lr_linear_maps=get("lr_linear_maps", defaults.lr_linear_maps),
        seed=get("seed", defaults.seed),
    )


def recipe_from_saved_config(saved: Mapping[str, Any]) -> tuple[TrainingRecipe, dict[str, Any]]:
    """Read a saved ``config``. Returns the recipe and the saved settings that no longer exist.

    The final models were saved before ``base`` was a setting for the 0.5B pair, so
    a missing ``base`` means the default, Qwen2.5-0.5B.
    """

    to_field = dict(SAVED_TO_RECIPE_FIELD)
    values = {to_field[key]: value for key, value in saved.items() if key in to_field}
    dropped = {key: value for key, value in saved.items() if key not in to_field}
    return recipe_from_values(values), dropped


@dataclass(frozen=True)
class LoadedRun:
    """A trained model as it was trained.

    Attributes
    ----------
    model:
        The model in eval mode.
    history:
        Rows of ``history.json``.
    recipe:
        The settings from the run's own ``model.pt``.
    run_dir:
        The folder it came from.
    """

    model: TopicVAE
    history: list[dict[str, float]]
    recipe: TrainingRecipe
    run_dir: Path


def save_trained_model(
    *,
    model: TopicVAE,
    recipe: TrainingRecipe,
    history: list[dict[str, float]],
    run_dir: str | Path,
) -> Path:
    """Write ``model.pt`` and ``history.json``. Refuses a folder that already holds a ``model.pt``."""

    run_dir = Path(run_dir)
    weights = {
        module_name_to_saved_name(name): parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    path = save_model_file(run_dir / "model.pt", config=recipe_to_saved_config(recipe), weights=weights)
    write_json(run_dir / "history.json", history, indent=None)
    return path


def load_trained_run(
    *,
    run_dir: str | Path,
    device: str | torch.device,
    encoder_max_tokens: int | None = None,
) -> LoadedRun:
    """Load a trained model with the settings in its own ``model.pt``.

    Strict about every tensor except the two unused LayerNorm ones, and refuses a
    model trained with that LayerNorm in use (``no_norm=False``), since this code
    cannot run it as it was trained.

    ``encoder_max_tokens`` is for smoke tests only; left unset, the saved value is
    used.
    """

    run_dir = Path(run_dir)
    path = run_dir / "model.pt"
    saved = load_model_file(path, map_location="cpu")
    has_unused_norm = not UNUSED_SAVED_WEIGHTS.isdisjoint(saved.weights)
    if has_unused_norm and not saved.config.get("no_norm", False):
        raise ValueError(
            f"{path} was trained with a LayerNorm before mu (no_norm=False); this code has none, "
            "so it cannot run this model as it was trained."
        )
    recipe, dropped = recipe_from_saved_config(saved.config)
    if dropped:
        print(f"{run_dir}: ignoring settings that are now fixed: {dropped}")
    model_config = recipe.model
    if encoder_max_tokens is not None:
        model_config = replace(model_config, encoder_max_tokens=encoder_max_tokens)
    model = TopicVAE(config=model_config)

    weights = {
        saved_name_to_module_name(name): tensor
        for name, tensor in saved.weights.items()
        if name not in UNUSED_SAVED_WEIGHTS
    }
    trained = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing, unexpected = sorted(trained - weights.keys()), sorted(weights.keys() - trained)
    if missing or unexpected:
        raise ValueError(
            f"{path} does not hold the trained tensors of {recipe.model}\n"
            f"  {len(missing)} missing from the file, e.g. {missing[:3]}\n"
            f"  {len(unexpected)} in the file but not in the model, e.g. {unexpected[:3]}"
        )
    model.load_state_dict(weights, strict=False)  # strict=False only because the frozen weights are not in the file
    print(f"{run_dir}: {len(weights)} trained tensors loaded, {len(saved.weights) - len(weights)} unused ones ignored")
    history = read_json(run_dir / "history.json")
    return LoadedRun(model=model.to(device).eval(), history=history, recipe=recipe, run_dir=run_dir)


def checkpoint_path(*, run_dir: str | Path, step: int) -> Path:
    """Where the checkpoint of a step lives."""

    return Path(run_dir) / CHECKPOINT_DIR / f"ckpt_{step}.pt"


def list_checkpoints(*, run_dir: str | Path) -> dict[int, Path]:
    """The checkpoints of a run by step, in step order. Empty when there are none."""

    found = {}
    for path in (Path(run_dir) / CHECKPOINT_DIR).glob("ckpt_*.pt"):
        match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def _rng_states() -> list[torch.Tensor]:
    return [torch.get_rng_state()] + (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def save_checkpoint(*, model: torch.nn.Module, step: int, run_dir: str | Path) -> Path:
    """Write the trained tensors and the step. Never overwrites, and draws no random numbers."""

    path = checkpoint_path(run_dir=run_dir, step=step)
    if path.exists():
        raise FileExistsError(f"{path} already exists; a checkpoint is never overwritten.")
    before = _rng_states()
    weights = {
        module_name_to_saved_name(name): parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "weights": weights}, path)
    after = _rng_states()
    if len(before) != len(after) or not all(torch.equal(a, b) for a, b in zip(before, after)):
        raise AssertionError(f"saving {path} moved the random state.")
    return path


def read_checkpoint(path: str | Path) -> tuple[int, dict[str, torch.Tensor]]:
    """The step and the trained tensors of a checkpoint, under their saved names, on the CPU."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"step", "weights"}:
        raise ValueError(f"{path} must hold exactly 'step' and 'weights'.")
    return int(payload["step"]), payload["weights"]


def load_checkpoint(*, model: torch.nn.Module, path: str | Path) -> int:
    """Put a checkpoint's tensors into a model, strict about which tensors. Returns the step."""

    step, saved = read_checkpoint(path)
    weights = {saved_name_to_module_name(name): tensor for name, tensor in saved.items()}
    trained = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing, unexpected = sorted(trained - weights.keys()), sorted(weights.keys() - trained)
    if missing or unexpected:
        raise ValueError(
            f"{path} does not hold the trained tensors of this model\n"
            f"  {len(missing)} missing from the file, e.g. {missing[:3]}\n"
            f"  {len(unexpected)} in the file but not in the model, e.g. {unexpected[:3]}"
        )
    model.load_state_dict(weights, strict=False)  # strict=False only because the frozen weights are not in the file
    return step
