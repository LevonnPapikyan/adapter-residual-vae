"""Command-line and JSON configuration for training and evaluation.

What belongs here
-----------------
- The immutable ``TrainConfig``, ``EvaluationConfig`` and ``TrajectoryConfig``.
- Their parsers, used by both the command line and the notebooks.

What does NOT belong here
-------------------------
- The training loop, the scores, and anything that writes files.

Paths are relative to ``main/``, the folder commands run from. The small-data
options (``max_*``, ``skip_*``) are for smoke tests; left unset they change nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.factorization.checkpointing import recipe_from_values
from core.factorization.config import TrainingRecipe
from core.runs.io import parse_args_with_config, refuse_protected_path
from experiments.data.paths import PROTECTED_DIRS


def _require_optional_positive_int(name: str, value: object) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{name} must be a positive integer or unset, got {value!r}.")


def _sorted_steps(name: str, values: Iterable[int]) -> tuple[int, ...]:
    values = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError(f"{name} must be a list of non-negative integers, got {values!r}.")
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class TrainConfig:
    """One training run.

    Attributes
    ----------
    recipe:
        Every setting ``model.pt`` records.
    data_dir, output_dir:
        Read-only data folder and the run's own folder. A folder that already holds
        a ``model.pt`` is refused.
    device:
        ``None`` means the GPU when there is one.
    log_every:
        Steps between progress lines.
    max_train_texts:
        Smoke tests: keep only the first raw training posts.
    max_test_docs:
        Smoke tests: watch only the first test documents after each epoch.
    max_steps:
        Smoke tests: stop after this many steps. The schedule is unchanged.
    checkpoint_steps:
        Steps whose trained tensors go to ``checkpoints/``. Empty saves none; otherwise
        step 0 and the last step are saved too. Not part of the recipe.
    """

    recipe: TrainingRecipe
    data_dir: Path
    output_dir: Path
    device: str | None = None
    log_every: int = 100
    max_train_texts: int | None = None
    max_test_docs: int | None = None
    max_steps: int | None = None
    checkpoint_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "checkpoint_steps", _sorted_steps("checkpoint_steps", self.checkpoint_steps))
        if not isinstance(self.log_every, int) or self.log_every <= 0:
            raise ValueError(f"log_every must be a positive integer, got {self.log_every!r}.")
        _require_optional_positive_int("max_train_texts", self.max_train_texts)
        _require_optional_positive_int("max_test_docs", self.max_test_docs)
        _require_optional_positive_int("max_steps", self.max_steps)
        refuse_protected_path(self.output_dir, protected_dirs=PROTECTED_DIRS)


def build_train_parser() -> argparse.ArgumentParser:
    """Every training setting as an option. Defaults are the final runs' settings."""

    defaults = TrainingRecipe()
    parser = argparse.ArgumentParser(description="Train one topic-aligned VAE.")
    parser.add_argument("--base-model", default=defaults.model.base_model)
    parser.add_argument("--num-topics", type=int, default=defaults.model.num_topics)
    parser.add_argument("--teacher", default=defaults.teacher)
    parser.add_argument("--min-words", type=int, default=defaults.min_words)
    parser.add_argument("--train-encoder-lora", action=argparse.BooleanOptionalAction, default=defaults.model.train_encoder_lora)
    parser.add_argument("--mu-from", choices=["h", "delta"], default=defaults.model.mu_from)
    parser.add_argument("--lora-rank", type=int, default=defaults.model.lora.rank)
    parser.add_argument("--lam", type=float, default=defaults.model.objective.lam)
    parser.add_argument("--beta", type=float, default=defaults.model.objective.beta)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--val-frac", type=float, default=defaults.val_frac)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--encoder-max-tokens", type=int, default=defaults.model.encoder_max_tokens)
    parser.add_argument("--decoder-max-tokens", type=int, default=defaults.model.decoder_max_tokens)
    parser.add_argument("--lr-encoder-lora", type=float, default=defaults.lr_encoder_lora)
    parser.add_argument("--lr-decoder-lora", type=float, default=defaults.lr_decoder_lora)
    parser.add_argument("--lr-linear-maps", type=float, default=defaults.lr_linear_maps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--use-generator", action=argparse.BooleanOptionalAction, default=defaults.model.use_generator)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="outputs/factorization")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-texts", type=int, default=None)
    parser.add_argument("--max-test-docs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    return parser


RUN_SETTINGS = ("data_dir", "output_dir", "device", "log_every", "max_train_texts", "max_test_docs", "max_steps", "checkpoint_steps")


def _train_config_from_values(values: dict[str, Any]) -> TrainConfig:
    run_settings = {name: values[name] for name in RUN_SETTINGS}
    recipe_values = {name: value for name, value in values.items() if name not in RUN_SETTINGS and name != "config"}
    return TrainConfig(recipe=recipe_from_values(recipe_values), **run_settings)


def parse_train_config(argv: list[str] | None = None) -> TrainConfig:
    """Defaults, then ``--config FILE``, then the command line."""

    return _train_config_from_values(vars(parse_args_with_config(build_train_parser(), argv)))


def train_config_from_file(path: str | Path, **overrides: Any) -> TrainConfig:
    """For notebooks: a JSON config with some settings changed, named as in the JSON file."""

    values = vars(parse_args_with_config(build_train_parser(), ["--config", str(path)]))
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}.")
    return _train_config_from_values({**values, **overrides})


@dataclass(frozen=True)
class EvaluationConfig:
    """One evaluation of a trained pair.

    Attributes
    ----------
    runs_dir:
        Folder holding the two runs, ``h/`` and ``delta/``.
    base_model:
        The language model they were trained on; plain, it is also the frozen baseline.
    data_dir, baselines_dir:
        Read-only data and the LDA baselines' reports.
    output_dir:
        Where ``results.json``, ``comparison.json`` and the figures go.
    reference_results:
        The ``results.json`` to compare against, or ``None``.
    device:
        ``None`` means the GPU when there is one.
    seed:
        Seeds torch before the frozen map, the sampling, and the layer scan.
    max_docs, max_train_texts, max_generation_texts:
        Smoke tests: keep only the first documents of each list.
    encoder_max_tokens:
        Smoke tests: read fewer tokens than the runs were trained with.
    linear_map_steps, generation_per_topic, generation_max_new_tokens:
        Fixed by the paper's runs; smaller values are for smoke tests.
    skip_layers, skip_generation:
        Smoke tests: leave out the two slowest scores, which then stay empty in ``results.json``.
    overwrite:
        Replace a ``results.json`` already in ``output_dir``. Otherwise that is refused.
    """

    runs_dir: Path
    base_model: str
    data_dir: Path
    baselines_dir: Path
    output_dir: Path
    reference_results: Path | None = None
    device: str | None = None
    seed: int = 0
    max_docs: int | None = None
    max_train_texts: int | None = None
    max_generation_texts: int | None = None
    encoder_max_tokens: int | None = None
    linear_map_steps: int = 500
    generation_per_topic: int = 10
    generation_max_new_tokens: int = 96
    skip_layers: bool = False
    skip_generation: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in ("runs_dir", "data_dir", "baselines_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.reference_results is not None:
            object.__setattr__(self, "reference_results", Path(self.reference_results))
        _require_optional_positive_int("max_train_texts", self.max_train_texts)
        _require_optional_positive_int("max_generation_texts", self.max_generation_texts)
        _require_optional_positive_int("encoder_max_tokens", self.encoder_max_tokens)
        _require_optional_positive_int("linear_map_steps", self.linear_map_steps)
        _require_optional_positive_int("generation_per_topic", self.generation_per_topic)
        _require_optional_positive_int("generation_max_new_tokens", self.generation_max_new_tokens)
        if self.max_docs is not None and (not isinstance(self.max_docs, int) or self.max_docs < 5):
            raise ValueError(f"max_docs must be at least 5, the language probe's folds, or unset; got {self.max_docs!r}.")
        refuse_protected_path(self.output_dir, protected_dirs=PROTECTED_DIRS)

    @property
    def is_partial(self) -> bool:
        """Whether a small-data option is set, so the numbers cannot match the paper's."""

        return (
            self.max_docs is not None
            or self.max_train_texts is not None
            or self.max_generation_texts is not None
            or self.encoder_max_tokens is not None
            or self.linear_map_steps != 500
            or self.generation_per_topic != 10
            or self.generation_max_new_tokens != 96
            or self.skip_layers
            or self.skip_generation
        )


def build_evaluation_parser() -> argparse.ArgumentParser:
    """Every evaluation setting as an option."""

    parser = argparse.ArgumentParser(description="Score a trained pair: every number and figure of the paper.")
    parser.add_argument("--runs-dir", default="../models/final_0.5b")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--baselines-dir", default="../models/baselines")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--reference-results", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-train-texts", type=int, default=None)
    parser.add_argument("--max-generation-texts", type=int, default=None)
    parser.add_argument("--encoder-max-tokens", type=int, default=None)
    parser.add_argument("--linear-map-steps", type=int, default=500)
    parser.add_argument("--generation-per-topic", type=int, default=10)
    parser.add_argument("--generation-max-new-tokens", type=int, default=96)
    parser.add_argument("--skip-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-generation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_evaluation_config(argv: list[str] | None = None) -> EvaluationConfig:
    """Defaults, then ``--config FILE``, then the command line."""

    values = vars(parse_args_with_config(build_evaluation_parser(), argv))
    values.pop("config")
    return EvaluationConfig(**values)


def evaluation_config_from_file(path: str | Path, **overrides: Any) -> EvaluationConfig:
    """For notebooks: a JSON config with some settings changed, named as in the JSON file."""

    values = vars(parse_args_with_config(build_evaluation_parser(), ["--config", str(path)]))
    values.pop("config")
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}.")
    return EvaluationConfig(**{**values, **overrides})


@dataclass(frozen=True)
class TrajectoryConfig:
    """One analysis of a checkpointed pair.

    Attributes
    ----------
    runs_dir:
        Folder holding the two runs, ``h/`` and ``delta/``, each with ``checkpoints/``. Read only.
    data_dir:
        Read-only data folder.
    output_dir:
        ``results.json``, ``features/`` (the cache) and ``figures/`` go here.
    device:
        ``None`` means the GPU when there is one.
    batch_size:
        Texts per batch, as in the evaluation.
    train_sample, train_sample_seed:
        How many English training posts are encoded, from the training split, and the seed that picks them.
    map_steps:
        The steps wanted on the map grids. The nearest saved step stands in for each.
    tsne_perplexity, tsne_seed, pca_dims:
        t-SNE settings; it runs on this many PCA features, from a PCA start.
    max_docs, max_train_texts:
        Smoke tests: keep only the first documents of each list. ``max_train_texts`` must
        be the value the runs were trained with, so the held-out split comes out the same.
    saved_runs_dir, reference_results:
        Optional: the saved ``h/`` and ``delta/`` weights to score with the same code, and
        the ``results.json`` to compare against. Both read only.
    encoder_max_tokens:
        Smoke tests: read fewer tokens from the saved weights in ``saved_runs_dir``.
    figures_only:
        Encode nothing and load no model: everything from the cache, no GPU.
    saved_only:
        Only score the weights in ``saved_runs_dir`` and write ``saved_weights_check.json``.
    """

    runs_dir: Path
    data_dir: Path
    output_dir: Path
    device: str | None = None
    batch_size: int = 32
    train_sample: int = 1000
    train_sample_seed: int = 0
    map_steps: tuple[int, ...] = (0, 10, 20, 50, 100, 200, 400, 825, 1650, 2200)
    tsne_perplexity: float = 30.0
    tsne_seed: int = 0
    pca_dims: int = 50
    max_docs: int | None = None
    max_train_texts: int | None = None
    saved_runs_dir: Path | None = None
    reference_results: Path | None = None
    encoder_max_tokens: int | None = None
    figures_only: bool = False
    saved_only: bool = False

    def __post_init__(self) -> None:
        for name in ("runs_dir", "data_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("saved_runs_dir", "reference_results"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "map_steps", _sorted_steps("map_steps", self.map_steps))
        _require_optional_positive_int("batch_size", self.batch_size)
        _require_optional_positive_int("train_sample", self.train_sample)
        _require_optional_positive_int("pca_dims", self.pca_dims)
        _require_optional_positive_int("max_train_texts", self.max_train_texts)
        _require_optional_positive_int("encoder_max_tokens", self.encoder_max_tokens)
        if not self.map_steps:
            raise ValueError("map_steps must name at least one step.")
        if self.max_docs is not None and (not isinstance(self.max_docs, int) or self.max_docs < 5):
            raise ValueError(f"max_docs must be at least 5, the language probe's folds, or unset; got {self.max_docs!r}.")
        if self.figures_only and self.saved_only:
            raise ValueError("figures_only and saved_only exclude each other.")
        if self.saved_only and self.saved_runs_dir is None:
            raise ValueError("saved_only needs saved_runs_dir.")
        refuse_protected_path(self.output_dir, protected_dirs=PROTECTED_DIRS)

    @property
    def is_partial(self) -> bool:
        """Whether a small-data option is set, so the numbers cannot match the paper's."""

        return self.max_docs is not None or self.max_train_texts is not None or self.encoder_max_tokens is not None


def build_trajectory_parser() -> argparse.ArgumentParser:
    """Every trajectory setting as an option."""

    parser = argparse.ArgumentParser(description="Every checkpoint of a pair on the test documents and a training sample.")
    parser.add_argument("--runs-dir", default="outputs/checkpointed")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="outputs/checkpointed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-sample", type=int, default=1000)
    parser.add_argument("--train-sample-seed", type=int, default=0)
    parser.add_argument("--map-steps", type=int, nargs="*", default=[0, 10, 20, 50, 100, 200, 400, 825, 1650, 2200])
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-train-texts", type=int, default=None)
    parser.add_argument("--saved-runs-dir", default=None)
    parser.add_argument("--reference-results", default=None)
    parser.add_argument("--encoder-max-tokens", type=int, default=None)
    parser.add_argument("--figures-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--saved-only", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_trajectory_config(argv: list[str] | None = None) -> TrajectoryConfig:
    """Defaults, then ``--config FILE``, then the command line."""

    values = vars(parse_args_with_config(build_trajectory_parser(), argv))
    values.pop("config")
    return TrajectoryConfig(**values)
