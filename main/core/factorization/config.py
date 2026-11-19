"""Typed settings of the topic-aligned VAE and of the recipe that trains it.

What belongs here
-----------------
- ``LoraConfig``, ``ObjectiveConfig``, ``TopicVAEConfig``: what the model needs.
- ``TrainingRecipe``: every setting a trained model's ``model.pt`` records.

What does NOT belong here
-------------------------
- Paths, devices, and small-data options; those are run settings and live in
  ``experiments/factorization/config.py``.
- The names the settings carry inside a saved ``model.pt``; that mapping lives
  in ``core/factorization/checkpointing.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MuFrom = Literal["h", "delta"]
LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _require_non_negative_number(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}.")


def _require_positive_number(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}.")


@dataclass(frozen=True)
class LoraConfig:
    """A LoRA on the attention projections.

    ``alpha`` is always ``2 * rank``, so the update scale ``alpha / rank`` is 2
    for every rank.
    """

    rank: int = 16
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES

    def __post_init__(self) -> None:
        _require_positive_int("lora rank", self.rank)
        if not self.target_modules:
            raise ValueError("target_modules must name at least one module.")

    @property
    def alpha(self) -> int:
        """LoRA alpha, tied to the rank."""

        return 2 * self.rank


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights in ``rec + lam * KL(theta || theta_hat) + beta * KL(q || N(0, I))``."""

    lam: float = 1.0
    beta: float = 0.5

    def __post_init__(self) -> None:
        _require_non_negative_number("lam", self.lam)
        _require_non_negative_number("beta", self.beta)


@dataclass(frozen=True)
class TopicVAEConfig:
    """What the model is built from.

    Attributes
    ----------
    base_model:
        The language model behind both the posterior backbone and the generator.
    num_topics:
        Latent dimension, one coordinate per LDA topic.
    train_encoder_lora:
        Whether the posterior backbone gets a trained LoRA. The generator always
        has one.
    mu_from:
        The readout. ``"h"`` reads ``mu`` from the backbone state with its LoRA
        on. ``"delta"`` reads it from ``h_on - h_off``, what the LoRA adds, and
        so needs ``train_encoder_lora=True``.
    lora:
        LoRA settings for both copies.
    objective:
        Weights of the alignment and rate terms.
    encoder_max_tokens, decoder_max_tokens:
        Tokens the posterior reads and tokens the generator reconstructs.
    use_generator:
        Whether the loss has the reconstruction term. ``False`` is the ablation
        without a decoder: the generator is still built, so every random draw is
        the same, but it is frozen, never run, and not saved, and the loss is
        ``beta * rate / T + lam * align`` with ``T`` the generator's token count.
    """

    base_model: str = "Qwen/Qwen2.5-0.5B"
    num_topics: int = 20
    train_encoder_lora: bool = True
    mu_from: MuFrom = "h"
    lora: LoraConfig = field(default_factory=LoraConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    encoder_max_tokens: int = 512
    decoder_max_tokens: int = 128
    use_generator: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, str) or not self.base_model:
            raise ValueError(f"base_model must be a model name, got {self.base_model!r}.")
        _require_positive_int("num_topics", self.num_topics)
        _require_positive_int("encoder_max_tokens", self.encoder_max_tokens)
        _require_positive_int("decoder_max_tokens", self.decoder_max_tokens)
        if not isinstance(self.train_encoder_lora, bool):
            raise ValueError(f"train_encoder_lora must be a boolean, got {self.train_encoder_lora!r}.")
        if not isinstance(self.use_generator, bool):
            raise ValueError(f"use_generator must be a boolean, got {self.use_generator!r}.")
        if self.mu_from not in ("h", "delta"):
            raise ValueError(f"mu_from must be 'h' or 'delta', got {self.mu_from!r}.")
        if self.mu_from == "delta" and not self.train_encoder_lora:
            raise ValueError(
                "mu_from='delta' reads what the encoder's LoRA adds, so it needs train_encoder_lora=True."
            )


@dataclass(frozen=True)
class TrainingRecipe:
    """Everything a trained model records about how it was made.

    Attributes
    ----------
    model:
        The model's own settings.
    teacher:
        File name of the LDA teacher inside the data directory.
    min_words:
        Training posts with fewer of the teacher's words are left out.
    num_epochs, batch_size:
        Length of training and texts per step.
    val_frac:
        Share of the training texts held out to pick the epoch.
    lr_encoder_lora, lr_decoder_lora, lr_linear_maps:
        Learning rates of the three AdamW groups.
    seed:
        Seeds torch and the numpy generator that shuffles the data.
    """

    model: TopicVAEConfig = field(default_factory=TopicVAEConfig)
    teacher: str = "lda_octis.joblib"
    min_words: int = 5
    num_epochs: int = 4
    val_frac: float = 0.1
    batch_size: int = 16
    lr_encoder_lora: float = 5e-5
    lr_decoder_lora: float = 2e-4
    lr_linear_maps: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.teacher, str) or not self.teacher:
            raise ValueError(f"teacher must be a file name, got {self.teacher!r}.")
        _require_non_negative_number("min_words", self.min_words)
        _require_positive_int("num_epochs", self.num_epochs)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_number("lr_encoder_lora", self.lr_encoder_lora)
        _require_positive_number("lr_decoder_lora", self.lr_decoder_lora)
        _require_positive_number("lr_linear_maps", self.lr_linear_maps)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an integer, got {self.seed!r}.")
        if not isinstance(self.val_frac, (int, float)) or not 0.0 < self.val_frac < 1.0:
            raise ValueError(f"val_frac must be strictly between 0 and 1, got {self.val_frac!r}.")
