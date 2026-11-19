"""Public exports for the topic-aligned VAE."""

from core.factorization.adaptation import apply_lora
from core.factorization.checkpointing import (
    LoadedRun,
    checkpoint_path,
    list_checkpoints,
    load_checkpoint,
    load_trained_run,
    read_checkpoint,
    recipe_from_saved_config,
    recipe_from_values,
    recipe_to_saved_config,
    save_checkpoint,
    save_trained_model,
)
from core.factorization.config import LoraConfig, ObjectiveConfig, TopicVAEConfig, TrainingRecipe
from core.factorization.model import ReconstructionOutput, TopicVAE
from core.factorization.objective import ObjectiveOutput, compute_objective
from core.factorization.posterior import PosteriorOutput, sample_z

__all__ = [
    "LoadedRun",
    "LoraConfig",
    "ObjectiveConfig",
    "ObjectiveOutput",
    "PosteriorOutput",
    "ReconstructionOutput",
    "TopicVAE",
    "TopicVAEConfig",
    "TrainingRecipe",
    "apply_lora",
    "checkpoint_path",
    "compute_objective",
    "list_checkpoints",
    "load_checkpoint",
    "load_trained_run",
    "read_checkpoint",
    "recipe_from_saved_config",
    "recipe_from_values",
    "recipe_to_saved_config",
    "sample_z",
    "save_checkpoint",
    "save_trained_model",
]
