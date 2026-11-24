"""Configs parse, and bad settings fail with clear messages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conftest import MAIN_ROOT, RUNS_DIR, needs_weights
from core.factorization.checkpointing import recipe_from_saved_config, recipe_from_values, recipe_to_saved_config
from core.factorization.config import LoraConfig, ObjectiveConfig, TopicVAEConfig, TrainingRecipe
from experiments.factorization.config import (
    EvaluationConfig,
    parse_evaluation_config,
    parse_train_config,
    train_config_from_file,
)

PAPER = MAIN_ROOT / "configs" / "paper"
SMALL = MAIN_ROOT / "configs" / "small_cpu_example"


def test_cli_overrides_config_and_defaults(tmp_path: Path) -> None:
    config = parse_train_config(["--config", str(PAPER / "factorization_delta.json"), "--seed", "7", "--output-dir", str(tmp_path / "x")])
    assert config.recipe.seed == 7
    assert config.recipe.model.mu_from == "delta"
    assert config.output_dir == tmp_path / "x"
    assert config.max_train_texts is None and config.max_steps is None and config.max_test_docs is None


@pytest.mark.parametrize("name", sorted(path.name for path in SMALL.glob("factorization_*.json")) )
def test_small_configs_parse(name: str) -> None:
    config = train_config_from_file(SMALL / name)
    assert config.max_train_texts == 16 and config.recipe.batch_size == 4


def test_evaluation_configs_parse() -> None:
    assert parse_evaluation_config(["--config", str(PAPER / "evaluation.json")]).is_partial is False
    assert parse_evaluation_config(["--config", str(SMALL / "evaluation.json")]).is_partial is True


def test_unknown_json_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"mu_form": "h"}))
    with pytest.raises(ValueError, match="mu_form"):
        parse_train_config(["--config", str(path)])


def test_non_object_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="object"):
        parse_train_config(["--config", str(path)])


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: TopicVAEConfig(mu_from="bad"), "mu_from"),
        (lambda: TopicVAEConfig(mu_from="delta", train_encoder_lora=False), "train_encoder_lora"),
        (lambda: TopicVAEConfig(num_topics=0), "num_topics"),
        (lambda: LoraConfig(rank=0), "rank"),
        (lambda: ObjectiveConfig(lam=-1.0), "lam"),
        (lambda: TrainingRecipe(val_frac=1.5), "val_frac"),
        (lambda: TrainingRecipe(batch_size=2.5), "batch_size"),
        (lambda: TrainingRecipe(lr_linear_maps=0.0), "lr_linear_maps"),
    ],
)
def test_bad_settings_raise_value_error(build, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_outputs_cannot_go_into_protected_folders() -> None:
    with pytest.raises(ValueError, match="read-only"):
        train_config_from_file(PAPER / "factorization_h.json", output_dir=MAIN_ROOT.parent / "models" / "final_0.5b" / "x")
    with pytest.raises(ValueError, match="read-only"):
        EvaluationConfig(
            runs_dir="../models/final_0.5b",
            base_model="Qwen/Qwen2.5-0.5B",
            data_dir="../data",
            baselines_dir="../models/baselines",
            output_dir=MAIN_ROOT.parent / "figures",
        )


@pytest.mark.parametrize(
    "folder",
    [
        MAIN_ROOT.parent / "models",
        MAIN_ROOT.parent / "data",
        MAIN_ROOT.parent / "figures",
    ],
    ids=lambda path: path.name,
)
def test_every_shared_folder_is_protected(folder) -> None:
    with pytest.raises(ValueError, match="read-only"):
        train_config_from_file(PAPER / "factorization_h.json", output_dir=folder / "x")


def test_small_data_options_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        train_config_from_file(PAPER / "factorization_h.json", max_steps=0)


def test_saved_config_round_trip() -> None:
    recipe = recipe_from_values({"mu_from": "delta", "lora_rank": 8, "num_epochs": 2, "lam": 0.0})
    restored, dropped = recipe_from_saved_config(recipe_to_saved_config(recipe))
    assert restored == recipe and dropped == {}
    assert list(recipe_to_saved_config(recipe)) == [
        "base", "K", "teacher", "min_words", "enc_lora", "mu_from", "lora_r", "lam", "beta", "epochs",
        "val_frac", "batch_size", "enc_len", "dec_len", "lr_enc", "lr_dec", "lr_head", "seed",
    ]


@needs_weights
@pytest.mark.parametrize(
    ("run", "config_name"),
    [
        ("final_0.5b/h", "factorization_h.json"),
        ("final_0.5b/delta", "factorization_delta.json"),
    ],
)
def test_paper_configs_are_the_saved_recipes(run: str, config_name: str) -> None:
    saved = torch.load(RUNS_DIR / run / "model.pt", map_location="cpu")["config"]
    recipe, dropped = recipe_from_saved_config(saved)
    assert train_config_from_file(PAPER / config_name).recipe == recipe
    assert set(dropped) == {"no_norm", "delta_norm", "rate_per_doc", "prior_var", "lr_schedule"}
