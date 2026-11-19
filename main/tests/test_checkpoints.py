"""Saving checkpoints changes nothing about a run, and they load back. The real runs take minutes on one CPU thread."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from conftest import DATA_DIR, MAIN_ROOT, needs_data, needs_qwen
from core.analysis.encoding import encoded_states, run_encoder_in_float32
from core.factorization.checkpointing import checkpoint_path, list_checkpoints, load_checkpoint, read_checkpoint, save_checkpoint
from core.factorization.model import TopicVAE
from core.train.loop import planned_checkpoint_steps
from experiments.factorization.config import parse_evaluation_config, parse_train_config, train_config_from_file
from experiments.factorization.train import train_run

PAPER = MAIN_ROOT / "configs" / "paper"
SMALL = MAIN_ROOT / "configs" / "small_cpu_example"
TEXTS = ["The goalie made thirty saves and the team won.", "Der Torwart hielt dreissig Schuesse."]


class _TinyModel(torch.nn.Module):
    """The linear maps alone: enough for the saved names, with no language model."""

    def __init__(self) -> None:
        super().__init__()
        self.mu_head = torch.nn.Linear(3, 2)
        self.logvar_head = torch.nn.Linear(3, 2)
        self.frozen = torch.nn.Linear(3, 3).requires_grad_(False)


def test_checkpoint_steps_default_to_none(tmp_path: Path) -> None:
    config = parse_train_config(["--config", str(PAPER / "factorization_delta.json"), "--output-dir", str(tmp_path / "x")])
    assert config.checkpoint_steps == ()
    assert planned_checkpoint_steps(requested=config.checkpoint_steps, last_step=2200) == frozenset()


def test_planned_steps_add_the_first_and_the_last() -> None:
    assert planned_checkpoint_steps(requested=(10, 20, 5000), last_step=100) == frozenset({0, 10, 20, 100})


@pytest.mark.parametrize("bad", [[-1], [1.5], [True]])
def test_bad_checkpoint_steps_are_refused(bad: list) -> None:
    with pytest.raises(ValueError, match="checkpoint_steps"):
        train_config_from_file(PAPER / "factorization_h.json", checkpoint_steps=bad)


@pytest.mark.parametrize("name", ["h", "delta"])
def test_checkpointed_configs_are_the_same_recipes(name: str) -> None:
    plain = train_config_from_file(PAPER / f"factorization_{name}.json")
    checkpointed = train_config_from_file(PAPER / f"checkpointed_{name}.json")
    assert checkpointed.recipe == plain.recipe
    assert checkpointed.output_dir == Path(f"outputs/checkpointed/{name}")
    assert checkpointed.checkpoint_steps == (*range(0, 201, 10), 300, 400, 550, 825, 1100, 1650, 2200)
    assert train_config_from_file(SMALL / f"checkpointed_{name}.json").checkpoint_steps == (0, 2, 4, 6)


def test_evaluation_of_the_checkpointed_pair_parses() -> None:
    config = parse_evaluation_config(["--config", str(PAPER / "evaluation_checkpointed.json")])
    assert config.runs_dir == Path("outputs/checkpointed") and config.is_partial is False


def test_saving_does_not_move_the_random_state(tmp_path: Path) -> None:
    model = _TinyModel()
    torch.manual_seed(3)
    before = torch.get_rng_state()
    save_checkpoint(model=model, step=7, run_dir=tmp_path)
    assert torch.equal(before, torch.get_rng_state())
    step, weights = read_checkpoint(checkpoint_path(run_dir=tmp_path, step=7))
    assert step == 7 and set(weights) == {"mu_head.weight", "mu_head.bias", "logvar_head.weight", "logvar_head.bias"}


def test_a_save_that_draws_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_save = torch.save

    def drawing_save(*args, **kwargs) -> None:
        torch.rand(1)
        real_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", drawing_save)
    with pytest.raises(AssertionError, match="random state"):
        save_checkpoint(model=_TinyModel(), step=0, run_dir=tmp_path)


def test_checkpoints_are_never_overwritten_and_load_strictly(tmp_path: Path) -> None:
    model = _TinyModel()
    path = save_checkpoint(model=model, step=0, run_dir=tmp_path)
    with pytest.raises(FileExistsError, match="never overwritten"):
        save_checkpoint(model=model, step=0, run_dir=tmp_path)
    assert list_checkpoints(run_dir=tmp_path) == {0: path}

    other = _TinyModel()
    assert load_checkpoint(model=other, path=path) == 0
    assert torch.equal(other.mu_head.weight, model.mu_head.weight) and not torch.equal(other.frozen.weight, model.frozen.weight)

    other.extra = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="does not hold the trained tensors"):
        load_checkpoint(model=other, path=path)


def _small_config(*, run_dir: Path, mu_from: str = "delta", **overrides):
    config = train_config_from_file(
        SMALL / f"factorization_{mu_from}.json",
        data_dir=str(DATA_DIR),
        output_dir=str(run_dir),
        device="cpu",
        max_train_texts=12,
        max_test_docs=2,
        max_steps=3,
        **overrides,
    )
    model = replace(config.recipe.model, encoder_max_tokens=16, decoder_max_tokens=12)
    return replace(config, recipe=replace(config.recipe, model=model, batch_size=4))


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory: pytest.TempPathFactory):
    """The same seed without and with checkpoints."""

    root = tmp_path_factory.mktemp("runs")
    plain_config = _small_config(run_dir=root / "plain")
    saved_config = _small_config(run_dir=root / "saved", checkpoint_steps=[1])
    assert plain_config.recipe == saved_config.recipe
    return train_run(config=plain_config), train_run(config=saved_config), saved_config


@needs_data
@needs_qwen
@pytest.mark.needs_model
def test_no_checkpoint_steps_writes_only_the_three_files(two_runs) -> None:
    plain, _, saved_config = two_runs
    plain_dir = saved_config.output_dir.parent / "plain"
    assert sorted(path.name for path in plain_dir.iterdir()) == ["history.json", "model.pt", "test.json"]
    assert plain.num_steps == 3


@needs_data
@needs_qwen
@pytest.mark.needs_model
def test_same_seed_with_and_without_checkpoints_is_identical(two_runs) -> None:
    plain, saved, saved_config = two_runs
    plain_dir, saved_dir = saved_config.output_dir.parent / "plain", saved_config.output_dir
    assert plain.history == saved.history
    plain_file = torch.load(plain_dir / "model.pt", map_location="cpu")
    saved_file = torch.load(saved_dir / "model.pt", map_location="cpu")
    assert plain_file["config"] == saved_file["config"] and plain_file["weights"].keys() == saved_file["weights"].keys()
    assert all(torch.equal(plain_file["weights"][name], saved_file["weights"][name]) for name in plain_file["weights"])
    assert json.loads((plain_dir / "history.json").read_text()) == json.loads((saved_dir / "history.json").read_text())
    assert json.loads((plain_dir / "test.json").read_text()) == json.loads((saved_dir / "test.json").read_text())


@needs_data
@needs_qwen
@pytest.mark.needs_model
def test_a_checkpoint_loads_into_a_model_with_the_right_tensors(two_runs) -> None:
    _, saved, saved_config = two_runs
    checkpoints = list_checkpoints(run_dir=saved_config.output_dir)
    assert list(checkpoints) == [0, 1, 3]  # step 0 and the last step come on top of the one asked for

    _, final = read_checkpoint(checkpoints[3])
    kept = torch.load(saved_config.output_dir / "model.pt", map_location="cpu")["weights"]
    assert final.keys() == kept.keys() and all(torch.equal(final[name], kept[name]) for name in final)  # one epoch, so the last is the kept

    model = TopicVAE(config=saved_config.recipe.model).eval()
    assert load_checkpoint(model=model, path=checkpoints[3]) == 3
    parameters = dict(saved.model.named_parameters())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter.detach(), parameters[name].detach().cpu())
    states = encoded_states(model=model, texts=TEXTS)
    with torch.no_grad():
        assert torch.equal(torch.from_numpy(states.mu), model.posterior(texts=TEXTS).mu)  # same arithmetic as the posterior
    assert abs(states.on - states.off).max() > 0

    run_encoder_in_float32(model)
    assert load_checkpoint(model=model, path=checkpoints[0]) == 0
    start = encoded_states(model=model, texts=TEXTS)
    assert start.on.dtype.name == "float32" and (start.on == start.off).all()  # every LoRA B is 0 at step 0
