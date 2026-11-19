"""Loading the saved weights and taking a few training steps on one CPU thread. Takes minutes."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from conftest import DATA_DIR, FINAL_DELTA, MAIN_ROOT, needs_data, needs_qwen, needs_weights
from core.analysis.encoding import topic_mixtures
from core.factorization.checkpointing import load_trained_run
from experiments.factorization.config import train_config_from_file
from experiments.factorization.train import train_or_load, train_run

TEXTS = [
    "The goalie made thirty saves and the team won the hockey game in overtime.",
    "Der Torwart hielt dreissig Schuesse und die Mannschaft gewann das Eishockeyspiel.",
    "I need a new SCSI driver for my disk controller on this PC.",
]


@pytest.fixture(scope="module")
def delta_run():
    return load_trained_run(run_dir=FINAL_DELTA, device="cpu")


@needs_weights
@needs_qwen
@pytest.mark.needs_model
def test_load_final_delta_and_encode_three_texts(delta_run) -> None:
    model = delta_run.model
    assert model.config.mu_from == "delta" and model.config.num_topics == 20
    assert not model.training
    with torch.no_grad():
        posterior = model.posterior(texts=TEXTS)
    assert posterior.mu.shape == (3, 20) and posterior.logvar.shape == (3, 20)
    assert posterior.mu.dtype == torch.float32
    assert torch.isfinite(posterior.mu).all() and (posterior.logvar >= -8).all() and (posterior.logvar <= 4).all()
    mixtures = topic_mixtures(model=model, texts=TEXTS)
    assert mixtures.shape == (3, 20) and abs(float(mixtures.sum(1).mean()) - 1.0) < 1e-5
    assert len(delta_run.history) > 0 and set(delta_run.history[0]) == {"step", "rec", "align", "rate"}


@needs_weights
@needs_qwen
@pytest.mark.needs_model
def test_generator_writes_from_z(delta_run) -> None:
    written = delta_run.model.write_from_z(z=torch.zeros(1, 20), max_new_tokens=4)
    assert len(written) == 1 and isinstance(written[0], str)


def _small_config(*, run_dir: Path, mu_from: str = "delta", **overrides):
    config = train_config_from_file(
        MAIN_ROOT / "configs" / "small_cpu_example" / f"factorization_{mu_from}.json",
        data_dir=str(DATA_DIR),
        output_dir=str(run_dir),
        device="cpu",
        max_train_texts=12,
        max_test_docs=2,
        **overrides,
    )
    model = replace(config.recipe.model, encoder_max_tokens=16, decoder_max_tokens=12)
    return replace(config, recipe=replace(config.recipe, model=model, batch_size=4))


@needs_data
@needs_qwen
@pytest.mark.needs_model
def test_small_training_run_writes_the_three_files_and_refuses_a_second(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = _small_config(run_dir=run_dir, max_steps=2)
    output = train_run(config=config)
    assert output.num_steps == 2 and len(output.history) == 2
    assert sorted(path.name for path in run_dir.iterdir()) == ["history.json", "model.pt", "test.json"]
    rows = json.loads((run_dir / "test.json").read_text())
    assert [row["epoch"] for row in rows] == [0, 1]
    assert list(rows[0]) == ["epoch", "step", "val", "rec", "align", "en", "de"]

    saved = torch.load(run_dir / "model.pt", map_location="cpu")
    assert all(name.split(".")[0] in {"enc", "dec", "mu_head", "logvar_head", "prefix"} for name in saved["weights"])
    assert saved["config"]["enc_len"] == 16 and saved["config"]["mu_from"] == "delta"

    loaded = load_trained_run(run_dir=run_dir, device="cpu")
    assert loaded.recipe == config.recipe
    for name, parameter in output.model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter.detach().cpu(), dict(loaded.model.named_parameters())[name].detach().cpu())

    with pytest.raises(FileExistsError, match="never overwritten"):
        train_run(config=config)
    _, history = train_or_load(config=config)
    assert history == output.history
    with pytest.raises(ValueError, match="pick another output_dir"):
        train_or_load(config=replace(config, recipe=replace(config.recipe, seed=1)))
