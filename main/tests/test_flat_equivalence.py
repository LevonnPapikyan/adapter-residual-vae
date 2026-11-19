"""The package against the flat code: same numbers, bit for bit.

The flat code is no longer next to ``main/``, so these tests skip unless ``VAE_FLAT_ROOT``
names a copy of its ``model.py``, ``data.py``, ``train.py`` and ``inference.py`` with a
``data`` folder or link beside them. A copy of the last revision is needed: the earlier
one in ``archive-2026-10-30`` still has the LayerNorm and a different ``Config``.

The flat code is read, never changed, and no bytecode is written next to it. Its
``train`` function is cut out of ``train.py`` by name, since that file runs top to
bottom as a notebook when imported.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
import pytest
import torch

from conftest import DATA_DIR, FINAL_DELTA, FINAL_H, FLAT_ROOT, MAIN_ROOT, needs_data, needs_qwen, needs_weights
from core.factorization.checkpointing import load_trained_run, module_name_to_saved_name, recipe_to_saved_config
from experiments.factorization.config import train_config_from_file
from experiments.factorization.train import train_run

TEXTS = [
    "The goalie made thirty saves and the team won the hockey game in overtime.",
    "Der Torwart hielt dreissig Schuesse und die Mannschaft gewann das Eishockeyspiel.",
    "I need a new SCSI driver for my disk controller on this PC.",
]
needs_flat_code = pytest.mark.skipif(not (FLAT_ROOT / "model.py").exists(), reason="no flat code: set VAE_FLAT_ROOT to a copy of it")


def _import_flat(name: str) -> ModuleType:
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(FLAT_ROOT / f"{name}.py"):
        return sys.modules[name]
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, FLAT_ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _flat_train_namespace(*, num_texts: int, num_docs: int, device: str) -> dict:
    """The flat ``train``, ``test_scores`` and ``held_out_loss``, with the data cut down the way the options do."""

    data, model = _import_flat("data"), _import_flat("model")
    full_docs = data.load_translations()
    namespace = {
        "F": torch.nn.functional,
        "LDA": data.LDA,
        "Path": Path,
        "TopicVAE": model.TopicVAE,
        "json": json,
        "load": model.load,
        "load_train": lambda: data.load_corpus()[0][:num_texts],
        "load_translations": lambda: {lang: texts[:num_docs] for lang, texts in full_docs.items()},
        "math": math,
        "np": np,
        "save": model.save,
        "time": time,
        "torch": torch,
        "DEVICE": device,
    }
    tree = ast.parse((FLAT_ROOT / "train.py").read_text(encoding="utf-8"))
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"test_scores", "held_out_loss", "train"}]
    assert len(wanted) == 3
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "flat train.py", "exec"), namespace)
    return namespace


@needs_flat_code
@needs_weights
@needs_qwen
@pytest.mark.needs_model
@pytest.mark.parametrize("run_dir", [FINAL_DELTA, FINAL_H], ids=["delta", "h"])
def test_posterior_is_bit_identical_on_saved_weights(run_dir: Path) -> None:
    flat_model, _ = _import_flat("model").load(run_dir, "cpu")
    new_run = load_trained_run(run_dir=run_dir, device="cpu")
    with torch.no_grad():
        flat_mu, flat_logvar = flat_model.encode(TEXTS)
        posterior = new_run.model.posterior(texts=TEXTS)
        flat_reconstruction = flat_model.nll(flat_mu, TEXTS)
        reconstruction = new_run.model.reconstruction_nll(z=posterior.mu, texts=TEXTS)
    assert torch.equal(flat_mu, posterior.mu)
    assert torch.equal(flat_logvar, posterior.logvar)
    assert torch.equal(flat_reconstruction[0], reconstruction.nll)
    assert torch.equal(flat_reconstruction[1], reconstruction.num_tokens)


@needs_flat_code
@needs_data
@needs_qwen
@pytest.mark.needs_model
@pytest.mark.parametrize("mu_from", ["delta", "h"])
def test_training_is_bit_identical_to_the_flat_recipe(mu_from: str, tmp_path: Path) -> None:
    """Same model construction, split, batches, noise, Adam groups, schedule, clip, epoch choice, and files."""

    num_texts, num_docs = 14, 2
    config = train_config_from_file(
        MAIN_ROOT / "configs" / "small_cpu_example" / f"factorization_{mu_from}.json",
        data_dir=str(DATA_DIR),
        output_dir=str(tmp_path / "new"),
        device="cpu",
        max_train_texts=num_texts,
        max_test_docs=num_docs,
    )
    model_config = replace(config.recipe.model, encoder_max_tokens=12, decoder_max_tokens=8)
    config = replace(config, recipe=replace(config.recipe, model=model_config, batch_size=4, num_epochs=2))
    train_run(config=config)

    flat_config = _import_flat("model").Config(**recipe_to_saved_config(config.recipe))
    namespace = _flat_train_namespace(num_texts=num_texts, num_docs=num_docs, device="cpu")
    namespace["train"](flat_config, tmp_path / "flat", device="cpu")

    new_saved = torch.load(tmp_path / "new" / "model.pt", map_location="cpu")
    flat_saved = torch.load(tmp_path / "flat" / "model.pt", map_location="cpu")
    assert new_saved["config"] == flat_saved["config"]
    assert list(new_saved["config"]) == list(flat_saved["config"])
    assert list(new_saved["weights"]) == list(flat_saved["weights"])
    for name, tensor in flat_saved["weights"].items():
        assert torch.equal(new_saved["weights"][name], tensor), name
    for file_name in ("history.json", "test.json"):
        assert (tmp_path / "new" / file_name).read_text() == (tmp_path / "flat" / file_name).read_text(), file_name

    # And the flat loader reads a file the package wrote.
    flat_model, _ = _import_flat("model").load(tmp_path / "new", "cpu")
    assert flat_model.cfg == flat_config
    assert module_name_to_saved_name("z_to_prefix.weight") == "prefix.weight"


def _flat_inference_functions(*, text_lang: np.ndarray, text_doc: np.ndarray, num_docs: int) -> dict:
    """The score functions of the flat ``inference.py``, with the globals they read."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    namespace = {
        "F": torch.nn.functional,
        "GroupKFold": GroupKFold,
        "LANGS": ["en", "de", "fr", "es", "zh"],
        "LogisticRegression": LogisticRegression,
        "N": num_docs,
        "StandardScaler": StandardScaler,
        "DEVICE": "cpu",
        "cross_val_score": cross_val_score,
        "make_pipeline": make_pipeline,
        "np": np,
        "text_doc": text_doc,
        "text_lang": text_lang,
        "torch": torch,
    }
    names = {"fit_linear_map", "topic_score", "language_score", "finds_original", "variance_split"}
    tree = ast.parse((FLAT_ROOT / "inference.py").read_text(encoding="utf-8"))
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(wanted) == len(names)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "flat inference.py", "exec"), namespace)
    return namespace


@needs_flat_code
def test_scores_match_the_flat_definitions() -> None:
    from core.analysis.language_probe import language_score
    from core.analysis.linear_map import fit_linear_map
    from core.analysis.representation import TextIndex, finds_original, variance_split
    from core.analysis.topic_score import topic_score

    index = TextIndex(num_langs=5, num_docs=30)
    flat = _flat_inference_functions(text_lang=index.text_lang, text_doc=index.text_doc, num_docs=30)
    rng = np.random.default_rng(0)
    documents = rng.normal(size=(30, 8))
    features = np.concatenate([documents + rng.normal(size=8) + 0.3 * rng.normal(size=(30, 8)) for _ in range(5)]).astype(np.float32)
    for lang_index in range(1, 5):
        assert flat["finds_original"](features, lang_index) == finds_original(features=features, lang_index=lang_index, index=index)
    split = variance_split(features=features, index=index)
    assert flat["variance_split"](features) == (split.language, split.document)

    theta = rng.dirichlet(np.ones(6), size=150).astype(np.float32)
    torch.manual_seed(3)
    flat_map = flat["fit_linear_map"](features, theta, steps=20)
    torch.manual_seed(3)
    new_map = fit_linear_map(features=features, theta=theta, device="cpu", steps=20)
    assert np.array_equal(flat_map(features[:40]), new_map.predict(features=features[:40]))

    target = theta[:30]
    prediction = new_map.predict(features=features[:30])
    assert flat["topic_score"](prediction, target) == topic_score(theta_pred=prediction, target=target)
    thetas = {str(j): new_map.predict(features=features[index.text_lang == j]) for j in range(5)}
    assert flat["language_score"](thetas) == language_score(thetas=thetas)
