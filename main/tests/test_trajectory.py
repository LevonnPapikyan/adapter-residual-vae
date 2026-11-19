"""The trajectory analysis on planted states: scores, maps and figures, with no language model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import MAIN_ROOT
from core.analysis.representation import TextIndex
from core.analysis.trajectory import (
    checkpoint_row,
    effective_perplexity,
    lora_distance,
    nearest_saved_steps,
    pca_basis,
    procrustes_to,
    project,
    training_sample_indices,
    variance_in_plane,
    vector_row,
    weights_row,
)
from core.runs.io import read_json
from experiments.factorization.config import TrajectoryConfig, parse_trajectory_config
from experiments.factorization.trajectory import run_trajectory
from experiments.factorization.trajectory_cache import MODELS, PRECISIONS, TrajectoryPaths, write_json_whole, write_npz
from experiments.factorization.trajectory_figures import topic_label_places
from experiments.factorization.trajectory_results import build_maps, build_results

LANGS = ["en", "de", "fr", "es", "zh"]
NUM_DOCS, HIDDEN, TOPICS, TRAIN = 12, 6, 4, 5


def test_vector_row_of_a_constant_vector_has_no_numbers() -> None:
    index = TextIndex(num_langs=5, num_docs=NUM_DOCS)
    row = vector_row(features=np.zeros((60, 3), dtype=np.float32), index=index, langs=LANGS)
    assert row == {"defined": False, "value_max": 0.0}


def test_absolute_variance_is_the_share_times_the_total() -> None:
    index = TextIndex(num_langs=2, num_docs=3)
    docs = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
    features = np.concatenate([docs, docs + 0.1], dtype=np.float32)
    row = vector_row(features=features, index=index, langs=["en", "de"])
    total = ((features - features.mean(0)) ** 2).sum(1).mean()
    assert row["variance_total"] == pytest.approx(total)
    assert row["variance_document"] == pytest.approx(row["document_share"] * total)
    assert row["finds_original"] == {"de": 3}


def test_step_zero_of_the_delta_model_gives_none_not_noise() -> None:
    index = TextIndex(num_langs=5, num_docs=NUM_DOCS)
    rng = np.random.default_rng(0)
    off = rng.normal(size=(60, HIDDEN)).astype(np.float32)
    target = rng.dirichlet(np.ones(TOPICS), size=NUM_DOCS)
    mu = np.tile(np.array([0.1, 0.5, -0.2, 0.0], dtype=np.float32), (60, 1))
    row = checkpoint_row(on=off.copy(), off=off, mu=mu, target=target, index=index, langs=LANGS, read="delta")
    assert row["language_probe"] is None and row["retrieval"] is None
    assert row["delta_ratio"] == 0.0 and row["delta_max"] == 0.0 and row["delta_zero_fraction"] == 1.0
    assert row["vectors"]["delta"] == {"defined": False, "value_max": 0.0}
    assert row["topic"]["en"] == int((target.argmax(1) == 1).sum())  # one point: everyone gets its top topic


def test_lora_distance_matches_forming_the_products() -> None:
    torch.manual_seed(0)
    name = "enc.base_model.model.layers.3.self_attn.q_proj.lora_{}.default.weight"
    make = lambda: {name.format("A"): torch.randn(2, 5), name.format("B"): torch.randn(4, 2)}
    now, final = make(), make()
    product = lambda w: (w[name.format("B")] @ w[name.format("A")]).double()
    row = lora_distance(weights=now, final=final, part="enc")
    assert row["norm"] == pytest.approx(product(now).norm().item())
    assert row["distance"] == pytest.approx((product(now) - product(final)).norm().item())
    assert row["relative"] == pytest.approx(row["distance"] / product(final).norm().item())
    assert lora_distance(weights=final, final=final, part="enc")["distance"] == pytest.approx(0, abs=1e-6)


def test_procrustes_recovers_a_rotation_and_a_reflection() -> None:
    reference = np.random.default_rng(0).normal(size=(50, 2))
    reference -= reference.mean(0)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.diag([1.0, -1.0])
    aligned, disparity = procrustes_to(reference @ rotation, reference=reference)
    assert np.allclose(aligned, reference) and disparity == pytest.approx(0, abs=1e-9)


def test_pca_basis_finds_the_planted_direction() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 1)) * np.array([[3.0, 4.0, 0.0]]) / 5 + 0.01 * rng.normal(size=(200, 3))
    mean, components = pca_basis(features)
    assert abs(components[0] @ np.array([0.6, 0.8, 0.0])) > 0.99
    assert project(features, mean=mean, components=components, dims=2).shape == (200, 2)


def test_variance_in_plane_is_a_share_and_the_own_basis_holds_the_most() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 6)) * np.array([5.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    mean, components = pca_basis(features)
    own = variance_in_plane(features, mean=mean, components=components)
    other_mean, other_components = pca_basis(rng.normal(size=(300, 6)))
    other = variance_in_plane(features, mean=other_mean, components=other_components)
    assert 0 < other < own <= 1
    singular = np.linalg.svd(features - features.mean(0), compute_uv=False) ** 2
    assert own == pytest.approx(singular[:2].sum() / singular.sum())


def test_document_share_has_a_floor_near_one_over_the_languages() -> None:
    from core.analysis.representation import variance_split

    index = TextIndex(num_langs=5, num_docs=400)
    split = variance_split(features=np.random.default_rng(0).normal(size=(2000, 50)), index=index)
    assert split.document == pytest.approx(0.2, abs=0.01) and split.language < 0.01


def test_a_topic_split_into_two_blobs_gets_two_labels() -> None:
    rng = np.random.default_rng(0)
    blobs = np.concatenate([rng.normal(size=(50, 2)) * 0.1 + 10, rng.normal(size=(50, 2)) * 0.1 - 10])
    assert len(topic_label_places(blobs, all_points=blobs, spread=10.0)) == 2
    spread_out = rng.normal(size=(100, 2)) * 10
    assert topic_label_places(spread_out, all_points=spread_out, spread=1.0) == []
    crowd = np.concatenate([blobs, rng.normal(size=(1000, 2)) * 0.1 + 10, rng.normal(size=(1000, 2)) * 0.1 - 10])
    assert topic_label_places(blobs, all_points=crowd, spread=10.0) == []  # a topic inside a mixed cloud gets no number


def test_nearest_saved_steps_uses_saved_steps_only() -> None:
    saved = [0, 10, 20, 30, 40, 50, 100, 825]
    assert nearest_saved_steps(wanted=[0, 10, 25, 50, 800, 2200], saved=saved) == [0, 10, 20, 50, 825]


def test_training_sample_is_fixed_and_within_the_split() -> None:
    first = training_sample_indices(num_texts=100, size=10, seed=0)
    assert np.array_equal(first, training_sample_indices(num_texts=100, size=10, seed=0))
    assert len(set(first)) == 10 and first.max() < 100 and np.array_equal(first, np.sort(first))
    assert not np.array_equal(first, training_sample_indices(num_texts=100, size=10, seed=1))
    assert len(training_sample_indices(num_texts=4, size=10, seed=0)) == 4


def test_perplexity_is_stated_as_asked_unless_there_are_too_few_points() -> None:
    assert effective_perplexity(num_points=5000, requested=30.0) == 30.0
    assert effective_perplexity(num_points=60, requested=30.0) == pytest.approx(59 / 3)


def test_trajectory_configs_parse(tmp_path: Path) -> None:
    config = parse_trajectory_config(["--config", str(MAIN_ROOT / "configs" / "paper" / "trajectory.json"), "--figures-only"])
    assert config.figures_only and not config.is_partial and config.tsne_perplexity == 30.0
    assert config.output_dir == Path("outputs/checkpointed")
    small = parse_trajectory_config(["--config", str(MAIN_ROOT / "configs" / "small_cpu_example" / "trajectory.json")])
    assert small.is_partial and small.max_docs == 12
    with pytest.raises(ValueError, match="read-only"):
        TrajectoryConfig(runs_dir=tmp_path, data_dir=tmp_path, output_dir=MAIN_ROOT.parent / "figures")
    with pytest.raises(ValueError, match="saved_only"):
        TrajectoryConfig(runs_dir=tmp_path, data_dir=tmp_path, output_dir=tmp_path, saved_only=True)


def _weights(rng: np.random.Generator) -> dict[str, torch.Tensor]:
    weights = {"mu_head.weight": torch.from_numpy(rng.normal(size=(TOPICS, HIDDEN)))}
    for part in ("enc", "dec"):
        for name, shape in (("lora_A", (2, HIDDEN)), ("lora_B", (HIDDEN, 2))):
            weights[f"{part}.base_model.layers.0.self_attn.q_proj.{name}.default.weight"] = torch.from_numpy(rng.normal(size=shape))
    return weights


def _plant_cache(paths: TrajectoryPaths) -> None:
    """What ``trajectory_encode.py`` leaves behind, made of planted numbers."""

    rng = np.random.default_rng(0)
    target = rng.dirichlet(np.ones(TOPICS), size=NUM_DOCS)
    train_target = rng.dirichlet(np.ones(TOPICS), size=TRAIN)
    write_npz(
        paths.labels_npz,
        text_lang=np.repeat(np.arange(5), NUM_DOCS),
        text_doc=np.tile(np.arange(NUM_DOCS), 5),
        topic=np.tile(target.argmax(1), 5),
        target=target,
        train_target=train_target,
        train_indices=np.arange(TRAIN),
    )
    write_json_whole(
        paths.labels_json,
        {
            "langs": LANGS,
            "lang_names": ["English", "German", "French", "Spanish", "Chinese"],
            "num_docs": NUM_DOCS,
            "topic_names": [f"{k}: word word" for k in range(TOPICS)],
            "training_split_size": 40,
            "train_sample_size": TRAIN,
            "train_sample_seed": 0,
            "train_sample_also_held_out_text": 0,
            "median_words_test_english": 150.0,
            "median_words_train_sample": 90.0,
            "encoder_max_tokens": 512,
        },
    )
    steps = [0, 2, 4]
    off = rng.normal(size=(60, HIDDEN)).astype(np.float32)
    train_off = rng.normal(size=(TRAIN, HIDDEN)).astype(np.float32)
    head = rng.normal(size=(HIDDEN, TOPICS)).astype(np.float32)
    for model in MODELS:
        write_json_whole(
            paths.run_json(model),
            {"steps": steps, "final_step": 4, "kept_step": 2, "kept_epoch": 1, "read": "h_on" if model == "h" else "delta", "model_pt_equals_kept_checkpoint": True},
        )
        final = _weights(rng)
        write_json_whole(paths.weights_json(model), {str(step): weights_row(weights=_weights(rng), final=final) for step in steps})
        for precision in PRECISIONS:
            for step in steps:
                scale = step / 4
                delta = (scale * rng.normal(size=off.shape)).astype(np.float32)
                train_delta = (scale * rng.normal(size=train_off.shape)).astype(np.float32)
                on, train_on = off + delta, train_off + train_delta
                read, train_read = (on, train_on) if model == "h" else (delta, train_delta)
                write_npz(
                    paths.states(model, precision, step),
                    h_on=on,
                    h_off=off,
                    mu=read @ head,
                    train_h_on=train_on,
                    train_h_off=train_off,
                    train_mu=train_read @ head,
                )


def test_results_and_figures_come_from_the_cache_alone(tmp_path: Path) -> None:
    config = TrajectoryConfig(runs_dir=tmp_path / "none", data_dir=tmp_path / "none", output_dir=tmp_path, map_steps=(0, 2, 4), figures_only=True)
    paths = TrajectoryPaths(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="figures-only"):
        run_trajectory(config=config)
    _plant_cache(paths)
    run_trajectory(config=config)

    results = read_json(paths.results)
    assert results["tsne"]["perplexity"] == pytest.approx(59 / 3) and results["majority"] > 0
    delta = results["models"]["delta"]
    assert set(delta["rows"]) == {"float32", "bfloat16"}
    assert delta["rows"]["float32"]["0"]["language_probe"] is None
    assert delta["rows"]["float32"]["4"]["language_probe"] is not None
    assert delta["rows"]["float32"]["4"]["train_english"]["total"] == TRAIN
    assert results["h_off_max_gap_between_models"] == {"float32": 0.0, "bfloat16": 0.0}

    with np.load(paths.maps("delta", "delta")) as maps:
        assert np.isnan(maps["fixed"][0]).all() and np.isfinite(maps["fixed"][1]).all()
        assert np.isnan(maps["tsne"][0]).all() and np.isfinite(maps["train_fixed"][1]).all()
        assert maps["disparity"][2] == pytest.approx(0, abs=1e-9)  # the final step against itself
    with np.load(paths.maps("h", "h_on")) as maps:
        assert np.isfinite(maps["fixed"][0]).all()

    with np.load(paths.maps("h", "h_on")) as maps:
        assert maps["plane_fixed"][2] == pytest.approx(maps["plane_own"][2])  # the final step is its own basis
        assert (0 < maps["plane_fixed"]).all() and (maps["plane_fixed"] <= 1).all()

    # A map is tied to the checkpoints behind it: new md5s rebuild it, and the same ones reuse it.
    before = paths.maps("h", "h_on").stat().st_mtime_ns
    build_maps(config=config, paths=paths)
    assert paths.maps("h", "h_on").stat().st_mtime_ns == before
    run = read_json(paths.run_json("h"))
    write_json_whole(paths.run_json("h"), {**run, "checkpoint_md5": {"0": "a", "2": "b", "4": "c"}})
    build_maps(config=config, paths=paths)
    assert paths.maps("h", "h_on").stat().st_mtime_ns != before

    # Another training sample cannot share this cache.
    with pytest.raises(ValueError, match="training sample"):
        build_results(config=replace(config, train_sample_seed=1), paths=paths)

    names = {path.name for path in paths.figures.iterdir()}
    expected = {f"{kind}_{model}_{vector}_{color}" for kind in ("pca_fixed", "pca_aligned", "tsne") for model, vector in (("delta", "delta"), ("delta", "mu"), ("h", "h_on"), ("h", "mu")) for color in ("language", "topic")}
    expected |= {f"train_vs_test_pca_{model}_{vector}" for model, vector in (("delta", "delta"), ("delta", "mu"), ("h", "h_on"), ("h", "mu"))}
    expected |= {"pca_h_off", "tsne_h_off", "lines_topic", "lines_retrieval", "lines_language_probe", "lines_variance_shares", "lines_absolute_variance", "lines_delta_ratio", "lines_distance_to_final", "train_vs_test_topic"}
    assert names == {f"{name}.{suffix}" for name in expected for suffix in ("pdf", "png")}
