"""The CPU part of the trajectory analysis: scores and maps from the cached states.

What belongs here
-----------------
- One cached row of scores per checkpoint and float format, and ``results.json``.
- The map coordinates of the float32 states: PCA in a fixed basis, PCA refit and aligned
  to the final step, t-SNE, and the training sample in the fixed basis.

What does NOT belong here
-------------------------
- Encoding (``trajectory_encode.py``), the definitions of the scores (``core/analysis``),
  and the drawing (``trajectory_figures.py``).

Nothing here loads a model, so all of it runs without a GPU. A row already in the cache is
reused; delete it to compute it again. A map is reused only when it records the md5s of the
checkpoints its states came from, so a later retrain cannot draw figures from old weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from core.analysis.representation import TextIndex
from core.analysis.trajectory import (
    checkpoint_row,
    compare_row_to_reference,
    effective_perplexity,
    is_constant,
    nearest_saved_steps,
    pca_basis,
    procrustes_to,
    project,
    state_vectors,
    training_row,
    tsne_map,
    variance_in_plane,
)
from core.analysis.topic_score import majority_count
from core.runs.io import read_json
from experiments.factorization.config import TrajectoryConfig
from experiments.factorization.output import LABEL_DELTA, LABEL_H, LABEL_STATE_DELTA, LABEL_STATE_H
from experiments.factorization.trajectory_cache import MODELS, PRECISIONS, TrajectoryPaths, write_json_whole, write_npz

GRIDS: tuple[tuple[str, str], ...] = (("delta", "delta"), ("delta", "mu"), ("h", "h_on"), ("h", "mu"))
REFERENCE_NAMES = {"h": (LABEL_H, LABEL_STATE_H), "delta": (LABEL_DELTA, LABEL_STATE_DELTA)}


@dataclass(frozen=True)
class Labels:
    """Who each row of the cached arrays is. See ``trajectory_encode.py``."""

    index: TextIndex
    langs: list[str]
    target: np.ndarray
    train_target: np.ndarray
    meta: dict[str, Any]


def load_labels(paths: TrajectoryPaths) -> Labels:
    """The labels from the cache; a clear error when nothing has been encoded yet."""

    if not paths.labels_npz.exists():
        raise FileNotFoundError(f"{paths.labels_npz} does not exist; run this without --figures-only first, on a GPU.")
    meta = read_json(paths.labels_json)
    with np.load(paths.labels_npz) as saved:
        target, train_target = saved["target"], saved["train_target"]
    return Labels(
        index=TextIndex(num_langs=len(meta["langs"]), num_docs=meta["num_docs"]),
        langs=meta["langs"],
        target=target,
        train_target=train_target,
        meta=meta,
    )


def _check_same_sample(*, labels: Labels, config: TrajectoryConfig) -> None:
    """The cached training states belong to one sample, so another seed cannot share this cache."""

    if labels.meta["train_sample_seed"] != config.train_sample_seed:
        raise ValueError(
            f"The cache holds the training sample of seed {labels.meta['train_sample_seed']}, this config asks for "
            f"seed {config.train_sample_seed}. Use another output_dir, or delete its features/ folder."
        )


def _row(*, paths: TrajectoryPaths, labels: Labels, model: str, precision: str, step: int, read: str) -> dict[str, Any]:
    path = paths.row(model, precision, step)
    if path.exists():
        return read_json(path)
    with np.load(paths.states(model, precision, step)) as states:
        row = checkpoint_row(on=states["h_on"], off=states["h_off"], mu=states["mu"], target=labels.target, index=labels.index, langs=labels.langs, read=read)
        row["train_english"] = training_row(mu=states["train_mu"], target=labels.train_target)
    write_json_whole(path, row)
    print(f"{model} {precision} step {step}: topic {row['topic']}, probe {row['language_probe']}, retrieval {row['retrieval']}", flush=True)
    return row


def _saved_rows(*, config: TrajectoryConfig, paths: TrajectoryPaths, labels: Labels) -> dict[str, Any]:
    """The saved weights through the same code, against the reference file."""

    reference = read_json(config.reference_results) if config.reference_results else None
    checked: dict[str, Any] = {
        "partial": config.is_partial,
        "saved_runs_dir": str(config.saved_runs_dir),
        "note": "minus_reference compares each float format with the shipped file, which is bfloat16, so only the bfloat16 entries test reproduction",
        "models": {},
    }
    for model in MODELS:
        checked["models"][model] = {}
        for precision in PRECISIONS:
            with np.load(paths.saved_states(model, precision)) as states:
                row = checkpoint_row(on=states["h_on"], off=states["h_off"], mu=states["mu"], target=labels.target, index=labels.index, langs=labels.langs, read="h_on" if model == "h" else "delta")
            entry = {"row": row}
            if reference is not None:
                predictor, state = REFERENCE_NAMES[model]
                entry["minus_reference"] = compare_row_to_reference(
                    row=row,
                    reference_topic=reference["topic"][predictor],
                    reference_language=reference["language"][predictor],
                    reference_inside=reference["inside"][state],
                    langs=labels.langs,
                )
            checked["models"][model][precision] = entry
    write_json_whole(paths.saved_check, checked)
    print(f"wrote {paths.saved_check}", flush=True)
    return checked


def write_saved_check(*, config: TrajectoryConfig, paths: TrajectoryPaths) -> dict[str, Any]:
    """``saved_weights_check.json`` alone."""

    return _saved_rows(config=config, paths=paths, labels=load_labels(paths))


def _map_source(*, paths: TrajectoryPaths, model: str, steps: list[int]) -> str:
    """Which checkpoints a map comes from, as text to store in it and compare."""

    md5 = read_json(paths.run_json(model)).get("checkpoint_md5", {})
    return json.dumps({str(step): md5.get(str(step)) for step in steps}, sort_keys=True)


def _maps_for(*, paths: TrajectoryPaths, config: TrajectoryConfig, labels: Labels, model: str, vector: str, steps: list[int], final_step: int) -> None:
    """All three kinds of coordinates of one vector of one model. A blank panel is NaN."""

    path = paths.maps(model, vector)
    source = _map_source(paths=paths, model=model, steps=steps)
    if path.exists():
        with np.load(path) as saved:
            if saved["steps"].tolist() == steps and "source" in saved.files and str(saved["source"]) == source:
                return
    num_points = labels.index.num_langs * labels.index.num_docs
    perplexity = effective_perplexity(num_points=num_points, requested=config.tsne_perplexity)

    def vectors_of(step: int) -> tuple[np.ndarray, np.ndarray]:
        with np.load(paths.states(model, "float32", step)) as states:
            test = state_vectors(on=states["h_on"], off=states["h_off"], mu=states["mu"])[vector]
            train = state_vectors(on=states["train_h_on"], off=states["train_h_off"], mu=states["train_mu"])[vector]
        return test, train

    final_test, _ = vectors_of(final_step)
    mean, components = pca_basis(final_test)
    final_refit = project(final_test, mean=mean, components=components)
    shape = (len(steps), num_points, 2)
    fixed, aligned, tsne = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    train_fixed = np.full((len(steps), len(labels.train_target), 2), np.nan)
    disparity, plane_fixed, plane_own = np.full(len(steps), np.nan), np.full(len(steps), np.nan), np.full(len(steps), np.nan)
    blank_at_zero = vector == "delta" or (model, vector) == ("delta", "mu")  # step 0 has no signal there, whatever the rounding
    for i, step in enumerate(steps):
        test, train = vectors_of(step)
        if is_constant(test) or (step == 0 and blank_at_zero):
            continue
        fixed[i] = project(test, mean=mean, components=components)
        train_fixed[i] = project(train, mean=mean, components=components)
        step_mean, step_components = pca_basis(test)
        aligned[i], disparity[i] = procrustes_to(project(test, mean=step_mean, components=step_components), reference=final_refit)
        plane_fixed[i] = variance_in_plane(test, mean=mean, components=components)
        plane_own[i] = variance_in_plane(test, mean=step_mean, components=step_components)
        tsne[i] = tsne_map(features=test, perplexity=perplexity, seed=config.tsne_seed, pca_dims=config.pca_dims)
        print(f"map {model} {vector} step {step}: done, Procrustes disparity {disparity[i]:.3f}", flush=True)
    write_npz(
        path,
        steps=np.array(steps),
        fixed=fixed,
        aligned=aligned,
        tsne=tsne,
        train_fixed=train_fixed,
        disparity=disparity,
        plane_fixed=plane_fixed,
        plane_own=plane_own,
        source=np.array(source),
    )


def _map_h_off(*, paths: TrajectoryPaths, config: TrajectoryConfig, labels: Labels, step: int) -> None:
    """Plain Qwen once: it does not change with the checkpoints."""

    if paths.maps_h_off.exists():
        with np.load(paths.maps_h_off) as saved:
            if "plane" in saved.files:
                return
    with np.load(paths.states("delta", "float32", step)) as states:
        off = states["h_off"]
    mean, components = pca_basis(off)
    perplexity = effective_perplexity(num_points=len(off), requested=config.tsne_perplexity)
    write_npz(
        paths.maps_h_off,
        fixed=project(off, mean=mean, components=components),
        plane=np.array(variance_in_plane(off, mean=mean, components=components)),
        tsne=tsne_map(features=off, perplexity=perplexity, seed=config.tsne_seed, pca_dims=config.pca_dims),
    )


def build_maps(*, config: TrajectoryConfig, paths: TrajectoryPaths) -> None:
    """Every map in the cache."""

    labels = load_labels(paths)
    _check_same_sample(labels=labels, config=config)
    for model, vector in GRIDS:
        run = read_json(paths.run_json(model))
        steps = nearest_saved_steps(wanted=config.map_steps, saved=run["steps"])
        _maps_for(paths=paths, config=config, labels=labels, model=model, vector=vector, steps=steps, final_step=run["final_step"])
    _map_h_off(paths=paths, config=config, labels=labels, step=read_json(paths.run_json("delta"))["steps"][0])


def build_results(*, config: TrajectoryConfig, paths: TrajectoryPaths) -> dict[str, Any]:
    """``results.json``: every number, from the cached states."""

    labels = load_labels(paths)
    _check_same_sample(labels=labels, config=config)
    reference = read_json(config.reference_results) if config.reference_results else None
    models: dict[str, Any] = {}
    for model in MODELS:
        run = read_json(paths.run_json(model))
        rows = {
            precision: {str(step): _row(paths=paths, labels=labels, model=model, precision=precision, step=step, read=run["read"]) for step in run["steps"]}
            for precision in PRECISIONS
        }
        entry = {key: run[key] for key in ("steps", "final_step", "kept_step", "kept_epoch", "read", "model_pt_equals_kept_checkpoint")}
        entry["rows"] = rows
        entry["weights"] = read_json(paths.weights_json(model))
        kept = str(run["kept_step"])
        if reference is not None and kept in rows["bfloat16"]:
            predictor, state = REFERENCE_NAMES[model]
            entry["kept_minus_reference"] = compare_row_to_reference(
                row=rows["bfloat16"][kept],
                reference_topic=reference["topic"][predictor],
                reference_language=reference["language"][predictor],
                reference_inside=reference["inside"][state],
                langs=labels.langs,
            )
        models[model] = entry
    num_points = labels.index.num_langs * labels.index.num_docs
    results = {
        "documents": {
            "test": f"{labels.index.num_docs:,} test documents x {labels.index.num_langs} languages",
            "train_english": f"{labels.meta['train_sample_size']:,} English posts from the {labels.meta['training_split_size']:,} trained on, seed {labels.meta['train_sample_seed']}",
            **labels.meta,
        },
        "float_formats": {
            "float32": "the same weights with the posterior backbone in float32; every figure",
            "bfloat16": "as trained; at small Delta h this is mostly rounding",
        },
        "partial": config.is_partial,
        "majority": majority_count(target=labels.target),
        "tsne": {
            "perplexity": effective_perplexity(num_points=num_points, requested=config.tsne_perplexity),
            "seed": config.tsne_seed,
            "pca_dims": config.pca_dims,
            "init": "pca",
        },
        "map_steps_wanted": list(config.map_steps),
        "models": models,
    }
    if config.saved_runs_dir is not None and all(paths.saved_states(model, precision).exists() for model in MODELS for precision in PRECISIONS):
        results["saved_weights"] = _saved_rows(config=config, paths=paths, labels=labels)
    for precision in PRECISIONS:
        first = {model: read_json(paths.run_json(model))["steps"][0] for model in MODELS}
        with np.load(paths.states("h", precision, first["h"])) as states_h, np.load(paths.states("delta", precision, first["delta"])) as states_delta:
            gap = float(np.abs(states_h["h_off"] - states_delta["h_off"]).max())
        results.setdefault("h_off_max_gap_between_models", {})[precision] = gap
    write_json_whole(paths.results, results)
    print(f"wrote {paths.results}", flush=True)
    return results
