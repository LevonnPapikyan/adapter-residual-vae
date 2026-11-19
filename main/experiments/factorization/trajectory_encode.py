"""The GPU part of the trajectory analysis: every checkpoint on the texts, into the cache.

What belongs here
-----------------
- The labels: which documents, the teacher's answers, the fixed sample of training posts.
- Encoding each checkpoint in float32 and in bfloat16, and the saved weights the same way.
- The distances of W and BA to the final checkpoint, read from the checkpoints.

What does NOT belong here
-------------------------
- Scoring (``trajectory_results.py``) and the drawing (``trajectory_figures.py``).

The same training split as the run: the same seed, filter and ``split_held_out``, so the
sample comes only from the posts trained on, never from the held-out ones. Test texts are
batched per language, as the topic score of the evaluation does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.analysis.encoding import encoded_states, run_encoder_in_float32
from core.analysis.trajectory import read_vector_name, training_sample_indices, weights_row
from core.checkpoints.io import load_model_file
from core.factorization.checkpointing import list_checkpoints, load_checkpoint, load_trained_run, read_checkpoint, recipe_from_saved_config
from core.factorization.config import TrainingRecipe
from core.factorization.model import TopicVAE
from core.runs.io import read_json
from core.train.loop import resolve_device
from experiments.data.corpus import load_corpus
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import LANG_NAMES, LANGS, check_alignment, load_translations
from experiments.factorization.config import TrainConfig, TrajectoryConfig
from experiments.factorization.train import load_training_data
from experiments.factorization.trajectory_cache import MODELS, PRECISIONS, TrajectoryPaths, file_md5, write_json_whole, write_npz

SPLIT_SETTINGS = ("teacher", "min_words", "val_frac", "seed")


def _read_recipe(run_dir: Path) -> TrainingRecipe:
    return recipe_from_saved_config(load_model_file(run_dir / "model.pt", map_location="cpu").config)[0]


def _encode_test(*, model: TopicVAE, texts: dict[str, list[str]], batch_size: int) -> dict[str, np.ndarray]:
    states = [encoded_states(model=model, texts=texts[lang], batch_size=batch_size) for lang in LANGS]
    return {
        "h_on": np.concatenate([state.on for state in states]),
        "h_off": np.concatenate([state.off for state in states]),
        "mu": np.concatenate([state.mu for state in states]),
    }


def _write_labels(*, config: TrajectoryConfig, paths: TrajectoryPaths, recipe: TrainingRecipe) -> tuple[dict[str, list[str]], list[str]]:
    """Write who each row is and the teacher's answers. Returns the test texts and the training sample's texts."""

    data_paths = DataPaths(data_dir=config.data_dir)
    check_alignment(translated=load_translations(paths=data_paths), corpus=load_corpus(paths=data_paths))
    train_config = TrainConfig(
        recipe=recipe,
        data_dir=config.data_dir,
        output_dir=config.runs_dir / "not_written",
        max_train_texts=config.max_train_texts,
        max_test_docs=config.max_docs,
    )
    data = load_training_data(config=train_config, rng=np.random.default_rng(recipe.seed))  # the generator seed_run returns
    chosen = training_sample_indices(num_texts=len(data.texts), size=config.train_sample, seed=config.train_sample_seed)
    sample_texts = [data.texts[i] for i in chosen]
    if paths.labels_json.exists():  # a cached state is tied to its checkpoint, but its training rows are tied to this sample too
        cached = read_json(paths.labels_json)
        if (cached["train_sample_seed"], cached["train_sample_size"]) != (config.train_sample_seed, len(chosen)):
            raise ValueError(
                f"{paths.labels_json} holds a training sample of {cached['train_sample_size']} posts with seed {cached['train_sample_seed']}, "
                f"this config asks for {len(chosen)} with seed {config.train_sample_seed}. Use another output_dir, or delete its features/ folder."
            )
    held_out = set(data.val_texts)
    teacher = LDATeacher(paths=data_paths, name=recipe.teacher)
    target = data.theta_test
    num_docs = len(target)
    write_npz(
        paths.labels_npz,
        text_lang=np.repeat(np.arange(len(LANGS)), num_docs),
        text_doc=np.tile(np.arange(num_docs), len(LANGS)),
        topic=np.tile(target.argmax(1), len(LANGS)),
        target=target,
        train_target=data.theta[chosen].numpy(),
        train_indices=chosen,
    )
    write_json_whole(
        paths.labels_json,
        {
            "langs": list(LANGS),
            "lang_names": [LANG_NAMES[lang] for lang in LANGS],
            "num_docs": num_docs,
            "topic_names": [f"{k}: {' '.join(words.split()[:2])}" for k, words in enumerate(teacher.top_words(2))],
            "training_split_size": len(data.texts),
            "train_sample_size": len(chosen),
            "train_sample_seed": config.train_sample_seed,
            "train_sample_also_held_out_text": sum(text in held_out for text in sample_texts),
            "median_words_test_english": float(np.median([len(text.split()) for text in data.test_docs.texts["en"]])),
            "median_words_train_sample": float(np.median([len(text.split()) for text in sample_texts])),
            "encoder_max_tokens": recipe.model.encoder_max_tokens,
        },
    )
    print(f"{num_docs} test documents x {len(LANGS)} languages; {len(chosen)} training posts of the {len(data.texts)} trained on", flush=True)
    return data.test_docs.texts, sample_texts


def _write_run(*, config: TrajectoryConfig, paths: TrajectoryPaths, model: str, recipe: TrainingRecipe, checkpoints: dict[int, Path]) -> None:
    """The run's steps and kept step, and the distances to the final checkpoint."""

    run_dir = config.runs_dir / model
    steps = list(checkpoints)
    final_step = steps[-1]
    kept = min(read_json(run_dir / "test.json")[1:], key=lambda row: row["val"])
    saved = load_model_file(run_dir / "model.pt", map_location="cpu").weights
    same = None
    if kept["step"] in checkpoints:
        _, weights = read_checkpoint(checkpoints[kept["step"]])
        same = weights.keys() == saved.keys() and all(torch.equal(weights[name], saved[name]) for name in weights)
    write_json_whole(
        paths.run_json(model),
        {
            "steps": steps,
            "final_step": final_step,
            "kept_step": kept["step"],
            "kept_epoch": kept["epoch"],
            "read": read_vector_name(recipe.model.mu_from),
            "model_pt_equals_kept_checkpoint": same,
            "checkpoint_md5": {str(step): file_md5(path) for step, path in checkpoints.items()},
        },
    )
    _, final = read_checkpoint(checkpoints[final_step])
    write_json_whole(paths.weights_json(model), {str(step): weights_row(weights=read_checkpoint(path)[1], final=final) for step, path in checkpoints.items()})


def _cached_is_current(*, path: Path, md5: str, num_docs: int, num_train: int) -> bool:
    if not path.exists():
        return False
    with np.load(path) as cached:
        return (
            str(cached["checkpoint_md5"]) == md5
            and len(cached["mu"]) == num_docs * len(LANGS)
            and len(cached["train_mu"]) == num_train
        )


def encode_run(*, config: TrajectoryConfig, paths: TrajectoryPaths, model_name: str, test_texts: dict[str, list[str]], train_texts: list[str], device: str) -> None:
    """Every checkpoint of one run that the cache does not hold yet, float32 first."""

    run_dir = config.runs_dir / model_name
    checkpoints = list_checkpoints(run_dir=run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"{run_dir} holds no checkpoints; train it with checkpoint_steps first.")
    recipe = _read_recipe(run_dir)
    _write_run(config=config, paths=paths, model=model_name, recipe=recipe, checkpoints=checkpoints)
    md5 = {step: file_md5(path) for step, path in checkpoints.items()}
    num_docs = len(test_texts[LANGS[0]])
    for precision in PRECISIONS:
        pending = [
            step
            for step in checkpoints
            if not _cached_is_current(path=paths.states(model_name, precision, step), md5=md5[step], num_docs=num_docs, num_train=len(train_texts))
        ]
        if not pending:
            continue
        model = TopicVAE(config=recipe.model).to(device).eval()
        if precision == "float32":
            run_encoder_in_float32(model)
        for step in pending:
            load_checkpoint(model=model, path=checkpoints[step])
            test = _encode_test(model=model, texts=test_texts, batch_size=config.batch_size)
            train = encoded_states(model=model, texts=train_texts, batch_size=config.batch_size)
            paths.row(model_name, precision, step).unlink(missing_ok=True)  # its scores came from the states replaced here
            write_npz(
                paths.states(model_name, precision, step),
                checkpoint_md5=np.array(md5[step]),
                **test,
                train_h_on=train.on,
                train_h_off=train.off,
                train_mu=train.mu,
            )
            print(f"{model_name} {precision} step {step}: max |delta h| {np.abs(test['h_on'] - test['h_off']).max():.4g}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def encode_saved_runs(*, config: TrajectoryConfig, paths: TrajectoryPaths, test_texts: dict[str, list[str]], device: str) -> None:
    """The saved weights in ``saved_runs_dir``, read only, through the same code and in both float formats."""

    for model_name in MODELS:
        if all(paths.saved_states(model_name, precision).exists() for precision in PRECISIONS):
            continue
        run = load_trained_run(run_dir=config.saved_runs_dir / model_name, device=device, encoder_max_tokens=config.encoder_max_tokens)
        write_npz(paths.saved_states(model_name, "bfloat16"), **_encode_test(model=run.model, texts=test_texts, batch_size=config.batch_size))
        run_encoder_in_float32(run.model)
        write_npz(paths.saved_states(model_name, "float32"), **_encode_test(model=run.model, texts=test_texts, batch_size=config.batch_size))
        del run
        if device == "cuda":
            torch.cuda.empty_cache()


def encode_everything(*, config: TrajectoryConfig, paths: TrajectoryPaths) -> None:
    """Labels, every checkpoint of both runs, and the saved weights if asked for."""

    device = resolve_device(config.device)
    print(f"device {device}, gpu {torch.cuda.get_device_name(0) if device == 'cuda' else None}, torch {torch.__version__}", flush=True)
    if config.saved_only:
        recipe = _read_recipe(config.saved_runs_dir / "h")
    else:
        recipes = {model: _read_recipe(config.runs_dir / model) for model in MODELS}
        if any(getattr(recipes["h"], name) != getattr(recipes["delta"], name) for name in SPLIT_SETTINGS):
            raise ValueError(f"The two runs should share {SPLIT_SETTINGS}, so that they share one training split.")
        recipe = recipes["h"]
    test_texts, train_texts = _write_labels(config=config, paths=paths, recipe=recipe)
    if not config.saved_only:
        for model_name in MODELS:
            encode_run(config=config, paths=paths, model_name=model_name, test_texts=test_texts, train_texts=train_texts, device=device)
    if config.saved_runs_dir is not None:
        encode_saved_runs(config=config, paths=paths, test_texts=test_texts, device=device)
