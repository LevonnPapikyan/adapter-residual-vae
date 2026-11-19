"""Sweeps around the paper's Δh model: one setting changed per run, everything else as configs/paper/factorization_delta.json.

    cd main && python -u ../extensions/sweeps/train_and_score.py --config ../extensions/sweeps/configs/beta_0.1.json
    cd main && python -u ../extensions/sweeps/train_and_score.py --score-only ../models/final_0.5b/delta --name paper_delta

The unchanged training code of main/ trains the model (seed 0, epoch with the lowest held-out loss is kept). Then
the saved weights are scored on the 1,000 test documents in 5 languages:

    topic_right        argmax of mu is LDA's top topic of the English original, of 1,000, per language
    language_score     5-fold probe from theta-hat to the language (chance 0.20, lower is better)
    rec_per_token_en   reconstruction of the English documents from mu, per token
    rate_en            KL(q(z|x) || N(0, I)) per English document
    generation         for each LDA topic, 10 texts written from the topic's mean z; how many LDA puts back in the topic
                       (of 200). The paper uses all 7,107 test posts for the means; here the 1,000 scored documents.

This folder imports main/ and writes nothing under it. Scores go to scores/<name>.json, written texts to samples/<name>.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

from core.analysis.generation import generate_by_topic, generation_score
from core.analysis.language_probe import language_score
from core.analysis.topic_score import majority_count, topic_score
from core.factorization.checkpointing import load_trained_run
from core.train.loop import resolve_device
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import LANG_NAMES, LANGS, load_translations
import experiments.factorization.train as train_module

HERE = Path(__file__).resolve().parent
BATCH = 32


@torch.no_grad()
def score(model, *, texts: dict[str, list[str]], teacher: LDATeacher, target: np.ndarray) -> tuple[dict, dict]:
    model.eval()
    mus, rate, rec = {}, 0.0, 0.0
    for lang in LANGS:
        parts = []
        for i in range(0, len(texts[lang]), BATCH):
            batch = texts[lang][i : i + BATCH]
            posterior = model.posterior(texts=batch)
            mu, logvar = posterior.mu.float(), posterior.logvar.float()
            parts.append(mu.cpu())
            if lang == "en":
                rate += (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)).sum().item()
                plain = model.reconstruction_nll(z=posterior.mu, texts=batch)
                rec += (plain.nll / plain.num_tokens).sum().item()
        mus[lang] = torch.cat(parts).numpy()
    thetas = {lang: torch.softmax(torch.tensor(mus[lang]), -1).numpy() for lang in LANGS}
    torch.manual_seed(0)  # sampling z, and the generator's sampling
    written = generate_by_topic(model=model, teacher=teacher, texts=texts["en"], per_topic=10, max_new_tokens=96)
    per_topic = generation_score(teacher=teacher, generated=written)
    scores = {
        "topic_right": {LANG_NAMES[lang]: int(topic_score(theta_pred=thetas[lang], target=target)) for lang in LANGS},
        "language_score": round(float(language_score(thetas=thetas)), 3),
        "rec_per_token_en": round(rec / len(target), 4),
        "rate_en": round(rate / len(target), 3),
        "generation_right_of_200": int(sum(per_topic.values())),
        "generation_per_topic": {str(topic): count for topic, count in per_topic.items()},
    }
    return scores, {str(topic): texts_[:3] for topic, texts_ in written.items()}


def main() -> None:
    argv = sys.argv[1:]
    if "--score-only" in argv:
        run_dir = Path(argv[argv.index("--score-only") + 1])
        name = argv[argv.index("--name") + 1]
        settings, data_dir = {"run_dir": str(run_dir)}, Path("../data")
    else:
        config_path = Path(argv[argv.index("--config") + 1])
        config = json.loads(config_path.read_text())
        name, run_dir, data_dir = config_path.stem, Path(config["output_dir"]), Path(config["data_dir"])
        settings = {key: config[key] for key in ("mu_from", "beta", "lam", "lora_rank", "base_model", "num_epochs", "seed")}
        if not (run_dir / "model.pt").exists():
            train_module.main(argv)
    paths = DataPaths(data_dir=data_dir)
    translated = load_translations(paths=paths)
    teacher = LDATeacher(paths=paths)
    target = teacher.theta(translated.texts["en"])
    model = load_trained_run(run_dir=run_dir, device=resolve_device(None)).model
    scores, samples = score(model, texts=translated.texts, teacher=teacher, target=target)
    history = run_dir / "test.json"
    if history.exists():
        epochs = json.loads(history.read_text())
        scores["held_out_loss_by_epoch"] = [round(e["val"], 3) for e in epochs]
    results = {"name": name, "settings": settings, "majority_baseline": int(majority_count(target=target)), **scores}
    for folder, content in (("scores", results), ("samples", samples)):
        (HERE / folder).mkdir(exist_ok=True)
        (HERE / folder / f"{name}.json").write_text(json.dumps(content, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
