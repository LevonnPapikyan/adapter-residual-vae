"""The generation grid again, with the 12 merged 20 Newsgroups groups in place of LDA's 20 topics.

    cd main && python -u ../extensions/two_latents/group_grid.py

Saved weights of the two_latents run; nothing is trained.

1. Expected, fixed before anything is written, from the English TRAINING posts only:
   - each LDA topic belongs to the merged group that most of its training posts come from;
   - each group's expected LDA topic is the most common LDA top topic among its training posts.
2. z_topic of a group    = mean mu_topic of the test texts of that group (all five versions);
   z_language of a language = mean mu_language of the test texts in that language.
3. For each of 12 groups x 5 languages, 50 texts are written from (group mean, language mean).
4. Each text: its language is detected; it is translated back to English; the English LDA gives its top topic.
   - "group right": LDA's top topic belongs to the chosen group;
   - "expected topic": LDA's top topic is the group's expected topic.
5. The same judge on the real test documents of each group and language is the ceiling.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_two_latents import LANG_NAMES, LANGS, NUM_LANGS, detect_language, load_two_latent_model, to_english, write_all  # noqa: E402

from core.train.loop import resolve_device
from experiments.data.corpus import load_corpus
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher
from experiments.data.translations import NUM_WORDS, load_translations

MERGED = {
    "PC, Mac, MS Windows": ("comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.os.ms-windows.misc"),
    "graphics, X Windows": ("comp.graphics", "comp.windows.x"),
    "religion": ("alt.atheism", "soc.religion.christian", "talk.religion.misc"),
    "sports: baseball, hockey": ("rec.sport.baseball", "rec.sport.hockey"),
    "autos, motorcycles": ("rec.autos", "rec.motorcycles"),
    "politics: guns, misc": ("talk.politics.guns", "talk.politics.misc"),
    "politics: mideast": ("talk.politics.mideast",),
    "for sale": ("misc.forsale",),
    "cryptography": ("sci.crypt",),
    "electronics": ("sci.electronics",),
    "medicine": ("sci.med",),
    "space": ("sci.space",),
}
GROUPS = list(MERGED)
PER_CELL = 50
OUT = HERE / "group_grid"


def heatmap(ax, values, title, rows, columns, fmt="{:.2f}"):
    image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(columns))); ax.set_xticklabels(columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", color="w" if values[i, j] < 0.6 else "k", fontsize=7)
    ax.set_title(title, fontsize=10)
    return image


def main() -> None:
    OUT.mkdir(exist_ok=True)
    device = resolve_device(None)
    paths = DataPaths(data_dir=Path("../data"))
    corpus = load_corpus(paths=paths)
    teacher = LDATeacher(paths=paths)
    topic_words = teacher.top_words(6)
    group_of_label = np.array([next(g for g, (_, names) in enumerate(MERGED.items()) if name in names) for name in corpus.label_names])

    # 1. Expected, from the English training posts only.
    train_texts = [" ".join(text.split()[:NUM_WORDS]) for text in corpus.train_texts]
    keep = np.flatnonzero(np.asarray(teacher.counts(train_texts).sum(1)).ravel() >= 5)
    train_top = teacher.theta([train_texts[i] for i in keep]).argmax(1)
    train_group = group_of_label[corpus.train_labels[keep]]
    table = np.zeros((20, len(GROUPS)))
    np.add.at(table, (train_top, train_group), 1)
    group_of_topic = table.argmax(1)
    expected_topic = table.argmax(0)
    expected = {
        "each LDA topic's group": {
            f"{t}: {topic_words[t]}": {"group": GROUPS[group_of_topic[t]], "share of the topic's training posts in it": round(float(table[t].max() / table[t].sum()), 2)} for t in range(20)
        },
        "each group's expected LDA topic": {
            GROUPS[g]: {"topic": f"{expected_topic[g]}: {topic_words[expected_topic[g]]}", "share of the group's training posts with it": round(float(table[:, g].max() / table[:, g].sum()), 2),
                        "LDA topics that belong to the group": [int(t) for t in np.flatnonzero(group_of_topic == g)]} for g in range(len(GROUPS))
        },
    }
    print(json.dumps(expected, indent=1, ensure_ascii=False), flush=True)

    # 2. Centers, from the saved latents of the test texts.
    saved = np.load(HERE / "main_features.npz")
    mu_topic, mu_language, language = saved["mu_topic"], saved["mu_language"], saved["language"]
    text_group = group_of_label[saved["newsgroup"]]
    group_center = np.stack([mu_topic[text_group == g].mean(0) for g in range(len(GROUPS))])
    language_center = np.stack([mu_language[language == k].mean(0) for k in range(NUM_LANGS)])

    # 5. The ceiling: the real test documents through the same judge.
    translated = load_translations(paths=paths)
    doc_group = text_group[: translated.num_docs]
    ceiling_group = np.zeros((len(GROUPS), NUM_LANGS)); ceiling_expected = np.zeros_like(ceiling_group)
    for k, lang in enumerate(LANGS):
        top = teacher.theta(to_english(translated.texts[lang], lang)).argmax(1)
        for g in range(len(GROUPS)):
            ceiling_group[g, k] = (group_of_topic[top[doc_group == g]] == g).mean()
            ceiling_expected[g, k] = (top[doc_group == g] == expected_topic[g]).mean()

    # 3 and 4. Write and judge.
    model = load_two_latent_model(argv=["--config", str(HERE / "main.json")], device=device)
    torch.manual_seed(0)
    chosen = np.repeat(np.arange(len(GROUPS)), PER_CELL)
    group_right = np.zeros((len(GROUPS), NUM_LANGS)); expected_right = np.zeros_like(group_right); language_right = np.zeros_like(group_right)
    confusion = np.zeros((len(GROUPS), len(GROUPS)))
    said: dict = {}
    samples: dict = {}
    for k, lang in enumerate(LANGS):
        written = write_all(model, group_center[chosen], np.repeat(language_center[k][None], len(chosen), 0))
        detected = np.array([detect_language(text) for text in written])
        english = to_english(written, lang)
        top = teacher.theta(english).argmax(1)
        for g in range(len(GROUPS)):
            cell = chosen == g
            group_right[g, k] = (group_of_topic[top[cell]] == g).mean()
            expected_right[g, k] = (top[cell] == expected_topic[g]).mean()
            language_right[g, k] = (detected[cell] == lang).mean()
            np.add.at(confusion[g], group_of_topic[top[cell]], 1)
            said[f"{GROUPS[g]} | {LANG_NAMES[lang]}"] = {
                "LDA's top topics of the written texts": {f"{t}: {topic_words[t]}": c for t, c in Counter(top[cell].tolist()).most_common(3)},
                "languages written": dict(Counter(detected[cell].tolist()).most_common()),
            }
        samples[lang] = {GROUPS[g]: [{"written": written[i], "in_english": english[i], "lda_top_topic": f"{top[i]}: {topic_words[top[i]]}", "detected": str(detected[i])} for i in np.flatnonzero(chosen == g)[:3]] for g in range(len(GROUPS))}
        print(LANG_NAMES[lang], "language right", round(float(language_right[:, k].mean()), 3), "group right", round(float(group_right[:, k].mean()), 3),
              "expected topic", round(float(expected_right[:, k].mean()), 3), "| ceiling group", round(float(ceiling_group[:, k].mean()), 3), flush=True)

    names = [LANG_NAMES[lang] for lang in LANGS]
    results = {
        "expected": expected,
        "per_language_mean": {n: {"language_right": round(float(language_right[:, k].mean()), 3), "group_right": round(float(group_right[:, k].mean()), 3),
                                  "expected_topic_right": round(float(expected_right[:, k].mean()), 3), "ceiling_group_right_on_real_documents": round(float(ceiling_group[:, k].mean()), 3),
                                  "ceiling_expected_topic_on_real_documents": round(float(ceiling_expected[:, k].mean()), 3)} for k, n in enumerate(names)},
        "group_right": {GROUPS[g]: dict(zip(names, group_right[g].round(2).tolist())) for g in range(len(GROUPS))},
        "expected_topic_right": {GROUPS[g]: dict(zip(names, expected_right[g].round(2).tolist())) for g in range(len(GROUPS))},
        "language_right": {GROUPS[g]: dict(zip(names, language_right[g].round(2).tolist())) for g in range(len(GROUPS))},
        "ceiling_group_right": {GROUPS[g]: dict(zip(names, ceiling_group[g].round(2).tolist())) for g in range(len(GROUPS))},
        "what_lda_and_the_detector_said": said,
        "texts_per_cell": PER_CELL,
        "chance_group_right": {GROUPS[g]: round(float((group_of_topic == g).sum() / 20), 2) for g in range(len(GROUPS))},
    }
    (OUT / "group_grid_scores.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (OUT / "group_grid_samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))

    rows = [f"{g}  (expects topic {expected_topic[i]}: {' '.join(topic_words[expected_topic[i]].split()[:3])})" for i, g in enumerate(GROUPS)]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.6), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    heatmap(axes[0], group_right, "Written text: LDA's topic belongs to the chosen group", rows, names)
    heatmap(axes[1], ceiling_group, "Ceiling: the same judge on real documents", [""] * len(GROUPS), names)
    heatmap(axes[2], expected_right, "Written text: LDA's topic is the expected topic", [""] * len(GROUPS), names)
    image = heatmap(axes[3], language_right, "Written text: language is the chosen one", [""] * len(GROUPS), names)
    fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01)
    fig.savefig(OUT / "group_grid_heatmaps.png", dpi=150, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    heatmap(ax, confusion / confusion.sum(1, keepdims=True), "Chosen group (rows) against the group LDA reads (columns), all languages", GROUPS, GROUPS)
    fig.savefig(OUT / "group_grid_confusion.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
