"""Every figure of the paper, under the file names the paper and the slides include.

What belongs here
-----------------
- One function per figure. Each takes numbers, draws, and saves.

What does NOT belong here
-------------------------
- Computing the numbers (``evaluate.py``).
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

IN_NOTEBOOK = hasattr(builtins, "__IPYTHON__")  # IPython sets this; a plain script has no display
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from experiments.data.translations import LANG_NAMES, LANGS  # noqa: E402

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# Okabe-Ito colours, readable for colour-blind readers; a marker per language so colour is never the only cue.
COLOR_H, COLOR_DELTA = "#0072B2", "#D55E00"
LANG_STYLE = {"en": ("#000000", "o"), "de": ("#E69F00", "s"), "fr": ("#56B4E9", "^"), "es": ("#009E73", "D"), "zh": ("#CC79A7", "v")}


def save_figure(fig: Any, *, output_dir: str | Path, name: str) -> Path:
    """Save under ``name``; in a notebook also show it."""

    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"saved {path}")
    return path


def smooth(values: Sequence[float], *, window: int = 200) -> np.ndarray:
    """Moving average, for the training curves. A shorter history than the window uses all of it."""

    if len(values) == 0:
        return np.array([])
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_theta_example(*, theta: np.ndarray, top_words: Sequence[str], doc: int, output_dir: str | Path) -> Path:
    """LDA's mixture for one document."""

    top = theta.argmax()
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.bar(range(len(theta)), theta, color=["black" if k == top else "0.7" for k in range(len(theta))])
    ax.set(xticks=range(len(theta)), xlabel="topic", ylabel="θ", title=f"LDA's θ for document {doc}: top topic {top} ({top_words[top]})")
    return save_figure(fig, output_dir=output_dir, name="theta_example.pdf")


def plot_training(
    *,
    runs: Mapping[str, tuple[Sequence[Mapping[str, float]], Sequence[Mapping[str, Any]]]],
    num_docs: int,
    output_dir: str | Path,
    name: str | None = "training.pdf",
    ylim: tuple[float, float] | None = (300, 750),
) -> None:
    """Training curves and per-epoch test scores. ``runs`` maps a label to (history rows, test rows)."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.3))
    for (label, (history, test_rows)), color in zip(runs.items(), (COLOR_H, COLOR_DELTA)):
        axes[0].plot(smooth([row["rec"] for row in history]), color=color, label=label)
        axes[1].plot(smooth([row["align"] for row in history]), color=color, label=label)
        epochs = list(test_rows)[1:]  # one row per epoch; row 0 is before training
        kept = min(epochs, key=lambda row: row["val"])["epoch"]
        axes[2].plot([row["epoch"] for row in epochs], [row["en"] for row in epochs], "o-", color=color, label=f"{label}, English")
        axes[2].plot([row["epoch"] for row in epochs], [row["de"] for row in epochs], "s--", color=color, label=f"{label}, German")
        axes[2].axvline(kept, color=color, lw=0.8, ls=":")
    axes[0].set(title="training: reconstruction per token", xlabel="step")
    axes[1].set(title="training: alignment  −Σ θ log θ̂", xlabel="step")
    axes[2].set(title=f"test: top topic right (of {num_docs:,})\ndotted: the kept epoch", xlabel="epoch", ylim=ylim)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    if name is not None:
        save_figure(fig, output_dir=output_dir, name=name)
    elif IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


def plot_topic_score(*, scores: Mapping[str, Mapping[str, int]], majority: int, num_docs: int, output_dir: str | Path) -> Path:
    """Score 1 per language and predictor."""

    fig, ax = plt.subplots(figsize=(10, 3.4))
    for j, (name, color) in enumerate(zip(scores, ["0.75", "0.5", COLOR_H, COLOR_DELTA])):
        ax.bar(np.arange(len(LANGS)) + (j - 1.5) * 0.2, [scores[name][lang] for lang in LANGS], 0.2, label=name, color=color)
    ax.axhline(majority, color="black", lw=0.8, ls="--", label="always the most common topic")
    ax.set(xticks=range(len(LANGS)), xticklabels=[LANG_NAMES[lang] for lang in LANGS], ylim=(0, num_docs), ylabel=f"top topic right (of {num_docs:,})")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    return save_figure(fig, output_dir=output_dir, name="topic_score.pdf")


def plot_topk(
    *,
    topk: Mapping[str, Mapping[str, Sequence[int]]],
    floor: Sequence[int],
    num_docs: int,
    output_dir: str | Path,
) -> Path:
    """The teacher's top topic within the predictor's top k, per predictor and language."""

    ks = np.arange(1, len(floor) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, (name, per_lang) in zip(axes, topk.items()):
        for lang in LANGS:
            color, marker = LANG_STYLE[lang]
            ax.plot(ks, per_lang[lang], marker=marker, ms=4, lw=1.5, color=color, label=LANG_NAMES[lang])
        ax.plot(ks, floor, color="black", ls="--", lw=1, label="always the k most common")
        ax.set(title=name, xlabel="k", xticks=[1, 5, 10], xlim=(0.5, 10.5), ylim=(0, num_docs))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"LDA's top topic in the\nmodel's top k (of {num_docs:,})")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    return save_figure(fig, output_dir=output_dir, name="topk.pdf")


def plot_one_document(
    *,
    thetas: Mapping[str, Mapping[str, np.ndarray]],
    top_topic: int,
    doc: int,
    output_dir: str | Path,
) -> Path:
    """One document's mixture in five languages, per model; the black bar is the teacher's top topic."""

    fig, axes = plt.subplots(2, len(LANGS), figsize=(14, 4), sharey=True)
    for r, (name, per_lang) in enumerate(thetas.items()):
        for c, lang in enumerate(LANGS):
            theta = per_lang[lang][doc]
            axes[r, c].bar(range(len(theta)), theta, color=["black" if t == top_topic else "0.7" for t in range(len(theta))])
            axes[r, c].set(title=LANG_NAMES[lang] if r == 0 else "", xticks=[])
        axes[r, 0].set_ylabel(name)
    return save_figure(fig, output_dir=output_dir, name="one_document.pdf")


def plot_language_score(*, lang_scores: Mapping[str, float], output_dir: str | Path) -> Path:
    """Score 2 per predictor."""

    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.barh(list(lang_scores), list(lang_scores.values()), color=["0.5", COLOR_H, COLOR_DELTA])
    ax.axvline(0.2, color="black", lw=0.8, ls="--")
    ax.text(0.205, 2.4, "chance", fontsize=8)
    ax.set(xlim=(0, 1), xlabel="language guessed right from θ̂ (lower is better)")
    ax.invert_yaxis()
    return save_figure(fig, output_dir=output_dir, name="language_score.pdf")


def _dots(ax: Any, xy: np.ndarray, labels: np.ndarray, names: Mapping[int, str], colors: Mapping[int, Any], *, markers: Mapping[int, str] | None = None, num_labeled: int = 0, legend: bool = False) -> None:
    for c in np.unique(labels):
        chosen = labels == c
        ax.scatter(*xy[chosen].T, s=14, color=colors[c], marker=markers[c] if markers else "o", alpha=0.6, lw=0, label=names[c])
    for c in np.argsort(-np.bincount(labels))[:num_labeled]:  # name the largest classes at the middle of their dots
        if (labels == c).any():
            ax.annotate(names[c], np.median(xy[labels == c], 0), fontsize=9, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", lw=0.6, alpha=0.9))
    if legend:
        ax.legend(markerscale=2, frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set(xticks=[], yticks=[])


def plot_latent_space(
    *,
    thetas: Mapping[str, Mapping[str, np.ndarray]],
    topic: np.ndarray,
    text_lang: np.ndarray,
    top_words_two: Sequence[str],
    output_dir: str | Path,
) -> Path:
    """t-SNE of the latents, coloured by topic and by language, per model."""

    num_topics = len(top_words_two)
    topic_names = {k: f"{k}: {' '.join(words.split()[:2])}" for k, words in enumerate(top_words_two)}
    topic_colors = {k: plt.cm.tab20(k) for k in range(num_topics)}
    lang_names = {j: LANG_NAMES[lang] for j, lang in enumerate(LANGS)}
    lang_colors = {j: LANG_STYLE[lang][0] for j, lang in enumerate(LANGS)}
    lang_markers = {j: LANG_STYLE[lang][1] for j, lang in enumerate(LANGS)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for col, (name, per_lang) in enumerate(thetas.items()):
        # log theta_hat = mu - logsumexp(mu), so centering each row gives mu back, up to a shift softmax ignores
        logs = np.log(np.concatenate([per_lang[lang] for lang in LANGS]))
        latent = logs - logs.mean(1, keepdims=True)
        perplexity = 30 if len(latent) > 90 else max(1, (len(latent) - 1) // 3)  # smaller only in smoke tests
        xy = TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=0).fit_transform(latent)
        print(f"{name}: latent {latent.shape} -> t-SNE {xy.shape}")
        _dots(axes[0, col], xy, topic, topic_names, topic_colors, num_labeled=14)
        _dots(axes[1, col], xy, text_lang, lang_names, lang_colors, markers=lang_markers, legend=(col == 1))
        axes[0, col].set_title(f"{name}: color = LDA topic", fontsize=13)
        axes[1, col].set_title(f"{name}: color and shape = language", fontsize=13)
    fig.tight_layout()
    return save_figure(fig, output_dir=output_dir, name="latent_space.pdf")


def plot_generation_score(
    *,
    gen_scores: Mapping[str, Mapping[int, int]],
    num_topics: int,
    per_topic: int,
    output_dir: str | Path,
) -> Path:
    """Score 3 per topic and model."""

    topics = sorted(next(iter(gen_scores.values())))
    fig, ax = plt.subplots(figsize=(11, 3))
    for j, ((name, scores), color) in enumerate(zip(gen_scores.items(), (COLOR_H, COLOR_DELTA))):
        ax.bar(np.arange(len(topics)) + (j - 0.5) * 0.4, [scores[k] for k in topics], 0.4, label=name, color=color)
    ax.axhline(per_topic / num_topics, color="black", lw=0.8, ls="--")
    ax.set(xticks=range(len(topics)), xticklabels=topics, xlabel="LDA topic", ylim=(0, per_topic), ylabel=f"written texts LDA\nputs in the topic (of {per_topic})")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    return save_figure(fig, output_dir=output_dir, name="generation_score.pdf")


def plot_layers(*, layers: Sequence[Mapping[str, Any]], majority: int, num_docs: int, output_dir: str | Path) -> Path:
    """The three scores of plain Qwen after every block."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for lang in LANGS:
        color, marker = LANG_STYLE[lang]
        axes[0].plot([row["topic"][lang] for row in layers], marker=marker, ms=4, color=color, label=LANG_NAMES[lang])
    axes[0].axhline(majority, color="black", lw=0.8, ls="--")
    axes[0].set(title=f"topic: top topic right (of {num_docs:,})", xlabel="block", ylim=(0, num_docs))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot([row["language"] for row in layers], "o-", color="black", ms=4)
    axes[1].axhline(0.2, color="black", lw=0.8, ls="--")
    axes[1].set(title="language from θ̂ (lower is better)", xlabel="block", ylim=(0, 1))
    axes[2].plot([row["finds_original"] for row in layers], "o-", color="black", ms=4)
    axes[2].set(title=f"translation finds its English original\n(of {num_docs:,}; mean of de, fr, es, zh)", xlabel="block", ylim=(0, num_docs))
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir=output_dir, name="layers.pdf")
