"""The trajectory figures, from ``results.json`` and the cached maps. No GPU.

What belongs here
-----------------
- One function per figure. Each reads numbers, draws, and saves a PDF and a PNG.

What does NOT belong here
-------------------------
- Computing the numbers (``trajectory_results.py``).

Colours are Okabe-Ito throughout. Twenty topics cannot have twenty colours that everyone
can tell apart, so a topic is a colour and a marker together, and a topic whose points
have gathered is also named by its number on the panel. Every figure says which
documents it plots and in which float format.
"""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402

from core.analysis.trajectory import STATE_VECTORS  # noqa: E402
from core.runs.io import read_json  # noqa: E402
from experiments.factorization.figures import COLOR_DELTA, COLOR_H, LANG_STYLE  # noqa: E402
from experiments.factorization.trajectory_cache import MODELS, TrajectoryPaths  # noqa: E402
from experiments.factorization.trajectory_results import GRIDS  # noqa: E402

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

MODEL_LABEL = {"h": "h model (μ from h_on)", "delta": "Δh model (μ from Δh)"}
MODEL_COLOR = {"h": COLOR_H, "delta": COLOR_DELTA}
VECTOR_LABEL = {"h_off": "h_off", "h_on": "h_on", "delta": "Δh", "mu": "μ"}
TOPIC_COLORS = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")  # Okabe-Ito without its yellow
TOPIC_MARKERS = ("o", "s", "^")
TEST_COLOR, TRAIN_COLOR = "#0072B2", "#D55E00"
SHARE_COLORS = {"language": "#CC79A7", "document": "#009E73", "neither": "0.55"}
SHARE_MARKERS = {"language": "o", "document": "s", "neither": "^"}
FLOAT_NOTE = "float32 evaluation of weights trained in bfloat16"
TSNE_NOTE = "t-SNE distances and axes are not comparable across panels."
CORNER_NOTE = (
    "Corner text: 'in plane' is the share of this vector's variance that the 2D plane shows; 'full space' shares and variances "
    "are measured in the whole vector, not on the plane. x and y are scaled independently and carry no units."
)
TOPIC_LABEL_NOTE = "A number marks a tight group of points that is at least 30% that topic; a missing number is only that threshold, not a claim about the topic."
STEP_ZERO_NOTE = "Curves of Δh and of the Δh model's μ start at step 10: at step 0 Δh = 0 exactly, so these scores are undefined there."
H_OFF_NOTE = "float32, plain Qwen with no trained weights."
NUM_CHUNKS = 25  # draw order: random chunks, so no language or topic always sits on top


def topic_style(topic: int) -> tuple[str, str]:
    """A topic's colour and marker."""

    return TOPIC_COLORS[topic % len(TOPIC_COLORS)], TOPIC_MARKERS[(topic // len(TOPIC_COLORS)) % len(TOPIC_MARKERS)]


def save_both(fig: Any, *, figures_dir: Path, name: str) -> None:
    """Save a figure as PDF and PNG, and close it."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"saved {figures_dir / name}", flush=True)


class Drawing:
    """What every figure reads: the numbers, the labels, and where to save."""

    def __init__(self, *, paths: TrajectoryPaths) -> None:
        self.paths = paths
        self.results = read_json(paths.results)
        self.meta = self.results["documents"]
        self.langs: list[str] = self.meta["langs"]
        self.lang_names = dict(zip(self.langs, self.meta["lang_names"]))
        self.num_docs: int = self.meta["num_docs"]
        with np.load(paths.labels_npz) as labels:
            self.text_lang, self.topic = labels["text_lang"], labels["topic"]
        self.order = np.random.default_rng(0).permutation(len(self.text_lang))

    @property
    def test_note(self) -> str:
        return f"{self.num_docs:,} test documents x {len(self.langs)} languages, {FLOAT_NOTE}."

    def rows(self, model: str) -> Mapping[str, Mapping[str, Any]]:
        return self.results["models"][model]["rows"]["float32"]

    def steps(self, model: str) -> list[int]:
        return self.results["models"][model]["steps"]

    def save(self, fig: Any, name: str) -> None:
        save_both(fig, figures_dir=self.paths.figures, name=name)


# ----------------------------------------------------------------------------- maps


def _scatter(drawing: Drawing, ax: Any, xy: np.ndarray, color_by: str) -> None:
    ax.set_rasterization_zorder(2)  # the points become one image in the PDF, the text stays text
    for chunk in np.array_split(drawing.order, NUM_CHUNKS):
        if color_by == "language":
            for j, lang in enumerate(drawing.langs):
                color, marker = LANG_STYLE[lang]
                pick = chunk[drawing.text_lang[chunk] == j]
                ax.scatter(*xy[pick].T, s=4, color=color, marker=marker, alpha=0.5, lw=0, zorder=1)
        else:
            styles = [topic_style(int(topic)) for topic in drawing.topic[chunk]]
            for marker in TOPIC_MARKERS:
                keep = [i for i, style in enumerate(styles) if style[1] == marker]
                if keep:
                    ax.scatter(*xy[chunk[keep]].T, s=9, color=[styles[i][0] for i in keep], marker=marker, alpha=0.6, lw=0, zorder=1)
    if color_by == "topic":
        _name_gathered_topics(drawing, ax, xy)


def _name_gathered_topics(drawing: Drawing, ax: Any, xy: np.ndarray) -> None:
    """Write a topic's number on each dense group of its points, not where they are spread over the whole panel."""

    spread = np.median(np.linalg.norm(xy - np.median(xy, 0), axis=1))
    span = np.percentile(xy, 99, axis=0) - np.percentile(xy, 1, axis=0)
    placed: list[np.ndarray] = []
    for topic in np.unique(drawing.topic):
        for middle in topic_label_places(xy[drawing.topic == topic], all_points=xy, spread=spread):
            if any((np.abs(middle - other) < 0.05 * span).all() for other in placed):  # labels must not cover each other, or leave their points
                continue
            placed.append(middle)
            ax.annotate(str(topic), middle, fontsize=7, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", lw=0.4, alpha=0.85))


def topic_label_places(points: np.ndarray, *, all_points: np.ndarray, spread: float) -> list[np.ndarray]:
    """The middle of each tight group of one topic's points. A group needs a tenth of the topic, so a topic split by language
    gets one label per blob, and at least 30% of all the points around it, so a topic merely inside a mixed cloud gets none."""

    if spread <= 0:
        return []
    groups = DBSCAN(eps=0.15 * spread, min_samples=5).fit_predict(points)
    places = []
    for group in set(groups) - {-1}:
        members = points[groups == group]
        middle = np.median(members, 0)
        radius = np.median(np.linalg.norm(members - middle, axis=1))
        if len(members) < 0.1 * len(points) or radius >= 0.5 * spread:
            continue
        around = (np.linalg.norm(all_points - middle, axis=1) <= radius).sum()
        if (np.linalg.norm(members - middle, axis=1) <= radius).sum() >= 0.3 * around:
            places.append(middle)
    return places


def _legend(drawing: Drawing, fig: Any, color_by: str, *, side: str = "right") -> None:
    where = {"loc": "center left", "bbox_to_anchor": (1, 0.5)} if side == "right" else {"loc": "center right", "bbox_to_anchor": (0, 0.5)}
    if color_by == "language":
        handles = [plt.Line2D([], [], ls="", marker=LANG_STYLE[lang][1], color=LANG_STYLE[lang][0], label=drawing.lang_names[lang], ms=5) for lang in drawing.langs]
        fig.legend(handles=handles, frameon=False, **where)
    else:
        handles = [plt.Line2D([], [], ls="", marker=topic_style(k)[1], color=topic_style(k)[0], label=name, ms=5) for k, name in enumerate(drawing.meta["topic_names"])]
        fig.legend(handles=handles, frameon=False, fontsize=8, title="LDA top topic of\nthe English original", **where)


def _limits(coords: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    points = coords.reshape(-1, 2)
    points = points[np.isfinite(points).all(1)]
    low, high = np.percentile(points, [0.2, 99.8], axis=0)
    pad = 0.05 * (high - low)
    return (low[0] - pad[0], high[0] + pad[0]), (low[1] - pad[1], high[1] + pad[1])


def _blank_panel(ax: Any, label: str, step: int) -> None:
    ax.text(
        0.5,
        0.5,
        f"{label} is the same for every text at step {step}:\nevery LoRA B is 0, so Δh = 0 exactly.\nNothing to plot.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="0.3",
    )


def _share_note(row: Mapping[str, Any]) -> str:
    return (
        f"full space: lang {row['language_share']:.3f} (var {row['variance_language']:.3g})\n"
        f"full space: doc {row['document_share']:.3f} (var {row['variance_document']:.3g})"
    )


def _num_outside(xy: np.ndarray, limits: tuple[tuple[float, float], tuple[float, float]]) -> int:
    (x_low, x_high), (y_low, y_high) = limits
    return int((~((xy[:, 0] >= x_low) & (xy[:, 0] <= x_high) & (xy[:, 1] >= y_low) & (xy[:, 1] <= y_high))).sum())


def _panels(num: int, *, per_row: int = 5) -> tuple[Any, np.ndarray]:
    num_rows = int(np.ceil(num / per_row))
    fig, axes = plt.subplots(num_rows, per_row, figsize=(3 * per_row, 2.9 * num_rows + 0.9), squeeze=False)
    axes = axes.ravel()
    for ax in axes[num:]:
        ax.axis("off")
    return fig, axes


def _finish(fig: Any, title: str, note: str) -> None:
    width, height = fig.get_size_inches()
    note = textwrap.fill(note, width=int(width * 13))
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=9, color="0.25")
    fig.tight_layout(rect=(0, (0.1 + 0.17 * (note.count("\n") + 1)) / height, 1, 0.97))


def map_grid(drawing: Drawing, *, model: str, vector: str, kind: str, color_by: str) -> None:
    """One grid: one panel per step, one vector of one model, one kind of map, one colouring."""

    with np.load(drawing.paths.maps(model, vector)) as maps:
        steps, coords, disparity = maps["steps"].tolist(), maps[kind], maps["disparity"]
        plane = {"fixed": maps["plane_fixed"], "aligned": maps["plane_own"]}.get(kind)
    kept = drawing.results["models"][model]["kept_step"]
    fig, axes = _panels(len(steps))
    limits = _limits(coords) if kind == "fixed" else None
    label = VECTOR_LABEL[vector]
    for i, step in enumerate(steps):
        ax = axes[i]
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"step {step}" + (" (kept)" if step == kept else ""), fontsize=10)
        if not np.isfinite(coords[i]).all():
            _blank_panel(ax, label, step)
            continue
        _scatter(drawing, ax, coords[i], color_by)
        if limits:
            ax.set(xlim=limits[0], ylim=limits[1])
        row = drawing.rows(model)[str(step)]["vectors"][vector]
        note = _share_note(row)
        if plane is not None:
            note = f"in plane {100 * plane[i]:.1f}%\n" + note
        if kind == "aligned":
            note += f"\nProcrustes disparity {disparity[i]:.2f}"
        if limits and _num_outside(coords[i], limits):
            note += f"\n{_num_outside(coords[i], limits)} of {len(coords[i]):,} points outside the axes"
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=6.5, color="0.2", va="bottom", bbox=dict(fc="white", alpha=0.75, lw=0, pad=1))
    if kind == "fixed":
        how = f"PCA fitted at step {drawing.results['models'][model]['final_step']} on the test texts, every checkpoint projected into it, same axes in every panel"
        extra = " Axis limits are the 0.2 to 99.8 percentiles over all panels; a panel says how many of its points fall outside. " + CORNER_NOTE
    elif kind == "aligned":
        how = f"PCA refit at each checkpoint, aligned to step {drawing.results['models'][model]['final_step']} by orthogonal Procrustes, each panel its own scale"
        extra = " A disparity near 1 means the alignment does little. " + CORNER_NOTE
    else:
        tsne = drawing.results["tsne"]
        how = f"t-SNE per checkpoint on PCA-{tsne['pca_dims']} features, PCA start, perplexity {tsne['perplexity']:g}, seed {tsne['seed']}"
        extra = " " + TSNE_NOTE + " Corner text: shares and variances measured in the whole vector, not on this map."
    if color_by == "topic":
        extra += " " + TOPIC_LABEL_NOTE
    _finish(fig, f"{label} of the {MODEL_LABEL[model]}, by {color_by}. {how}", drawing.test_note + extra)
    _legend(drawing, fig, color_by)
    drawing.save(fig, f"{'tsne' if kind == 'tsne' else 'pca_' + kind}_{model}_{vector}_{color_by}")


def h_off_figure(drawing: Drawing, *, kind: str) -> None:
    """Plain Qwen, once: it does not change with the checkpoints."""

    with np.load(drawing.paths.maps_h_off) as maps:
        coords, plane = maps["fixed" if kind == "pca" else "tsne"], float(maps["plane"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, color_by, title in zip(axes, ("language", "topic"), ("by language", "by the LDA top topic of the English original")):
        _scatter(drawing, ax, coords, color_by)
        ax.set(xticks=[], yticks=[])
        ax.set_title(title, fontsize=10)
        if kind == "pca":
            ax.text(0.02, 0.02, f"in plane {100 * plane:.1f}%", transform=ax.transAxes, fontsize=7, color="0.2", va="bottom", bbox=dict(fc="white", alpha=0.75, lw=0, pad=1))
    tsne = drawing.results["tsne"]
    how = "PCA in its own basis" if kind == "pca" else f"t-SNE on PCA-{tsne['pca_dims']} features, PCA start, perplexity {tsne['perplexity']:g}, seed {tsne['seed']}"
    documents = f"{drawing.num_docs:,} test documents x {len(drawing.langs)} languages, {H_OFF_NOTE}"
    if kind == "pca":
        note = f"{documents} Not the fixed basis of the pca_fixed grids, so the same points look different at step 0 of pca_fixed_h_h_on_*. 'In plane' is the share of the variance this plane shows."
    else:
        note = f"{documents} {TSNE_NOTE}"
    _finish(fig, f"h_off, plain Qwen, the same at every checkpoint. {how}", note + " " + TOPIC_LABEL_NOTE)
    _legend(drawing, fig, "language", side="left")
    _legend(drawing, fig, "topic", side="right")
    drawing.save(fig, f"{kind}_h_off")


def train_overlay_figure(drawing: Drawing, *, model: str, vector: str) -> None:
    """The English training sample on the test English points, in the fixed basis."""

    with np.load(drawing.paths.maps(model, vector)) as maps:
        steps, coords, train = maps["steps"].tolist(), maps["fixed"], maps["train_fixed"]
    english = coords[:, : drawing.num_docs]
    fig, axes = _panels(len(steps))
    limits = _limits(np.concatenate([english, train], axis=1))
    kept = drawing.results["models"][model]["kept_step"]
    rng = np.random.default_rng(0)
    for i, step in enumerate(steps):
        ax = axes[i]
        ax.set(xticks=[], yticks=[])
        ax.set_title(f"step {step}" + (" (kept)" if step == kept else ""), fontsize=10)
        if not np.isfinite(english[i]).all():
            _blank_panel(ax, VECTOR_LABEL[vector], step)
            continue
        ax.set_rasterization_zorder(2)
        test_chunks = np.array_split(rng.permutation(len(english[i])), NUM_CHUNKS)  # neither colour always on top
        train_chunks = np.array_split(rng.permutation(len(train[i])), NUM_CHUNKS)
        for test_chunk, train_chunk in zip(test_chunks, train_chunks):
            ax.scatter(*english[i][test_chunk].T, s=8, color=TEST_COLOR, marker="o", alpha=0.5, lw=0, zorder=1)
            ax.scatter(*train[i][train_chunk].T, s=8, color=TRAIN_COLOR, marker="^", alpha=0.5, lw=0, zorder=1)
        ax.set(xlim=limits[0], ylim=limits[1])
    meta = drawing.meta
    note = (
        f"{meta['train_sample_size']:,} English training posts, plus {drawing.num_docs:,} English test documents, {FLOAT_NOTE}. "
        f"Basis fitted on all test texts. Test documents are the first 150 words (median {meta['median_words_test_english']:.0f}); "
        f"training posts run up to {meta['encoder_max_tokens']} tokens (median {meta['median_words_train_sample']:.0f} words), so length differs too."
    )
    _finish(fig, f"{VECTOR_LABEL[vector]} of the {MODEL_LABEL[model]}: English training sample on English test points, fixed PCA basis", note)
    handles = [
        plt.Line2D([], [], ls="", marker="o", color=TEST_COLOR, label="English test", ms=5),
        plt.Line2D([], [], ls="", marker="^", color=TRAIN_COLOR, label="English training", ms=5),
    ]
    fig.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    drawing.save(fig, f"train_vs_test_pca_{model}_{vector}")


# ----------------------------------------------------------------------------- lines against step


def _step_axis(ax: Any, *, final_step: int, kept_step: int | None = None, narrow: bool = False) -> None:
    """Log x with a marked 0: linear up to 10, logarithmic after. A narrow panel leaves out the tick at 1,000, which would touch the last one."""

    ax.set_xscale("symlog", linthresh=10, linscale=0.5)
    ticks = [tick for tick in (0, 10, 100) + (() if narrow else (1000,)) if tick < final_step] + [final_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:,}" for tick in ticks])
    ax.minorticks_off()
    ax.set_xlim(0, final_step * 1.18)
    ax.axvline(0, color="0.8", lw=0.8)
    if kept_step is not None:
        ax.axvline(kept_step, color="0.5", lw=0.8, ls=":")
    ax.grid(alpha=0.25)


def _series(values: Sequence[Any]) -> list[float]:
    return [np.nan if value is None else value for value in values]


def _model_axis(drawing: Drawing, ax: Any, model: str, *, narrow: bool = False) -> None:
    run = drawing.results["models"][model]
    _step_axis(ax, final_step=run["final_step"], kept_step=run["kept_step"], narrow=narrow)
    ax.set_xlabel("training step (dotted: kept)")


def _shared_axis(drawing: Drawing, ax: Any, *, narrow: bool = False) -> None:
    """One axis for both models: the dotted kept line of each, in its colour, since seeds can keep different steps."""

    runs = drawing.results["models"]
    _step_axis(ax, final_step=max(runs[model]["final_step"] for model in MODELS), narrow=narrow)
    for model in MODELS:
        ax.axvline(runs[model]["kept_step"], color=MODEL_COLOR[model], lw=0.9, ls=":")
    ax.set_xlabel("training step (dotted: kept, one per model)")


def _kept_values_note(drawing: Drawing, *, what: str, langs: Sequence[str]) -> str:
    """The kept step in both float formats, since every line is float32 and the comparison tables are bfloat16."""

    parts = []
    for model in MODELS:
        run = drawing.results["models"][model]
        kept = str(run["kept_step"])
        if kept not in run["rows"]["float32"]:
            continue
        values = {precision: " ".join(str(run["rows"][precision][kept][what][lang]) for lang in langs) for precision in run["rows"]}
        parts.append(f"{MODEL_LABEL[model].split(' (')[0]}, kept step {kept}: float32 {values['float32']}; bfloat16 {values['bfloat16']}")
    return f"Kept step, {' '.join(langs)}. " + ". ".join(parts) + ". The lines are float32; the comparison tables and the shipped results are bfloat16, as trained."


def lines_topic(drawing: Drawing) -> None:
    """Topic score per language."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        rows, steps = drawing.rows(model), drawing.steps(model)
        for lang in drawing.langs:
            color, marker = LANG_STYLE[lang]
            ax.plot(steps, [rows[str(step)]["topic"][lang] for step in steps], color=color, marker=marker, ms=4, lw=1.5, label=drawing.lang_names[lang])
        ax.axhline(drawing.results["majority"], color="0.4", lw=0.8, ls="--", label="always the commonest topic")
        _model_axis(drawing, ax, model)
        ax.set(title=MODEL_LABEL[model], ylim=(0, drawing.num_docs))
    axes[0].set_ylabel(f"top topic right (of {drawing.num_docs:,})")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    _finish(fig, "Topic score against step", drawing.test_note + " " + _kept_values_note(drawing, what="topic", langs=drawing.langs))
    drawing.save(fig, "lines_topic")


def lines_retrieval(drawing: Drawing) -> None:
    """Retrieval per language on each model's read vector."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        rows, steps = drawing.rows(model), drawing.steps(model)
        for lang in drawing.langs[1:]:
            color, marker = LANG_STYLE[lang]
            found = [None if rows[str(step)]["retrieval"] is None else rows[str(step)]["retrieval"][lang] for step in steps]
            ax.plot(steps, _series(found), color=color, marker=marker, ms=4, lw=1.5, label=drawing.lang_names[lang])
        read = VECTOR_LABEL[drawing.results["models"][model]["read"]]
        plain = float(np.mean(list(rows[str(steps[0])]["vectors"]["h_off"]["finds_original"].values())))
        ax.axhline(plain, color="0.4", lw=0.8, ls="--", label="h_off, mean of the 4")
        _model_axis(drawing, ax, model)
        ax.set(title=f"{MODEL_LABEL[model]}, on {read}", ylim=(0, drawing.num_docs))
    axes[0].set_ylabel(f"translation finds its English original\n(of {drawing.num_docs:,})")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    _finish(fig, "Retrieval on the read vector against step", drawing.test_note + " " + STEP_ZERO_NOTE + " " + _kept_values_note(drawing, what="retrieval", langs=drawing.langs[1:]))
    drawing.save(fig, "lines_retrieval")


def _one_axis_per_model(drawing: Drawing, *, get: Callable[[Mapping[str, Any]], Any], ylabel: str, title: str, name: str, chance: float | None = None, log: bool = False, note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model in MODELS:
        rows, steps = drawing.rows(model), drawing.steps(model)
        ax.plot(steps, _series([get(rows[str(step)]) for step in steps]), color=MODEL_COLOR[model], marker="o" if model == "h" else "s", ms=4, lw=1.5, label=MODEL_LABEL[model])
    if chance is not None:
        ax.axhline(chance, color="0.4", lw=0.8, ls="--", label=f"chance, {chance:.2f}")
    if log:
        ax.set_yscale("log")
    _shared_axis(drawing, ax)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))  # outside, so no line runs through it
    _finish(fig, title, drawing.test_note + note)
    drawing.save(fig, name)


def lines_language_probe(drawing: Drawing) -> None:
    """The language probe from theta_hat."""

    _one_axis_per_model(
        drawing,
        get=lambda row: row["language_probe"],
        ylabel="language guessed right from θ̂\n(lower is better)",
        title="Language probe against step",
        name="lines_language_probe",
        chance=1 / len(drawing.langs),
        note=" " + STEP_ZERO_NOTE,
    )


def lines_delta_ratio(drawing: Drawing) -> None:
    """The size of Delta h."""

    _one_axis_per_model(
        drawing,
        get=lambda row: row["delta_ratio"] or None,
        ylabel="|Δh| / |h_off|, mean over texts",
        title="Size of Δh against step (step 0 is exactly 0, not on this log axis)",
        name="lines_delta_ratio",
        log=True,
    )


def _variance_grid(drawing: Drawing, *, keys: Mapping[str, str], ylabel: str, name: str, title: str, log: bool, note: str, neither: bool = False) -> None:
    fig, axes = plt.subplots(len(MODELS), len(STATE_VECTORS), figsize=(13, 6), sharey=True)
    for r, model in enumerate(MODELS):
        rows, steps = drawing.rows(model), drawing.steps(model)
        for c, vector in enumerate(STATE_VECTORS):
            ax = axes[r, c]
            for what, key in keys.items():
                values = [rows[str(step)]["vectors"][vector].get(key) for step in steps]
                ax.plot(steps, _series(values), color=SHARE_COLORS[what], marker=SHARE_MARKERS[what], ms=3, lw=1.5, label=what)
            if neither:
                shares = [rows[str(step)]["vectors"][vector] for step in steps]
                rest = [1 - share["language_share"] - share["document_share"] if share["defined"] else None for share in shares]
                ax.plot(steps, _series(rest), color=SHARE_COLORS["neither"], marker=SHARE_MARKERS["neither"], ms=3, lw=1.5, label="neither")
            if log:
                ax.set_yscale("log")
            elif r == 0 and c == 0:
                ax.set_ylim(0, 1)
            _model_axis(drawing, ax, model, narrow=True)
            ax.set_title(f"{VECTOR_LABEL[vector]}, {MODEL_LABEL[model].split(' (')[0]}", fontsize=9)
        axes[r, 0].set_ylabel(ylabel)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    _finish(fig, title, drawing.test_note + " " + STEP_ZERO_NOTE + note)
    drawing.save(fig, name)


def lines_variance_shares(drawing: Drawing) -> None:
    """Language and document shares of each vector."""

    _variance_grid(
        drawing,
        keys={"language": "language_share", "document": "document_share"},
        ylabel="share of the variance",
        name="lines_variance_shares",
        title="Variance shares against step. A share can fall because another part grew: read it with the absolute figure",
        log=False,
        neither=True,
        note=(
            " Neither = 1 - language - document: what is left to one document in one language."
            f" The document share has a floor near 1/{len(drawing.langs)} even on pure noise, since each document's mean over"
            f" {len(drawing.langs)} texts keeps 1/{len(drawing.langs)} of the noise's variance."
        ),
    )


def lines_absolute_variance(drawing: Drawing) -> None:
    """Language and document variance per text of each vector."""

    _variance_grid(
        drawing,
        keys={"language": "variance_language", "document": "variance_document"},
        ylabel="variance per text",
        name="lines_absolute_variance",
        title="Absolute variance against step",
        log=True,
        note="",
    )


def lines_distance_to_final(drawing: Drawing) -> None:
    """W and both LoRAs against the final checkpoint."""

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    parts = (("W", "W"), ("encoder_BA", "encoder LoRA, BA"), ("decoder_BA", "decoder LoRA, BA"))
    for ax, (key, label) in zip(axes, parts):
        for model in MODELS:
            weights, steps = drawing.results["models"][model]["weights"], drawing.steps(model)
            ax.plot(steps, _series([weights[str(step)][key]["relative"] for step in steps]), color=MODEL_COLOR[model], marker="o" if model == "h" else "s", ms=4, lw=1.5, label=MODEL_LABEL[model])
        _shared_axis(drawing, ax, narrow=True)
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel("distance to the final checkpoint,\nover its size")
    axes[-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    _finish(fig, "Weights against the final checkpoint", "Weights only, no documents involved. BA is without the factor alpha / rank = 2.")
    drawing.save(fig, "lines_distance_to_final")


def train_vs_test_topic(drawing: Drawing) -> None:
    """Topic score on English training posts and on English test documents."""

    meta = drawing.meta
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        rows, steps = drawing.rows(model), drawing.steps(model)
        test = [rows[str(step)]["topic"]["en"] / drawing.num_docs for step in steps]
        train = [rows[str(step)]["train_english"]["right"] / rows[str(step)]["train_english"]["total"] for step in steps]
        ax.plot(steps, test, color=TEST_COLOR, marker="o", ms=4, lw=1.5, label=f"English test ({drawing.num_docs:,} documents)")
        ax.plot(steps, train, color=TRAIN_COLOR, marker="^", ms=4, lw=1.5, label=f"English training ({meta['train_sample_size']:,} posts)")
        _model_axis(drawing, ax, model)
        ax.set(title=MODEL_LABEL[model], ylim=(0, 1))
    axes[0].set_ylabel("share with the teacher's top topic")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    note = (
        f"{meta['train_sample_size']:,} English training posts and {drawing.num_docs:,} English test documents, {FLOAT_NOTE}. "
        f"Test documents are the first 150 words (median {meta['median_words_test_english']:.0f}); "
        f"training posts run up to {meta['encoder_max_tokens']} tokens (median {meta['median_words_train_sample']:.0f} words), so length differs too."
    )
    _finish(fig, "Training against test, English", note)
    drawing.save(fig, "train_vs_test_topic")


def draw_everything(*, paths: TrajectoryPaths) -> None:
    """Every figure, from what the cache holds."""

    drawing = Drawing(paths=paths)
    for model, vector in GRIDS:
        for color_by in ("language", "topic"):
            for kind in ("fixed", "aligned", "tsne"):
                map_grid(drawing, model=model, vector=vector, kind=kind, color_by=color_by)
        train_overlay_figure(drawing, model=model, vector=vector)
    h_off_figure(drawing, kind="pca")
    h_off_figure(drawing, kind="tsne")
    lines_topic(drawing)
    lines_retrieval(drawing)
    lines_language_probe(drawing)
    lines_variance_shares(drawing)
    lines_absolute_variance(drawing)
    lines_delta_ratio(drawing)
    lines_distance_to_final(drawing)
    train_vs_test_topic(drawing)
