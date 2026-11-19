"""t-SNE of the two latents (two_latents), topics numbered 1 to 20.

    python plot_two_latents.py

The figure of extensions/two_latents/plot_tsne_for_slides.py with the same map (data/exp11_tsne_xy.npz); only the
topic numbers in the key start at 1. Reads data/exp11_main_features.npz and data/main_scores.json. Writes two_latents.{pdf,png}.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
LANG_STYLE = {"English": ("#000000", "o"), "German": ("#E69F00", "s"), "French": ("#56B4E9", "^"), "Spanish": ("#009E73", "D"), "Chinese": ("#CC79A7", "v")}

features = np.load(HERE / "data/exp11_main_features.npz", allow_pickle=True)
words = json.loads((HERE / "data/main_scores.json").read_text())["topic_words"]
names = [f"{k + 1}: {' '.join(w.split()[:2])}" for k, w in enumerate(words)]  # topics are numbered 1 to 20 in the figures
language, topic = features["language"], features["lda_top"]
maps = dict(np.load(HERE / "data/exp11_tsne_xy.npz"))

fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
for col, (key, name) in enumerate((("mu_topic", "topic latent"), ("mu_language", "language latent"))):
    xy = maps[key]
    for k in range(len(names)):
        axes[0, col].scatter(*xy[topic == k].T, s=14, color=plt.cm.tab20(k), alpha=0.6, lw=0, label=names[k])
    for j, (lang, (color, marker)) in enumerate(LANG_STYLE.items()):
        axes[1, col].scatter(*xy[language == j].T, s=14, color=color, marker=marker, alpha=0.6, lw=0, label=lang)
    axes[0, col].set_title(name[0].upper() + name[1:], fontsize=15, pad=10)  # one title per column
for ax in axes.ravel():
    ax.set(xticks=[], yticks=[])
axes[0, 0].set_ylabel("Colored by LDA topic", fontsize=14, labelpad=10)  # one label per row
axes[1, 0].set_ylabel("Colored by language", fontsize=14, labelpad=10)
axes[0, 1].legend(markerscale=2, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 1), handletextpad=0.2, labelspacing=0.35, title="LDA topic", title_fontsize=11, alignment="left")
axes[1, 1].legend(markerscale=2, frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1, 1), handletextpad=0.2, title="Language", title_fontsize=11, alignment="left")
fig.tight_layout()
for suffix in ("pdf", "png"):
    fig.savefig(HERE / f"two_latents.{suffix}", bbox_inches="tight", dpi=110)
