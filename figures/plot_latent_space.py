"""t-SNE of the latent mean of the paper's seed 0 pair, with the topic names in a key.

    python plot_latent_space.py

The same map as the paper's latent_space.pdf (rows of mu centred, PCA start, perplexity 30, seed 0); only the topic
names move from the plot to a key on the right, as in the two-latent figure. 1,000 test documents x 5 languages.
Reads data/mu_per_text.npz (vae/main/outputs/h_minus_base, each model's trained readout), data/lda_top.npz and
data/main_scores.json (two_latents: LDA's top topic of the English original, topic words). Writes latent_space.{pdf,png}.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

HERE = Path(__file__).parent
LANG_STYLE = {"en": ("English", "#000000", "o"), "de": ("German", "#E69F00", "s"), "fr": ("French", "#56B4E9", "^"), "es": ("Spanish", "#009E73", "D"), "zh": ("Chinese", "#CC79A7", "v")}
MODELS = (("adapter result ($\\mu$ from $h$)", "h|h_on"), ("adapter residual ($\\mu$ from $\\Delta h$)", "delta|h_on-h_off"))
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

mu = np.load(HERE / "data/mu_per_text.npz")
topic = np.load(HERE / "data/lda_top.npz")["lda_top"]  # 5 blocks of 1,000 in the order of LANG_STYLE
language = np.repeat(np.arange(5), 1000)
words = json.loads((HERE / "data/main_scores.json").read_text())["topic_words"]
names = [f"{k + 1}: {' '.join(w.split()[:2])}" for k, w in enumerate(words)]  # topics are numbered 1 to 20 in the figures

cache = HERE / "data/latent_space_xy.npz"
if cache.exists():
    maps = dict(np.load(cache))
else:
    maps = {}
    for _, key in MODELS:
        latent = np.concatenate([mu[f"{key}|{lang}"] for lang in LANG_STYLE]).astype(np.float64)
        maps[key] = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(latent - latent.mean(1, keepdims=True))
    np.savez(cache, **maps)

fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
for col, (name, key) in enumerate(MODELS):
    xy = maps[key]
    for k in range(len(names)):
        axes[0, col].scatter(*xy[topic == k].T, s=14, color=plt.cm.tab20(k), alpha=0.6, lw=0, label=names[k])
    for j, (label, color, marker) in enumerate(LANG_STYLE.values()):
        axes[1, col].scatter(*xy[language == j].T, s=14, color=color, marker=marker, alpha=0.6, lw=0, label=label)
    axes[0, col].set_title(name[0].upper() + name[1:], fontsize=15, pad=10)  # one title per column
for ax in axes.ravel():
    ax.set(xticks=[], yticks=[])
axes[0, 0].set_ylabel("Colored by LDA topic", fontsize=14, labelpad=10)  # one label per row
axes[1, 0].set_ylabel("Colored by language", fontsize=14, labelpad=10)
axes[0, 1].legend(markerscale=2, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 1), handletextpad=0.2, labelspacing=0.35, title="LDA topic", title_fontsize=11, alignment="left")
axes[1, 1].legend(markerscale=2, frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(1, 1), handletextpad=0.2, title="Language", title_fontsize=11, alignment="left")
fig.tight_layout()
for suffix in ("pdf", "png"):
    fig.savefig(HERE / f"latent_space.{suffix}", bbox_inches="tight", dpi=110)
for name, key in MODELS:
    top = np.concatenate([mu[f"{key}|{lang}"] for lang in LANG_STYLE]).argmax(1)
    print(key, "argmax mu = LDA top, per language:", [int((top == topic)[language == j].sum()) for j in range(5)])
