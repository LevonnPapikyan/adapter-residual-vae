"""Generation heatmap (two_latents grid): one panel with a judge-ceiling column, read from data/group_grid_scores.json.

    python plot_heatmaps.py

Cells: share of the 50 written texts of a (group, language) that the judge puts in the chosen group. The judge translates
the text back to English and takes LDA's top topic. "Judge ceiling": the same judge on real test documents of the
group, mean over the five languages. Row label: the group, its main LDA topic id and three words, and the other LDA
topics matched to it (share of the topic's training posts in the group >= ALSO). Writes heatmaps.{pdf,png}.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).parent
ALSO = 0.3
scores = json.loads((HERE / "data/group_grid_scores.json").read_text())
expected = scores["expected"]
groups = list(scores["group_right"])
langs = list(scores["group_right"][groups[0]])
share = {int(name.split(":")[0]): v["share of the topic's training posts in it"] for name, v in expected["each LDA topic's group"].items()}
CMAP = LinearSegmentedColormap.from_list("blue_orange", ["#2B6CA8", "#A9C9E2", "#FBF3DF", "#F2B061", "#D9781E"])  # colour-blind safe ends


def label(group):
    info = expected["each group's expected LDA topic"][group]
    main = int(info["topic"].split(":")[0])
    words = " ".join(info["topic"].split(": ")[1].split()[:3])
    also = [t for t in info["LDA topics that belong to the group"] if t != main and share[t] >= ALSO]
    return f"LDA id {main + 1}: {words}" + (f" ({', '.join(str(k + 1) for k in also)})" if also else "")  # topics are numbered 1 to 20 in the figures


def grid(key):
    return np.array([[scores[key][g][lang] for lang in langs] for g in groups])


written, real, language = grid("group_right"), grid("ceiling_group_right"), grid("language_right")
ceiling = real.mean(1)
rows = len(groups)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9.2, 7.4))
X0, GAP, MEAN_Y = 1.45, 0.55, rows + 0.55  # x of the first language column; gap above the mean row


def cell(x, y, value, bold=False, alpha=1.0):
    ax.add_patch(plt.Rectangle((x - 0.46, y - 0.42), 0.92, 0.84, color=CMAP(value), alpha=alpha, lw=0))
    ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold" if bold else "normal", color="white" if value < 0.2 or value > 0.9 else "#222222")


for r, group in enumerate(groups):
    cell(0, r, ceiling[r], alpha=0.75)
    for c in range(len(langs)):
        cell(X0 + c, r, written[r, c])
    ax.text(-0.62, r - 0.16, group, ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(-0.62, r + 0.2, label(group), ha="right", va="center", fontsize=8.5, color="0.35")
cell(0, MEAN_Y, ceiling.mean(), bold=True, alpha=0.75)
for c in range(len(langs)):
    cell(X0 + c, MEAN_Y, written[:, c].mean(), bold=True)
ax.text(-0.62, MEAN_Y, "Mean of 12 groups", ha="right", va="center", fontsize=11, fontweight="bold")
ax.text(0, -0.72, "Judge\nceiling", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
for c, lang in enumerate(langs):
    ax.text(X0 + c, -0.72, lang, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
ax.set_title("Generated text: LDA reads the chosen topic group", fontsize=12.5, fontweight="bold", pad=34, x=0.62)
ax.set_xlim(-0.5, X0 + len(langs) - 0.5); ax.set_ylim(MEAN_Y + 0.45, -0.5); ax.axis("off")
bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=CMAP), ax=ax, fraction=0.03, pad=0.02, shrink=0.45)
bar.outline.set_visible(False); bar.ax.set_title("LDA match\nshare", fontsize=9.5, fontweight="bold", loc="left", pad=8)
for suffix in ("pdf", "png"):
    fig.savefig(HERE / f"heatmaps.{suffix}", bbox_inches="tight", dpi=150)
print("written mean per language", np.round(written.mean(0), 3), "overall", written.mean().round(3), "| ceiling mean", ceiling.mean().round(3))
print("language right: cells at 1.00:", int((language == 1).sum()), "of", language.size, "| min", language.min(), "at", [(groups[r], langs[c]) for r, c in zip(*np.where(language == language.min()))])
