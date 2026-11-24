# %% [markdown]
# # Training: from raw 20 Newsgroups to the two topic-aligned VAEs
#
# **The question.** LDA gives every document a topic mixture $\theta$, but it can only read the English words it was
# fit on. Can a language model learn to produce LDA's $\theta$ from the text alone, so that it keeps working on text
# LDA cannot read?
#
# | step | what | on one GPU node |
# |---|---|---|
# | 1 | the corpus | CPU only, a few minutes for the check |
# | 2 | the teacher: LDA on OCTIS's preprocessing | CPU only; loading is instant |
# | 3 | the translated test set | seconds |
# | 4 | the model: one batch pushed through it | a minute |
# | 5 | training $\mu$ from $h$ and $\mu$ from $\Delta h$ | 24 and 27 min in the final runs |
# | 6 | the training curves and per-epoch test scores | seconds |
#
# Nothing here looks at the test set to choose anything. The code is the package; this notebook only calls it in
# order. `notebooks/inference.ipynb` is the second half. To run a step from a terminal instead:
# `python -m experiments.factorization.train --config configs/paper/factorization_delta.json`.

# %%
import os
import sys
from pathlib import Path

if Path.cwd().name == "notebooks":          # commands and configs read paths relative to main/
    os.chdir("..")
sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F

from core.factorization import TopicVAE
from core.runs.io import read_json
from core.train import resolve_device
from experiments.data.corpus import check_or_write_corpus, load_corpus
from experiments.data.paths import DataPaths
from experiments.data.teacher import LDATeacher, build_octis_counts
from experiments.data.translations import LANG_NAMES, LANGS, check_alignment, check_sample, load_translations
from experiments.factorization import figures
from experiments.factorization.config import train_config_from_file
from experiments.factorization.train import train_or_load

# The only things to change. configs/small_cpu_example has the same two files with a few texts and short sequences.
# "factorization" saves no checkpoints; CONFIG_PREFIX = "checkpointed" trains the same recipes with them.
CONFIG_DIR = Path(os.environ.get("VAE_CONFIG_DIR", "configs/paper"))
CONFIG_PREFIX = os.environ.get("VAE_CONFIG_PREFIX", "factorization")
CONFIGS = {"h": CONFIG_DIR / f"{CONFIG_PREFIX}_h.json", "delta": CONFIG_DIR / f"{CONFIG_PREFIX}_delta.json"}
CHECK_CORPUS = os.environ.get("VAE_CHECK_CORPUS", "1") == "1"   # step 1: rebuild from the raw download and compare

train_configs = {name: train_config_from_file(path) for name, path in CONFIGS.items()}
paths = DataPaths(data_dir=train_configs["h"].data_dir)
device = resolve_device(train_configs["h"].device)
print(f"device {device}, torch {torch.__version__}")
for name, config in train_configs.items():
    print(f"{name:6s} goes to {config.output_dir}\n       {config.recipe}")

# %% [markdown]
# ## 1. The corpus: what goes in?
#
# 20 Newsgroups with scikit-learn's split by date and `remove=("headers", "footers", "quotes")`, each post cleaned
# (separator lines, e-mail addresses, PGP blocks, whitespace) and kept if more than 50 characters are left. Training
# posts that also appear in the test set are dropped: byte-identical, or at least 90% of their 8-word shingles inside
# one test post.

# %%
if CHECK_CORPUS:
    check_or_write_corpus(paths=paths)
corpus = load_corpus(paths=paths)
words_per_post = np.array([len(text.split()) for text in corpus.train_texts])
print(f"{len(corpus.train_texts):,} training posts, {len(corpus.test_texts):,} test posts, {len(corpus.label_names)} newsgroups; "
      f"words per training post: median {np.median(words_per_post):.0f}, max {words_per_post.max():,}")

# %% [markdown]
# ## 2. The teacher: what is the target?
#
# LDA with $K = 20$ on OCTIS's preprocessing and 1,612-word vocabulary. Each document gets $\theta_d$, 20 numbers that
# sum to 1, and the model is trained to reproduce it. Posts with fewer than 5 vocabulary words carry no topic and stay
# out of the fit. Loading the saved teacher is instant; `python -m experiments.data.teacher --config
# configs/paper/teacher.json` refits it into `outputs/teacher/`.

# %%
octis = build_octis_counts(paths=paths, train_texts=corpus.train_texts)
print("raw:  ", repr(corpus.train_texts[0][:150]))
print("OCTIS:", repr(octis.words[0][:150]))
print(f"{octis.exact_octis_documents} of our {octis.long_enough_posts} posts with >= 5 vocabulary words are word for word an OCTIS document")
print(f"counts {octis.counts.shape}, {octis.counts.nnz:,} non-zero; median {np.median(octis.words_per_post):.0f} vocabulary words per post; "
      f"{octis.fit_rows.sum()} of {len(corpus.train_texts)} go into the fit")

teacher = LDATeacher(paths=paths)
for k, words in enumerate(teacher.top_words(8)):
    print(f"topic {k:2d}  {words}")
theta_train = teacher.theta(corpus.train_texts)
print(f"\ntheta {theta_train.shape}; largest share: median {np.median(theta_train.max(1)):.2f}, "
      f"above 0.5 for {(theta_train.max(1) > 0.5).mean():.0%} of the posts")

# %% [markdown]
# ## 3. The translated test set: how is "text LDA cannot read" made?
#
# 1,000 test posts, stratified by newsgroup, first 150 words, translated sentence by sentence with Marian. Document
# $i$ is the same post in every language, so each translation has a known answer: LDA's $\theta$ of its English
# original. The file is loaded, never rebuilt in place.

# %%
translated = load_translations(paths=paths)
check_sample(translated=translated, corpus=corpus)
check_alignment(translated=translated, corpus=corpus)
print({lang: len(translated.texts[lang]) for lang in LANGS})
print("per newsgroup", np.bincount(corpus.test_labels[translated.test_index], minlength=len(corpus.label_names)).tolist())
for lang in LANGS:
    print(f"{LANG_NAMES[lang]:8s} {translated.texts[lang][0][:140]}")

# %% [markdown]
# ## 4. The model: what is trained?
#
# Two frozen copies of the language model, each with a LoRA on its attention projections. The **posterior**
# $q(z \mid y) = \mathcal N(\mu, \mathrm{diag}\,\sigma^2)$ reads the document and two linear maps of a readout $u$ give
# the moments; $\hat\theta = \mathrm{softmax}(\mu)$. The readout is $h$, the state at the last token with the LoRA on,
# or $\Delta h = h_{\text{on}} - h_{\text{off}}$, what the LoRA adds. The **router** is the fixed $\mathcal N(0, I)$.
# The **generator** rewrites the document's first 128 tokens after one prefix vector made from $z$.
#
# $$\mathcal L = \frac{1}{\bar T}\Big[-\log p(y \mid z) + \beta\,\mathrm{KL}\big(q(z \mid y)\,\|\,\mathcal N(0, I)\big)\Big] - \lambda \sum_k \theta_k \log\hat\theta_k,\qquad \beta = 0.5,\ \lambda = 1.$$

# %%
demo = TopicVAE(config=train_configs["delta"].recipe.model).to(device).eval()
groups = demo.trainable_parameter_groups()
for group, parameters in groups.items():
    print(f"{group:14s} {sum(p.numel() for p in parameters):>12,}  in {len(parameters)} tensors")
print(f"{'frozen':14s} {sum(p.numel() for p in demo.parameters() if not p.requires_grad):>12,}")

texts = translated.texts["en"][:2]
theta = torch.tensor(teacher.theta(texts), dtype=torch.float32, device=device)
with torch.no_grad():
    input_ids, attention_mask = demo.tokenize(texts=texts, max_tokens=demo.config.encoder_max_tokens)
    readout = demo.readout(texts=texts)
    posterior = demo.posterior(texts=texts)
    reconstruction = demo.reconstruction_nll(z=posterior.mu, texts=texts)  # z = mu here: deterministic, for looking
    objective = demo.training_objective(texts=texts, theta=theta)           # the training loss, with z sampled
print(f"ids {tuple(input_ids.shape)}, real tokens per text {attention_mask.sum(1).tolist()}")
print(f"readout {tuple(readout.shape)}, max |readout| = {readout.abs().max().item():.3g}  (B = 0 at the start, so Δh is 0)")
print(f"mu {tuple(posterior.mu.shape)}, logvar in [{posterior.logvar.min().item():.2f}, {posterior.logvar.max().item():.2f}]")
print(f"theta_hat of text 0: {F.softmax(posterior.mu, -1)[0].cpu().numpy().round(3)}")
print(f"nll per token {(reconstruction.nll / reconstruction.num_tokens).tolist()}")
print(f"loss {objective.loss.item():.3f}: rec/token {objective.rec:.3f}, align {objective.align:.3f}, rate {objective.rate:.1f}")
del demo
if device == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Training: how is a model trained, and which weights are kept?
#
# AdamW over three groups (encoder LoRA $5\cdot10^{-5}$, decoder LoRA $2\cdot10^{-4}$, linear maps $10^{-3}$), warm-up
# over 5% of the steps and cosine decay to 0, gradient norm clipped at 2, and a step with a non-finite loss or gradient
# skipped. After every epoch the held-out loss and, to watch only, the test scores. The epoch with the lowest held-out
# loss is saved as `model.pt`, `history.json` and `test.json`.
#
# A folder that already holds a `model.pt` is loaded here after checking its recipe, never overwritten.
#
# **Checkpoints.** The optional run setting `checkpoint_steps` also saves the trained tensors at those steps, plus step 0
# and the last step, into `checkpoints/`. It is not part of the recipe and draws no random numbers. By default this notebook
# uses `factorization_*.json`, which saves none. `CONFIG_PREFIX = "checkpointed"` at the top uses
# `configs/paper/checkpointed_h.json` and `checkpointed_delta.json`, which do.

# %%
histories = {}
for name, config in train_configs.items():
    print(f"\n=== {name}: {config.output_dir} ===")
    model, histories[name] = train_or_load(config=config)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"{name}: {len(histories[name])} steps in history.json")

# %% [markdown]
# ## 6. What happened during training?
#
# The two loss parts the optimiser sees, and the per-epoch test scores, which never pick anything. The dotted line marks
# the epoch kept by the held-out loss alone.

# %%
curves = {}
for name, config in train_configs.items():
    test_rows = read_json(config.output_dir / "test.json")
    curves[{"h": "μ from h", "delta": "μ from Δh"}[name]] = (histories[name], test_rows)
    kept = min(test_rows[1:], key=lambda row: row["val"])
    print(f"{name}: kept epoch {kept['epoch']}")
    for row in test_rows[1:]:
        print(f"   epoch {row['epoch']}: held-out loss {row['val']:.3f}   test rec/token {row['rec']:.3f} align {row['align']:.3f}   "
              f"top topic right: English {row['en']}, German {row['de']}")
figures.plot_training(runs=curves, num_docs=translated.num_docs, output_dir="outputs", name=None, ylim=None)

# %% [markdown]
# **Next:** `notebooks/inference.ipynb` computes everything the paper reports.
