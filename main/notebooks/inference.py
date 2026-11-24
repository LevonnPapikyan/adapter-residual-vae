# %% [markdown]
# # Topics where LDA cannot read: inference
#
# **The question.** Can a language model trained on English alone produce LDA's $\theta$ on text LDA cannot read, the
# same test documents machine-translated into German, French, Spanish and Chinese?
#
# **What this notebook does.** Nothing is trained except two linear baselines. It loads the weights its config names,
# computes every score, baseline and figure, writes them to that config's `output_dir`, and compares `results.json` with
# the shipped one. Section 0 prints which weights and which folder. By default these are the saved weights in `../models/final_0.5b`
# and `outputs/evaluation/`; `configs/paper/evaluation_checkpointed.json` is the retrained pair and
# `outputs/checkpointed/evaluation/`. A folder that already holds a `results.json` is refused unless you ask to replace it.
# Run section 0 first; 6, 7 and 11 use section 5. Section 10 is the slow part. From a terminal, all of it:
# `python -m experiments.factorization.evaluate --config configs/paper/evaluation.json`.

# %% [markdown]
# ## 0. Setup
#
# The config at the top is the only thing to change: `configs/paper/evaluation_checkpointed.json` is the retrained pair
# with checkpoints.

# %%
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

if Path.cwd().name == "notebooks":          # commands and configs read paths relative to main/
    os.chdir("..")
sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers.utils import logging

from core.analysis import most_common_floor
from experiments.data.teacher import octis_words_cached
from experiments.data.translations import LANG_NAMES, LANGS
from experiments.factorization import evaluate, figures
from experiments.factorization.config import evaluation_config_from_file
from experiments.factorization.output import (
    LABEL_DELTA,
    LABEL_H,
    print_row,
    refuse_existing_results,
    write_comparison,
    write_environment,
    write_results,
)

logging.disable_progress_bar()

CONFIG = os.environ.get("VAE_EVAL_CONFIG", "configs/paper/evaluation.json")
OVERWRITE = os.environ.get("VAE_OVERWRITE_OUTPUT") == "1"   # True: replace the results.json and figures in the config's output_dir
DOC = evaluate.FOLLOWED_DOC                  # one document to follow through the notebook

config = evaluation_config_from_file(CONFIG, overwrite=OVERWRITE)
output_dir = config.output_dir
refuse_existing_results(output_dir=output_dir, overwrite=config.overwrite)
print(f"weights scored: {config.runs_dir}")
print(f"written to:     {output_dir}")
print(f"device {evaluate.resolve_device(config.device)} | torch {torch.__version__} | {asdict(config)}")

# %% [markdown]
# The data, the teacher and the three models (the two trained ones and plain Qwen), with the one check everything
# rests on: document $i$ is the same post in every language.

# %%
bundle = evaluate.load_evaluation_bundle(config=config)
corpus, translated, teacher, target = bundle.corpus, bundle.translated, bundle.teacher, bundle.target
N = bundle.num_docs

# %% [markdown]
# ## 1. The data: what is scored?
#
# 20 Newsgroups, cleaned, with leaked training posts removed. 1,000 test posts, stratified by newsgroup, first 150
# words, translated sentence by sentence. Every translation's answer is the teacher's $\theta$ of its English original.

# %%
print("documents per newsgroup:")
for group, count in zip(corpus.label_names, np.bincount(corpus.test_labels[translated.test_index], minlength=len(corpus.label_names))):
    print(f"  {count:3d}  {group}")
print(f"\ndocument {DOC} is test post {translated.test_index[DOC]} ({corpus.label_names[corpus.test_labels[translated.test_index[DOC]]]}):")
for lang in LANGS:
    print(f"{LANG_NAMES[lang]:8s} {translated.texts[lang][DOC][:140]}")

# %% [markdown]
# ## 2. The teacher: what does LDA say the answer is?
#
# $K = 20$ topics on OCTIS's preprocessing and 1,612-word vocabulary. Here is what LDA sees of one post, its topics by
# their most likely words, and $\theta$ of that post.

# %%
vocabulary = set(teacher.vectorizer.get_feature_names_out())
words = octis_words_cached([translated.texts["en"][DOC]], paths=teacher.paths)[0]
print("raw text:      ", translated.texts["en"][DOC][:300], "...\n")
print("after OCTIS:   ", words[:300], "...\n")
print("what LDA reads:", " ".join(word for word in words.split() if word in vocabulary))
for k, top in enumerate(teacher.top_words(8)):
    print(f"topic {k:2d}  {top}")
figures.plot_theta_example(theta=target[DOC], top_words=teacher.top_words(4), doc=DOC, output_dir=output_dir)

# %% [markdown]
# **Can LDA read the translations?** Its vocabulary is English: on German text it sees only the few words that look
# English.

# %%
for lang in LANGS:
    print(f"{LANG_NAMES[lang]:8s} median words LDA can read: {np.median(np.asarray(teacher.counts(translated.texts[lang]).sum(1)).ravel()):.0f}")

# %% [markdown]
# ## 3. The model: $h$ or $\Delta h$?
#
# The posterior reads the document; $\mu$ is a linear map of $h$, or of $\Delta h = h_{\text{on}} - h_{\text{off}}$, what
# the LoRA adds to plain Qwen. That readout is the only difference between the two models. Every setting comes from
# the run's own `model.pt`. What was trained:

# %%
for name, model in bundle.models.items():
    groups = model.trainable_parameter_groups()
    counts = {group: sum(p.numel() for p in parameters) for group, parameters in groups.items()}
    print(f"{name:10s} {sum(len(parameters) for parameters in groups.values())} trained tensors: "
          f"encoder LoRA {counts['encoder_lora'] / 1e6:.2f}M, decoder LoRA {counts['decoder_lora'] / 1e6:.2f}M, "
          f"linear maps {counts['linear_maps'] / 1e6:.2f}M   (width {model.hidden_size})")

# %% [markdown]
# ## 4. Training: did the loss go down, and which epoch was kept?

# %%
evaluate.summarize_training(bundle=bundle)

# %% [markdown]
# ## 5. Score 1: is the top topic right?
#
# For each document, is the predicted top topic on the (translated) text the teacher's top topic on the English text?
# Predictors: always the most common topic; LDA reading the translation; an LDA trained on German (read from
# `../models/baselines/report_german_lda.json`; rerun with `python -m experiments.baselines.lda_german`); plain Qwen with a
# linear map; and the two models.

# %%
topic_scores = evaluate.score_topics(bundle=bundle)
figures.plot_topic_score(scores=topic_scores.scores, majority=bundle.majority, num_docs=N, output_dir=output_dir)

# %% [markdown]
# **Is the top topic too strict?** The teacher's own $\theta$ puts only part of its mass on its top topic. So: is the
# teacher's top topic among the model's $k$ largest?

# %%
topk = evaluate.topk_tables(bundle=bundle, topic_scores=topic_scores)
figures.plot_topk(topk=topk, floor=most_common_floor(target=target), num_docs=N, output_dir=output_dir)

# %% [markdown]
# **What does $\hat\theta$ look like for one document?** One document in five languages, both models; the black bar is
# the teacher's top topic on the English original.

# %%
trained_thetas = {name: topic_scores.thetas[name] for name in (LABEL_H, LABEL_DELTA)}
top_topic = int(target[DOC].argmax())
figures.plot_one_document(thetas=trained_thetas, top_topic=top_topic, doc=DOC, output_dir=output_dir)
print(f"teacher's top topic on the English original: {top_topic} ({teacher.top_words(4)[top_topic]})")
for name, per_lang in trained_thetas.items():
    print(f"{name:10s} top topic per language: " + "  ".join(f"{LANG_NAMES[lang]} {per_lang[lang][DOC].argmax()}" for lang in LANGS))

# %% [markdown]
# ## 6. Score 2: does the topic mixture know the language?
#
# A logistic regression sees only the 20 numbers $\hat\theta$ and guesses the language: 5-fold cross-validation, all
# versions of a document in the same fold. Chance is 0.20. Lower is better.

# %%
lang_scores = evaluate.score_language(topic_scores=topic_scores)
figures.plot_language_score(lang_scores=lang_scores, output_dir=output_dir)

# %% [markdown]
# ## 7. The latent space: topics or languages?
#
# All texts by their latent, laid out with t-SNE. Top: coloured by the teacher's topic of the English original. Bottom:
# coloured and shaped by language.

# %%
figures.plot_latent_space(thetas=trained_thetas, topic=bundle.topic, text_lang=bundle.index.text_lang,
                          top_words_two=teacher.top_words(2), output_dir=output_dir)

# %% [markdown]
# ## 8. Why $\Delta h$ works: what is inside the encoder?
#
# Take a translated text and find the most similar English text by cosine similarity: if the representation is about
# the content, it finds its own original. Also the share of the variance explained by the language and by the document.

# %%
inside = evaluate.look_inside(bundle=bundle)

# %% [markdown]
# ## 9. Score 3: does the decoder write about the topic?
#
# For each LDA topic: sample $z$ for every English test post the teacher puts there, average them, let the generator
# write 10 texts from the average, and ask the teacher where they belong. Chance is about 1 in 20.

# %%
gen_scores = {}
if not config.skip_generation:
    generated, gen_scores = evaluate.score_generation(bundle=bundle)
    figures.plot_generation_score(gen_scores=gen_scores, num_topics=teacher.num_topics, per_topic=config.generation_per_topic, output_dir=output_dir)
    for topic in sorted(generated[LABEL_DELTA])[:3]:
        print(f"--- written from the average z of topic {topic} ({teacher.top_words(4)[topic]}):\n{generated[LABEL_DELTA][topic][0][:300]}\n")

# %% [markdown]
# ## 10. Where is the language inside Qwen?
#
# Plain Qwen, the last-token state after every block: a linear map to $\theta$ fit on English and scored in every
# language, the language score of that map, and finds-original on the state itself. The slow section.

# %%
layers = []
if not config.skip_layers:
    layers = evaluate.scan_layers(bundle=bundle)
    figures.plot_layers(layers=layers, majority=bundle.majority, num_docs=N, output_dir=output_dir)

# %% [markdown]
# ## 11. Summary: do we reproduce the shipped numbers?
#
# Everything in one table, then `results.json` with the shipped file's keys, then this run minus the shipped file. The
# two models' topic and language numbers and LDA's involve no random draw, but rounding differs by GPU type, so even the
# saved weights move by a few documents on another GPU. For a retrained pair the differences also hold the retraining. The
# frozen map and generation do draw at random. `environment.json` names the GPU of this run.

# %%
print_row("", list(bundle.models))
for lang in LANGS:
    print_row(f"top topic right, {LANG_NAMES[lang]}", [topic_scores.scores[name][lang] for name in bundle.models])
print_row("language from θ̂ (lower is better)", [f"{lang_scores[name]:.2f}" for name in bundle.models])
if gen_scores:
    print_row("written texts in their topic", [sum(gen_scores[name].values()) for name in bundle.models])

results = evaluate.assemble_results(bundle=bundle, topic_scores=topic_scores, lang_scores=lang_scores,
                                    gen_scores=gen_scores, inside=inside, layers=layers)
write_results(output_dir=output_dir, results=results)
write_environment(output_dir=output_dir, device=bundle.device)
comparison = write_comparison(output_dir=output_dir, results=json.loads(json.dumps(results, ensure_ascii=False)),
                              reference_path=config.reference_results, is_partial=config.is_partial)

# %% [markdown]
# **Caveats.** One seed per model; the paper reports the mean over five seeds. The other languages are machine translations of the test posts, not native text.
# "Right" means "agrees with LDA", including LDA's own mistakes.

# %% [markdown]
# ## 12. How the space forms during training
#
# The checkpoints of the pair trained with `configs/paper/checkpointed_{h,delta}.json`, and what
# `python -m experiments.factorization.trajectory --config configs/paper/trajectory.json` made of them, all read from
# `outputs/checkpointed/` whichever config is at the top of this notebook. Scores and maps are read from that job's results;
# nothing is encoded again except the one followed document at the chosen step. Each cell skips itself with a message when
# its files do not exist yet. See "Paper experiment configurations" in the README.
#
# The weights were trained in bfloat16. Everything here is float32, because at small Δh the bfloat16 Δh and μ are mostly
# rounding. Rounding is largest there but not gone at the kept step: the table below gives both formats. One seed.

# %%
import pandas as pd
from IPython.display import Image, display

from core.analysis.encoding import run_encoder_in_float32, topic_mixtures
from core.factorization.checkpointing import list_checkpoints, load_checkpoint, load_trained_run
from core.runs.io import read_json
from experiments.factorization.trajectory_cache import MODELS, TrajectoryPaths

CHECKPOINT_STEP = 100                        # the one setting of this section: which saved step to load below
TRAJECTORY_DIR = Path("outputs/checkpointed")

trajectory_paths = TrajectoryPaths(output_dir=TRAJECTORY_DIR)
checkpoints = {name: list_checkpoints(run_dir=TRAJECTORY_DIR / name) for name in MODELS}
for name, saved in checkpoints.items():
    print(f"{name}: {len(saved)} checkpoints, steps {list(saved)}" if saved else f"{name}: no checkpoints in {TRAJECTORY_DIR / name}, nothing to list")
if config.runs_dir.resolve() != TRAJECTORY_DIR.resolve():
    print(f"NOTE: sections 0 to 11 scored {config.runs_dir}. This section reads the retrained pair in {TRAJECTORY_DIR}, which are different weights.")

# %% [markdown]
# **One checkpoint.** Load the chosen step into each model and read the followed document in all five languages. Rows are
# languages, columns are topics, and the last row is the teacher's θ on the English original. The check line compares this
# with the θ̂ the trajectory job cached for the same document, which ran in larger batches.

# %%
for name, saved in checkpoints.items():
    if CHECKPOINT_STEP not in saved:
        print(f"{name}: no checkpoint at step {CHECKPOINT_STEP} (saved: {list(saved)}), skipping")
        continue
    run = load_trained_run(run_dir=TRAJECTORY_DIR / name, device=evaluate.resolve_device(config.device))
    run_encoder_in_float32(run.model)
    load_checkpoint(model=run.model, path=saved[CHECKPOINT_STEP])
    theta_at_step = topic_mixtures(model=run.model, texts=[translated.texts[lang][DOC] for lang in LANGS], batch_size=1)
    table = pd.DataFrame(theta_at_step, index=[LANG_NAMES[lang] for lang in LANGS])
    table.loc["LDA on English"] = target[DOC]
    print(f"{name} model, step {CHECKPOINT_STEP}, document {DOC}, float32. Top topic: " + "  ".join(f"{LANG_NAMES[lang]} {theta_at_step[i].argmax()}" for i, lang in enumerate(LANGS)) + f"  LDA {target[DOC].argmax()}")
    display(table.round(3))
    cached_path = trajectory_paths.states(name, "float32", CHECKPOINT_STEP)
    if cached_path.exists():
        with np.load(cached_path) as cached:
            cached_theta = torch.softmax(torch.from_numpy(cached["mu"][[i * N + DOC for i in range(len(LANGS))]]), -1).numpy()
        print(f"largest difference from the cached theta_hat: {np.abs(cached_theta - theta_at_step).max():.2e}")
    else:
        print(f"no cached states at {cached_path} yet, so no check against them")
    del run

# %% [markdown]
# **Scores at every checkpoint**, float32, from `outputs/checkpointed/results.json`. Topic is out of 1,000 per language,
# retrieval is how many of the 1,000 find their English original, and the probe and the two variance shares are of what
# the model reads (h for the h model, Δh for the Δ model). The bfloat16 rows are in the same file.

# %%
if not trajectory_paths.results.exists():
    print(f"no {trajectory_paths.results} yet: skipping the table")
else:
    trajectory_results = read_json(trajectory_paths.results)
    for name, entry in trajectory_results["models"].items():
        rows = []
        for step in entry["steps"]:
            row = entry["rows"]["float32"][str(step)]
            read_vector = row["vectors"][row["read"]]
            rows.append({
                "step": step,
                **{f"topic {lang}": row["topic"][lang] for lang in LANGS},
                "probe": row["language_probe"],
                **{f"retrieval {lang}": (row["retrieval"] or {}).get(lang) for lang in LANGS[1:]},  # not defined where Δh is still zero
                "language share": read_vector.get("language_share"),
                "document share": read_vector.get("document_share"),
            })
        print(f"{name} model, reads {entry['read']}, kept step {entry['kept_step']} (epoch {entry['kept_epoch']}), final step {entry['final_step']}")
        display(pd.DataFrame(rows).set_index("step").round(3))


# %% [markdown]
# **The kept step in both float formats.** The lines below are float32, as the table above. Section 8 above, section 11 and
# the shipped `results.json` are bfloat16, as trained, so a line at the dotted step does not equal the number there. This is
# the size of that gap, from `results.json`, both in batches per language. Section 8 batches differently, so it can differ from
# the bfloat16 row here by a few documents.

# %%
if trajectory_paths.results.exists():
    for name, entry in trajectory_results["models"].items():
        rows = {}
        for precision in ("float32", "bfloat16"):
            row = entry["rows"][precision][str(entry["kept_step"])]
            read_vector = row["vectors"][row["read"]]
            rows[precision] = {
                **{f"topic {lang}": row["topic"][lang] for lang in LANGS},
                "probe": row["language_probe"],
                **{f"retrieval {lang}": row["retrieval"][lang] for lang in LANGS[1:]},
                "language share": read_vector["language_share"],
                "document share": read_vector["document_share"],
            }
        print(f"{name} model, kept step {entry['kept_step']}")
        display(pd.DataFrame(rows).T.round(3))


# %%
def show_figures(*patterns: str) -> None:
    """Display the saved PNGs of the trajectory job that match, or say that there are none yet."""

    for pattern in patterns:
        found = sorted(trajectory_paths.figures.glob(pattern))
        if not found:
            print(f"no {pattern} in {trajectory_paths.figures} yet: skipping")
        for path in found:
            print(path.name)
            display(Image(filename=str(path)))


# %% [markdown]
# **Line plots**: scores, shares, the size of Δh and the distance of the weights to their final values, against the step.

# %%
show_figures("lines_*.png")

# %% [markdown]
# **PCA in one fixed basis.** Fitted at the last step on the test texts, every checkpoint projected into it, with the same
# axes in every panel, so panels of one grid can be compared directly. Each panel prints "in plane", the share of that
# vector's variance the plane shows. It differs a lot between vectors, so compare grids of different vectors only with it.

# %%
show_figures("pca_fixed_*.png")

# %% [markdown]
# **Plain Qwen's h_off**, once, in its own PCA basis, not the fixed basis above, so the same points look different at step 0
# of the h_on grid.

# %%
show_figures("pca_h_off.png")

# %% [markdown]
# **PCA refit at each checkpoint and aligned.** Each step gets its own PCA, turned onto the last step's by orthogonal
# Procrustes, and each panel has its own scale. A disparity near 1 means the alignment does little.

# %%
show_figures("pca_aligned_*.png")

# %% [markdown]
# **t-SNE at each checkpoint.** Caution: t-SNE distances and axes are not comparable across panels. Read each panel on its
# own, for what gathers with what, and do not read a change between panels as distance moved.

# %%
show_figures("tsne_*.png")

# %% [markdown]
# **Training versus test English.** These are not the same kind of text: test documents are the first 150 words and
# training posts run up to 512 tokens, so a gap can come from length as well as from having been trained on.

# %%
show_figures("train_vs_test_*.png")
