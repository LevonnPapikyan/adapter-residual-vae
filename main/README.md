# The model code: a VAE whose latent is read from the adapter residual

LDA gives each document a topic mixture $\theta$, but it only reads the English words it was fit on. This code trains a language model to produce LDA's $\theta$ from the text alone, on English only, and scores it on the same test documents machine-translated into German, French, Spanish and Chinese.

## The model

The model is a VAE over the document $y$. There is no prompt, so nothing conditions on an input $x$.

| part | what it is | in the code |
|---|---|---|
| posterior $q(z \mid y) = \mathcal N(\mu, \mathrm{diag}\,\sigma^2)$ | a frozen Qwen with a LoRA reads the document; two linear maps of a readout give $\mu$ and $\log\sigma^2$, and $\hat\theta = \mathrm{softmax}(\mu)$ | `TopicVAE.posterior`, `posterior_backbone`, `mu_head`, `logvar_head` |
| router $p(z) = \mathcal N(0, I)$ | fixed, no parameters | only the rate term in `core/factorization/objective.py` |
| generator $p(y \mid z)$ | a second frozen Qwen with its own LoRA writes the document after one prefix vector made from $z$ | `TopicVAE.reconstruction_nll`, `generator`, `z_to_prefix` |

The readout `mu_from` is the only difference between the two models of the paper: `"h"` is the backbone state at the last token with its LoRA on, and `"delta"` is $\Delta h = h_{\text{on}} - h_{\text{off}}$, what the LoRA adds to plain Qwen.

$$\mathcal L = \frac{1}{\bar T}\Big[-\log p(y \mid z) + \beta\,\mathrm{KL}\big(q(z \mid y)\,\|\,\mathcal N(0, I)\big)\Big] - \lambda \sum_k \theta_k \log\hat\theta_k$$

$\bar T$ is the mean number of tokens in the batch, $\lambda = 1$, $\beta = 0.5$, and $\theta$ is LDA's mixture for the document.

## Setup

From this folder, with Python 3.10 or newer:

```bash
pip install ".[test]"
export TOKENIZERS_PARALLELISM=false
```

`pyproject.toml` pins the versions the results were made with. The data and the trained weights are not in the repository; `../data/README.md` and `../models/README.md` say what goes where. Spacy needs `en_core_web_sm` and `de_core_news_sm` for texts missing from the word cache.

All commands run from this folder, and all paths in the configs are relative to it: `../data` and `../models` are read, and everything new goes to `outputs/`.

## Small CPU example

A few texts, short sequences and one epoch, on one CPU thread with the cached Qwen2.5-0.5B. The evaluation reads the saved weights in `../models/final_0.5b`.

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
python -m experiments.factorization.train --config configs/small_cpu_example/factorization_delta.json
python -m experiments.factorization.evaluate --config configs/small_cpu_example/evaluation.json
python -m pytest tests
```

## The two notebooks

`notebooks/train.ipynb` goes through corpus, teacher, translations, model, training and curves. It uses `factorization_{h,delta}.json`, which save no checkpoints; `VAE_CONFIG_PREFIX=checkpointed`, or `CONFIG_PREFIX` at its top, uses the configs that do.

`notebooks/inference.ipynb` loads the weights its config names and computes every score, baseline and figure into that config's `output_dir`, then compares `results.json` with `../figures/paper/results.json`. Its section 12 reads the checkpoints and the trajectory job's results and figures in `outputs/checkpointed/`, whichever config is at the top, and `CHECKPOINT_STEP` there chooses which step to load. Each notebook has its config at the top and imports everything else from the package.

The copy in `notebooks/inference.ipynb` holds the outputs of the retrained pair, and its first cell says so. To run it again on those weights:

```bash
mkdir -p outputs/checkpointed/logs
VAE_NOTEBOOK_OUTPUT=outputs/checkpointed/notebooks/inference.ipynb VAE_EVAL_CONFIG=configs/paper/evaluation_checkpointed.json VAE_OVERWRITE_OUTPUT=1 VAE_NOTEBOOK_FORCE=1 sbatch --output=outputs/checkpointed/logs/notebook.%j.out slurm/notebook.sbatch inference
```

Then copy that file over `notebooks/inference.ipynb` if you want it there. In an editor, set `CONFIG` in section 0 to `configs/paper/evaluation_checkpointed.json` and `OVERWRITE = True`. Section 0 prints which weights it scores.

Without that, the source's default is `configs/paper/evaluation.json`: the saved weights in `../models/final_0.5b`, written to `outputs/evaluation/`. Run All then stops at section 0 if `outputs/evaluation/results.json` exists. With `OVERWRITE = True` it replaces that folder, and sections 0 to 11 score the old weights while section 12 still reads the retrained pair; the notebook prints a note saying so.

The `.py` next to each notebook is its source. Edit that, then `python notebooks/build_notebooks.py`. A build has no outputs, so a notebook that has saved outputs is refused unless `--force`; `slurm/notebook.sbatch` does the same unless `VAE_NOTEBOOK_FORCE=1`. Copy an executed notebook elsewhere before forcing.

## Configuration

Training and evaluation take a JSON file through `--config PATH`. Command-line options override it, and an unknown key is an error.

- `base_model`, `num_topics`, `teacher`, `min_words`: the language model, the latent dimension, the LDA file in `data_dir`, and how many teacher words a training post needs.
- `mu_from`: the readout, `"h"` or `"delta"`. `"delta"` needs `train_encoder_lora`.
- `lam`, `beta`: the weights of the alignment and rate terms.
- `lora_rank`: LoRA rank for both copies; alpha is always twice the rank.
- `lr_encoder_lora`, `lr_decoder_lora`, `lr_linear_maps`: the three AdamW groups.
- `num_epochs`, `batch_size`, `val_frac`, `encoder_max_tokens`, `decoder_max_tokens`, `seed`: the rest of the recipe.
- `data_dir`, `output_dir`, `device`, `log_every`: where and how a run goes, not recorded in `model.pt`.
- `checkpoint_steps` (training): see "Checkpoints and the trajectory".
- `max_train_texts`, `max_test_docs`, `max_steps` (training), and `max_docs`, `max_train_texts`, `max_generation_texts`, `encoder_max_tokens`, `linear_map_steps`, `generation_per_topic`, `generation_max_new_tokens`, `skip_layers`, `skip_generation` (evaluation): for smoke tests. Left unset they change nothing.
- `overwrite` (evaluation): replace a `results.json` already in `output_dir`. Without it that is refused.

Four things the final runs fixed are not settings: no normalization of $h$, the rate counted per token, the prior $\mathcal N(0, I)$, and the warm-up cosine schedule.

Training refuses a folder that already holds a `model.pt`. No output may go into `../models`, `../data` or `../figures`.

## Paper experiment configurations

| config | what | command |
|---|---|---|
| `configs/paper/factorization_h.json`, `factorization_delta.json` | the two models on Qwen2.5-0.5B | `python -m experiments.factorization.train --config ...` |
| `configs/paper/factorization_h_1.5b.json`, `factorization_delta_1.5b.json` | the same on Qwen2.5-1.5B | same |
| `configs/paper/evaluation.json`, `evaluation_1.5b.json` | every score and figure of a pair | `python -m experiments.factorization.evaluate --config ...` |
| `configs/paper/teacher.json` | the LDA teacher | `python -m experiments.data.teacher --config ...` |
| `configs/paper/checkpointed_h.json`, `checkpointed_delta.json` | the two models with checkpoints | `python -m experiments.factorization.train --config ...` |
| `configs/paper/trajectory.json` | every checkpoint of that pair | `python -m experiments.factorization.trajectory --config ...` |
| `configs/paper/evaluation_checkpointed.json` | every score and figure of that pair | `python -m experiments.factorization.evaluate --config ...` |
| `configs/paper/evaluation_original.json`, `evaluation_original_h200.json` | the saved weights in `../models/final_0.5b`, into `outputs/checkpointed/`, to compare with the retrain on one GPU type (`--gres=gpu:a100:1`, `--gres=gpu:h200:1`) | same |

The recipe in each factorization config equals the one in the matching saved `model.pt`; `tests/test_config.py` checks this. The two LDA baselines are `python -m experiments.baselines.lda_german` and `python -m experiments.baselines.lda_pair`, and `python -m experiments.baselines.frozen_linear_map` scores the frozen map alone.

## Reproducing

From the saved weights, on one GPU:

```bash
mkdir -p outputs/logs
sbatch slurm/evaluate.sbatch configs/paper/evaluation.json --overwrite
VAE_OVERWRITE_OUTPUT=1 VAE_NOTEBOOK_OUTPUT=outputs/notebooks/inference_saved_weights.ipynb sbatch slurm/notebook.sbatch inference
```

`--overwrite` and `VAE_OVERWRITE_OUTPUT=1` are needed because `outputs/evaluation/results.json` already exists. Each writes `results.json` with the shipped file's keys, the figures under the names the paper includes, `comparison.json`, this run minus `../figures/paper/results.json`, and `environment.json`, the GPU and versions. The batch scripts ask for an A100 unless the sbatch line says otherwise.

The two models' topic and language numbers and LDA's involve no random draw, but rounding differs by GPU type, so they do move on another GPU. The saved weights on an A100 minus the shipped file: topic at most 2 of 1,000 for the h model and 7 for the $\Delta h$ model, language probe at most 0.001, $\Delta h$ retrieval 1 or 2 except Chinese at -16, and generation 0 and +5 of 200. The frozen linear map starts from seeded random weights and generation samples, so those can also move with hardware and software versions.

**Training GPU and scoring GPU differ.** The original weights were trained on an A100. The shipped `../figures/paper/results.json` records no GPU, but it looks scored on an H200: the same weights through the trajectory code on an H200 (`outputs/checkpointed/saved_check/`) give all 10 of its topic numbers, both probes and the Chinese retrieval exactly. So ask for an A100 to retrain, and to compare a retrain with the old weights, and do not expect an A100 to give the shipped scores exactly. This is inferred from that match; `configs/paper/evaluation_original_h200.json` tests it with the whole evaluation.

Retraining a model, 24 to 32 minutes on one GPU: `sbatch slurm/train.sbatch configs/paper/factorization_delta.json`. It writes `outputs/factorization_delta/`, the same three files as `../models/final_0.5b/delta/`, and no checkpoints. `tests/test_flat_equivalence.py` checks that weights, `history.json` and `test.json` are identical to those of the old flat code on a small run. That code is no longer next to `main/`, so those 5 tests skip unless `VAE_FLAT_ROOT` names a copy of its last revision, with a `data` folder or link beside it.

## Checkpoints and the trajectory

Only training through these two configs saves checkpoints. `configs/paper/checkpointed_h.json` and `checkpointed_delta.json` are the same recipes as `factorization_{h,delta}.json` with one more setting, `checkpoint_steps`. It saves only the trained tensors and the step at those steps, and always at step 0 before any update and at the last step, as `checkpoints/ckpt_{step}.pt`. A skipped step still counts as a step. It is a run setting, not part of the recipe; left out, it changes nothing. Saving draws no random numbers, and asserts that.

```bash
mkdir -p outputs/logs outputs/checkpointed/logs
sbatch --gres=gpu:a100:1 --output=outputs/checkpointed/logs/train_h.%j.out slurm/train.sbatch configs/paper/checkpointed_h.json
sbatch --gres=gpu:a100:1 --output=outputs/checkpointed/logs/train_delta.%j.out slurm/train.sbatch configs/paper/checkpointed_delta.json
sbatch --gres=gpu:a100:1 slurm/trajectory.sbatch configs/paper/trajectory.json        # after both, up to 16 hours; resubmit to resume
sbatch --partition=day --gres=none slurm/trajectory.sbatch configs/paper/trajectory.json --figures-only   # rebuild, no GPU
sbatch --gres=gpu:a100:1 slurm/evaluate.sbatch configs/paper/evaluation_checkpointed.json   # the paper's scores of the retrained pair
```

The notebook on these weights, with section 12 and the figures, is the command under "The two notebooks". A folder that already holds a `results.json` needs `--overwrite` after the config.

The originals were trained on an A100, and rounding differs by GPU type, so ask for one. Reruns diverge within a few steps even there, so a retrain is not bit-identical.

`experiments/factorization/trajectory.py` reads each checkpoint of each model on the 1,000 test documents in 5 languages and on a fixed sample of 1,000 English posts from the training split, in float32 (primary, for every figure) and bfloat16 (as trained, metrics only). At small $\Delta h$ bfloat16 is mostly rounding: $|\Delta h| / |h_{\text{off}}|$ has a floor of about 0.01 there. Rounding is not gone at the kept step: in the retrain at step 1650, $\Delta h$ retrieval is 34 to 72 of 1,000 lower in bfloat16 (744, 789, 875, 651 against 816, 843, 909, 690), and topic moves by up to 14 (German 576 against 566, Spanish 577 against 591). So a line figure at the dotted step does not equal the comparison tables, which are bfloat16; `lines_topic` and `lines_retrieval` print both. Texts are batched per language here and as one stacked list in `evaluate`, so bfloat16 $\Delta h$ retrieval also differs from `evaluate`'s by a few documents (744, 789, 875, 651 here, 748, 786, 876, 649 there). It writes under `outputs/checkpointed/`:

- `results.json`: per model, float format and step, the topic score per language, language probe, retrieval, shares and absolute variance of `h_off`, `h_on`, `delta`, `mu`, $|\Delta h| / |h_{\text{off}}|$ and the training sample's topic score; distances of $W$ and $BA$ to the final checkpoint; the kept step minus `../figures/paper/results.json` with the agreed tolerances. `saved_weights_check.json` is `../models/final_0.5b` through the same code.
- `features/`: the cache. States, one row of scores per checkpoint and the map coordinates. A state or row is reused, so delete it to compute it again. A map is rebuilt when its checkpoints' md5s change. Another `train_sample_seed` is refused: use another `output_dir`.
- `figures/`, PDF and PNG: `pca_fixed_*` (basis fitted at the final step), `pca_aligned_*` (refit per checkpoint, Procrustes to the final step), `tsne_*` (PCA-50, PCA start, perplexity 30, seed 0), each by language and by top topic; `pca_h_off`, `tsne_h_off`; `lines_*` against step; `train_vs_test_*`. `pca_h_off` has its own basis, not the fixed basis of the grids.

Training posts run up to 512 tokens while the test documents are their first 150 words, so the training and test comparison also compares lengths. $\Delta h$ and the $\Delta h$ model's $\mu$ at step 0 are the same for every text, so they get an empty panel and `null`, never noise. One seed.

Reading the maps: a 2D plane keeps very different amounts of each vector. In the fixed basis at the kept step it shows 11.6% of the variance of $\Delta h$, 35.4% of the $\Delta h$ model's $\mu$, 73.1% of $h_{\text{on}}$ and 36.4% of the $h$ model's $\mu$, so a shapeless $\Delta h$ panel says little about topic structure in $\Delta h$. Each PCA panel prints this as "in plane". The "full space" numbers beside it are measured in the whole vector, not on the plane. The two variance shares do not add up to 1: the rest is what belongs to one document in one language, and the document share has a floor near 1/5 even on noise.

## What one seed can say

The agreed tolerances (topic 15 of 1,000, probe 0.03, retrieval 20, variance share 2 points, generation 25 of 200) test whether a rerun reproduces seed 0. They are not the spread between seeds, which is far larger. Six trainings on A100s, seed 0 and `outputs/seeds/seed_{1..5}`, all scored the same way:

| | h model | $\Delta h$ model |
|---|---|---|
| topic, English | 667 to 697 | 622 to 651 |
| topic, German | 379 to 479 | 535 to 576 |
| topic, French | 486 to 544 | 549 to 582 |
| topic, Spanish | 385 to 524 | 550 to 587 |
| topic, Chinese | 519 to 602 | 555 to 587 |
| gap between its languages | 215 to 298 | 74 to 107 |
| language probe | 0.689 to 0.755 | 0.341 to 0.412 |
| language share | 0.760 to 0.780 | 0.043 to 0.052 |
| retrieval, any language | 4 to 214 | 581 to 879 |
| generation of 200 | 16 to 82 | 46 to 94 |

These hold in all six: the $\Delta h$ model beats the $h$ model on German (+68 to +175), French (+5 to +80) and Spanish (+61 to +167), loses on English (33 to 61), and has the narrower range, lower probe, lower language share and higher retrieval. Chinese does not hold either way: $\Delta h$ minus $h$ is -30, -2, +34, +64, +13, -12 for seeds 0 to 5. Generation does not hold either: $h$ beats $\Delta$ in seeds 1 and 5. The kept step varies too (1100, 1650 or 2200), so 1650 is not a constant.

## Outputs

- Training writes `model.pt` (`{"config", "weights"}` with only the trained tensors), `history.json` (loss parts per step taken) and `test.json` (per-epoch held-out loss and test scores).
- Evaluation writes `results.json`, `comparison.json`, `environment.json` and `theta_example.pdf`, `training.pdf`, `topic_score.pdf`, `topk.pdf`, `one_document.pdf`, `language_score.pdf`, `latent_space.pdf`, `generation_score.pdf`, `layers.pdf`.

A `model.pt` keeps the saved names of the final runs (`enc.*`, `dec.*`, `prefix.*`, `K`, `lora_r`, ...), so this code reads the old files. `core/factorization/checkpointing.py` is the one place that maps them to current names. The final files also hold an unused LayerNorm (`norm.*`), which is ignored; a model trained with it in use (`no_norm=False`) is refused. New files have 390 tensors, not 392: they leave out `norm.*`, add the setting `base`, and leave out `rate_per_doc`, `prior_var`, `delta_norm`, `no_norm` and `lr_schedule`. Shared names, shapes, float type and key order are the same.

## Layout

| | |
|---|---|
| `core/factorization/` | the model, its config, the LoRA, the objective, the posterior's pieces, saved runs |
| `core/train/` | seeding, the schedule, small pieces of the loop |
| `core/analysis/` | the scores: topic, language probe, finds original, variance split, top-k, layers, generation, comparison, trajectory |
| `core/checkpoints/`, `core/runs/` | the `model.pt` format; JSON and `--config` parsing |
| `experiments/data/` | the corpus, the translated test set, the teacher |
| `experiments/factorization/` | `train`, `train_step`, `epoch_eval`, `evaluate`, `figures`, `config`, `output`, `trajectory` and its `trajectory_*` parts |
| `experiments/baselines/` | frozen linear map, German LDA, LDA pair |
| `notebooks/`, `slurm/`, `tests/` | the two notebooks and their builder; batch scripts; pytest |
