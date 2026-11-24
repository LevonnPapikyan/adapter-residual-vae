# Lightweight Factor Isolation and Composition in Frozen LLMs via Aligned Adapter-Residual Variational Autoencoders

This repository contains the code associated with the paper *Lightweight Factor Isolation and Composition in Frozen LLMs via Aligned Adapter-Residual Variational Autoencoders*.

> ***Abstract.*** The hidden states of large language models (LLMs) encode rich semantic and stylistic attributes, yet these factors remain deeply entangled within a single activation space. We introduce a framework for isolating and controlling specific, interpretable dimensions such as document topic and language from frozen pretrained LLMs. Our approach builds a variational autoencoder (VAE) using two lightweight LoRA adapters attached to a frozen Qwen2.5 backbone as an encoder and a prefix-conditioned decoder. Because standard variational objectives leave the latent space under-specified, we anchor the latents to external reference distributions, such as English LDA topic mixtures or one-hot language labels. Rather than reading the latent directly from the adapted final hidden state $h_{\text{on}}$, we read it from the adapter residual $\Delta h = h_{\text{on}} - h_{\text{off}}$ introduced by the adapter relative to the frozen base model, filtering the representation to retain only the target attribute. Aligned solely to English LDA topics, the latent $\mu$ generated from $\Delta h$ reduces language predictability from 72\% to 37\% and improves zero-shot cross-lingual topic transfer on machine-translated German, French, Spanish, and Chinese documents. Training parallel topic and language $\Delta h$ encoders jointly with a shared LoRA decoder allows for the generation of orthogonal topic and language latents, enabling simultaneous, disentangled control over both content and generation language.

## Setup

Python 3.10 or newer. From the `main/` folder:

```bash
cd main
pip install ".[test]"
export TOKENIZERS_PARALLELISM=false
```

`main/pyproject.toml` pins the package versions. The data and the trained weights are not in the repository; `data/README.md` and `models/README.md` list the files and how each is made.

All commands below run from `main/`, and all paths in the configs are relative to it: `../data` and `../models` are read, and everything new goes to `outputs/`.

## Small CPU example

A few texts, short sequences and one epoch, on one CPU thread:

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
python -m experiments.factorization.train --config configs/small_cpu_example/factorization_delta.json
python -m experiments.factorization.evaluate --config configs/small_cpu_example/evaluation.json
python -m pytest tests
```

The evaluation reads the saved weights in `../models/final_0.5b`. Tests that need the weights or the data skip themselves when those are missing.

## Configuration

Training and evaluation take a JSON file through `--config PATH`. Command-line options override values in the file, and an unknown key is an error.

- `base_model`, `num_topics`, `teacher`, `min_words`: the language model, the latent dimension, the LDA file in `data_dir`, and how many LDA words a training document needs.
- `mu_from` selects the readout: `"h"` is the adapter result $h_{\text{on}}$, and `"delta"` is the adapter residual $\Delta h = h_{\text{on}} - h_{\text{off}}$. This is the only difference between the two single-attribute models of the paper.
- `lam` weights the alignment term and `beta` weights the KL term. The paper uses `lam = 1.0` and `beta = 0.5`.
- `lora_rank` sets the LoRA rank of the encoder and the decoder; alpha is always twice the rank. The paper uses rank 16.
- `lr_encoder_lora`, `lr_decoder_lora`, `lr_linear_maps`: the three AdamW learning rates.
- `num_epochs`, `batch_size`, `val_frac`, `encoder_max_tokens`, `decoder_max_tokens`, `seed`: the rest of the recipe.
- `data_dir`, `output_dir`, `device`, `log_every`: where and how a run goes.
- `checkpoint_steps` (training) saves the trained tensors at the listed steps.
- `max_train_texts`, `max_test_docs`, `max_steps` (training) and `max_docs`, `max_generation_texts`, `skip_layers`, `skip_generation` (evaluation) shorten a run for testing. Left unset they change nothing.

## Paper experiment configurations

Fit the LDA reference, train the adapter-result and adapter-residual models on Qwen2.5-0.5B, and evaluate the pair:

```bash
python -m experiments.data.teacher --config configs/paper/teacher.json
python -m experiments.factorization.train --config configs/paper/factorization_h.json
python -m experiments.factorization.train --config configs/paper/factorization_delta.json
python -m experiments.factorization.evaluate --config configs/paper/evaluation.json
```

The five seeds use the configs in `configs/seeds/` (`seed_{1..5}_{h,delta}.json`, `evaluation_seed_{1..5}.json`), and their summary is:

```bash
python -m experiments.factorization.aggregate_seeds --config configs/seeds/aggregate.json
```

The training-step analysis trains the same two recipes with checkpoints and then reads every checkpoint:

```bash
python -m experiments.factorization.train --config configs/paper/checkpointed_h.json
python -m experiments.factorization.train --config configs/paper/checkpointed_delta.json
python -m experiments.factorization.trajectory --config configs/paper/trajectory.json
```

The evaluation also scores the baselines of the paper: LDA on the translated text, and frozen Qwen with a linear map. The latter runs alone with `python -m experiments.baselines.frozen_linear_map`.

The model with parallel topic and language residuals lives in `extensions/two_latents/`. Its scripts need `PYTHONPATH` to include `main/`:

```bash
python -u ../extensions/multilingual_decode/translate_train.py
python -u ../extensions/two_latents/train_two_latents.py --config ../extensions/two_latents/main.json
python -u ../extensions/two_latents/group_grid.py
```

The first command translates the training documents into French, Spanish and Chinese. The second trains the two encoders and the shared decoder. The third writes texts for every pair of topic group and language and scores them. `extensions/multilingual_decode/` also holds the language detector that these scripts use.

The provided configurations are starting points; they do not specify every experiment reported in the paper. Training is not bit-reproducible: reruns of the same seed differ, and scores of the same weights move by a few documents between GPU types.

## Evaluation

**Topic accuracy** is the share of test documents whose largest coordinate of $\mu$ is the top LDA topic of the English original. It is computed separately for the English documents and for their German, French, Spanish and Chinese translations.

**Language accuracy** is the accuracy of a linear probe that predicts the document's language from $\text{softmax}(\mu)$, with 5-fold cross-validation in which all five versions of a document fall into the same fold. Chance is 0.20, and lower means less language in the latent.

**Top-$k$ topic retrieval** is the share of test documents whose top LDA topic is among the $k$ largest coordinates of $\mu$.

**Compositional generation** takes the mean $\mu_{\text{topic}}$ of a topic group and the mean $\mu_{\text{lang}}$ of a language and lets the decoder write 50 texts. A text counts as on topic if, after translation back to English, its top LDA topic belongs to the chosen group, and as in the right language if the language detector returns the chosen language. The same judge on real test documents gives the ceiling.

## Outputs

- Training writes `model.pt` (the trained tensors and the config), `history.json` (loss parts per step) and `test.json` (held-out loss and test scores per epoch) beneath its `output_dir`.
- Evaluation writes `results.json`, `comparison.json`, `environment.json` and its figures beneath its `output_dir`. `comparison.json` is the difference to `figures/paper/results.json`, the scores of the released seed 0 weights.
- The trajectory command writes `results.json`, a cache in `features/` and its figures beneath its `output_dir`.
- The two-latent scripts write `model.pt`, scores, written samples and the generation grid next to their config.

## Citation

```bibtex
@misc{papikyan2026lightweight,
  title  = {Lightweight Factor Isolation and Composition in Frozen LLMs via Aligned Adapter-Residual Variational Autoencoders},
  author = {Papikyan, Levon and Lichtman, Elliot and Lafferty, John},
  year   = {2026},
}
```

## License

This code is licensed under the MIT License.
