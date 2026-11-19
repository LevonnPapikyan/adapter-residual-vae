# adapter-residual-vae

Code for "Isolating Latent Dimensions in Language Models with Adapter-Residual Variational Autoencoders".

A frozen language model (Qwen2.5) is turned into a variational autoencoder with two LoRA adapters: one for the encoder
and one for a prefix-conditioned generator. The latent is aligned to a reference, such as LDA topic mixtures or the
language label. The point of the method is where the latent is read from:

- **adapter result**: the last hidden state of the adapted model, $h_{\text{on}}$;
- **adapter residual**: what the adapter adds to the frozen model on the same input, $\Delta h = h_{\text{on}} - h_{\text{off}}$.

Aligned to English LDA topics and trained on English only, the adapter residual gives a latent that holds much less of
the input language and transfers the topics to German, French, Spanish and Chinese text.

## Results in short

1,000 test posts of 20 Newsgroups in five languages (the non-English versions are machine translations), Qwen2.5-0.5B,
mean of five seeds. Language: accuracy of a linear probe on the latent, lower is better (chance 0.20). Topic: share of
posts whose largest latent coordinate is LDA's top topic of the English original (always the most common topic: 0.126).

| latent read from | language | English | German | French | Spanish | Chinese |
|---|---|---|---|---|---|---|
| adapter result | 0.72 | 0.68 | 0.44 | 0.53 | 0.47 | 0.56 |
| adapter residual | 0.37 | 0.64 | 0.54 | 0.56 | 0.58 | 0.58 |
| frozen model + linear map | 0.80 | 0.51 | 0.21 | 0.22 | 0.20 | 0.29 |

With two adapter residuals of one frozen model, one aligned to the topic and one to the language (one seed), the topic
latent holds the language at 0.22 and the language latent holds the topic at chance, and the generator writes in a
chosen language (98.5 to 100% of texts) about a chosen topic group.

## Layout

| Folder | What it holds |
|---|---|
| `main/` | the model, training, evaluation, baselines, configs and tests. Its own `README.md` documents every setting. |
| `extensions/language_latent/` | the latent aligned to the language label |
| `extensions/two_latents/` | a topic latent and a language latent at once, and generation from a chosen topic and language |
| `extensions/sweeps/` | the model with one setting changed per run ($\beta$, $\lambda$, LoRA rank), and one scoring routine for any saved model |
| `extensions/multilingual_decode/` | a generator trained on five languages; the other extensions use its language detector and its translated training posts |
| `figures/` | scripts and small data files that draw the figures; `figures/paper/` holds reference results |
| `data/`, `models/` | not tracked; their `README.md` say which files go there |

## Setup

Python 3.10 or newer and one GPU with 24 GB or more for training.

```bash
cd main
pip install ".[test]"
python -m pytest tests
```

Tests that need the trained weights or the data skip themselves when those are missing.

## Reproducing

All commands run from `main/`; every path in a config is relative to it.

| What | Command |
|---|---|
| Train the pair (adapter result, adapter residual) | `python -m experiments.factorization.train --config configs/paper/factorization_h.json`, then `factorization_delta.json` |
| Every score and figure of a pair | `python -m experiments.factorization.evaluate --config configs/paper/evaluation.json` |
| Five seeds and their summary | the configs in `configs/seeds/`, then `python -m experiments.factorization.aggregate_seeds --config configs/seeds/aggregate.json` |
| The same on Qwen2.5-1.5B | `configs/paper/factorization_{h,delta}_1.5b.json`, `configs/paper/evaluation_1.5b.json` |
| Language latent | `python -u ../extensions/language_latent/train_language_guide.py --config ../extensions/language_latent/delta.json` |
| Two latents | `python -u ../extensions/two_latents/train_two_latents.py --config ../extensions/two_latents/main.json` |
| Generation grid (topic group by language) | `python -u ../extensions/two_latents/group_grid.py` |
| One sweep run | `python -u ../extensions/sweeps/train_and_score.py --config ../extensions/sweeps/configs/beta_0.1.json` |
| Score any saved model | `python -u ../extensions/sweeps/train_and_score.py --score-only ../models/final_0.5b/delta --name paper_delta` |
| Figures | `python figures/plot_latent_space.py`, `plot_two_latents.py`, `plot_heatmaps.py` (from the repository root, no GPU) |

The extension scripts need `PYTHONPATH` to include `main/`. The `run.sbatch` next to each script and the files in
`main/slurm/` are Slurm templates: edit the `#SBATCH` lines for your cluster and set `VAE_ENV_ACTIVATE` to a script
that activates the Python environment.

One training run of the 0.5B model takes about 30 minutes on an A100. Training is not bit-reproducible: reruns of the
same seed differ, and scores of the same weights move by a few documents between GPU types.

## Limits

- One dataset (20 Newsgroups), and all non-English text is machine translation (Marian).
- The reference for the topic is LDA with 20 topics, so "topic right" means agreement with LDA.
- The language-latent and two-latent results are one seed each.

## License and citation

MIT, see `LICENSE`. To cite the code, use `CITATION.cff` (GitHub shows it under "Cite this repository").
