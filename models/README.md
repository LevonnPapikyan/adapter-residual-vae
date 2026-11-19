# Trained models

Nothing in this folder is tracked. Each model folder holds `model.pt` (the kept epoch), `history.json` and `test.json`.

| Folder | What it is |
|---|---|
| `final_0.5b/{h,delta}` | the pair behind the main numbers, Qwen2.5-0.5B, seed 0 |
| `final_1.5b/{h,delta}` | the same recipe on Qwen2.5-1.5B, seed 0 |
| `seeds/seed_{1..5}/{h,delta}` | five more seeds |
| `baselines/` | reports of the LDA baselines |

The extensions keep their own weights next to their code: `extensions/language_latent/runs/{h,delta,h_heldout,delta_heldout}`,
`extensions/two_latents/runs/main` and `extensions/sweeps/runs/<name>`.

`h` reads the latent from the adapter result (the adapted model's last hidden state); `delta` reads it from the adapter
residual (that state minus the frozen model's).
