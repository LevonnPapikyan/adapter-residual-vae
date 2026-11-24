# Trained models

Nothing in this folder is tracked. Each model folder holds `model.pt` (the kept epoch), `history.json` and `test.json`.

| Folder | What it is |
|---|---|
| `final_0.5b/{h,delta}` | the pair behind the main numbers, Qwen2.5-0.5B, seed 0 |
| `seeds/seed_{1..5}/{h,delta}` | five more seeds |
| `baselines/` | reports of the LDA baselines |

The two-latent model keeps its weights next to its code: `extensions/two_latents/runs/main`.

`h` reads the latent from the adapter result (the adapted model's last hidden state); `delta` reads it from the adapter
residual (that state minus the frozen model's).
