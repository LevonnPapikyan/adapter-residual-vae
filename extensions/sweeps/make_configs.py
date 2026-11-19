"""Writes configs/*.json: the paper's Δh config (seed 0) with one setting changed per file, and a smoke config.

    python make_configs.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = json.loads((HERE.parents[1] / "main/configs/paper/factorization_delta.json").read_text())
SWEEP = {"beta": (0.1, 0.25, 1.0, 2.0), "lam": (0.25, 0.5, 2.0, 4.0), "lora_rank": (4, 64)}  # paper: beta 0.5, lam 1, rank 16

(HERE / "configs").mkdir(exist_ok=True)
for setting, values in SWEEP.items():
    for value in values:
        name = f"{setting}_{value}"
        config = {**PAPER, setting: value, "output_dir": f"../extensions/sweeps/runs/{name}"}
        (HERE / "configs" / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
        print(name)
rerun = {**PAPER, "beta": 1.0, "seed": 1, "output_dir": "../extensions/sweeps/runs/beta_1.0_seed1"}  # beta 1.0 again with another seed
(HERE / "configs" / "beta_1.0_seed1.json").write_text(json.dumps(rerun, indent=2) + "\n")
smoke = {**PAPER, "output_dir": "../extensions/sweeps/runs/smoke", "max_steps": 6, "max_train_texts": 200, "max_test_docs": 40}
(HERE / "configs" / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
