"""Every checkpoint of a pair on the test documents and a sample of training posts.

    python -m experiments.factorization.trajectory --config configs/paper/trajectory.json
    python -m experiments.factorization.trajectory --config configs/paper/trajectory.json --figures-only

1. encode each checkpoint in float32 and bfloat16 into ``features/`` (needs the GPU),
2. score each one and write ``results.json``,
3. compute the maps and draw every figure into ``figures/``.

``--figures-only`` skips step 1 and loads no model, so it needs no GPU. A file already
in the cache is reused, so a job that was stopped can simply be submitted again.
``--saved-only`` puts the saved weights through the same code and stops.

What belongs here
-----------------
- The order of these steps.

What does NOT belong here
-------------------------
- Any of the steps themselves (``trajectory_encode``, ``trajectory_results``,
  ``trajectory_figures``, ``core/analysis/trajectory``).
"""

from __future__ import annotations

from dataclasses import asdict
import os

from experiments.factorization.config import TrajectoryConfig, parse_trajectory_config
from experiments.factorization.trajectory_cache import TrajectoryPaths
from experiments.factorization.trajectory_figures import draw_everything
from experiments.factorization.trajectory_results import build_maps, build_results, write_saved_check


def run_trajectory(*, config: TrajectoryConfig) -> None:
    """The three steps above, or the part of them the flags ask for."""

    paths = TrajectoryPaths(output_dir=config.output_dir)
    if not config.figures_only:
        from experiments.factorization.trajectory_encode import encode_everything  # loads transformers, needs no GPU to import

        encode_everything(config=config, paths=paths)
    if config.saved_only:
        write_saved_check(config=config, paths=paths)
        return
    build_results(config=config, paths=paths)
    build_maps(config=config, paths=paths)
    draw_everything(paths=paths)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = parse_trajectory_config(argv)
    print(asdict(config), flush=True)
    run_trajectory(config=config)


if __name__ == "__main__":
    main()
