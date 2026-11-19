"""The aggregation of seed runs, on small made-up evaluations. No language model involved."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.runs.io import write_json
from experiments.data.translations import LANGS
from experiments.factorization.aggregate_seeds import (
    AggregationConfig,
    aggregate,
    corrected_interval,
    language_verdicts,
    parse_config,
    rough_sampling_sd,
    run_aggregation,
    summarize,
    summarize_paired,
)
from experiments.factorization.output import (
    LABEL_DELTA,
    LABEL_FROZEN,
    LABEL_H,
    LABEL_LDA,
    LABEL_STATE_DELTA,
    LABEL_STATE_H,
    LABEL_STATE_OFF,
)


def _results(*, h: int, delta: int, frozen: int = 200) -> dict[str, Any]:
    state = {"finds_original": [100, 200, 300, 40], "language": 0.5, "document": 0.25}
    return {
        "topic": {
            LABEL_LDA: {lang: 300 for lang in LANGS},
            LABEL_FROZEN: {lang: frozen for lang in LANGS},
            LABEL_H: {"en": 700, "de": h, "fr": h, "es": h, "zh": h},
            LABEL_DELTA: {"en": 650, "de": delta, "fr": delta, "es": delta, "zh": delta},
        },
        "language": {LABEL_FROZEN: 0.8, LABEL_H: 0.7, LABEL_DELTA: 0.4},
        "generation": {LABEL_H: {"0": 3, "1": 4}, LABEL_DELTA: {"0": 5, "1": 6}},
        "inside": {
            LABEL_STATE_OFF: state,
            LABEL_STATE_H: {**state, "finds_original": [10, 20, 30, 4]},
            LABEL_STATE_DELTA: {**state, "finds_original": [700, 800, 850, 600]},
        },
        "majority": 126,
        "layers": [{"topic": {}, "language": 0.3, "finds_original": 1.0}],
    }


def _write_pair(*, evaluation: Path, runs: Path, log: Path, results: dict[str, Any]) -> None:
    write_json(evaluation / "results.json", results)
    write_json(evaluation / "comparison.json", {"reference": "shipped.json", "difference": {"topic": {**{label: {lang: 2 for lang in LANGS} for label in (LABEL_H, LABEL_DELTA)}, LABEL_FROZEN: {lang: 3 for lang in LANGS}}}})
    for model in ("h", "delta"):
        write_json(runs / model / "test.json", [{"epoch": 0, "val": 1.0}, {"epoch": 1, "val": 5.0}, {"epoch": 2, "val": 4.0}, {"epoch": 3, "val": 4.5}])
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("size of Δh relative to h_off: 0.05\n", encoding="utf-8")


def _config(root: Path, *, differences: list[int], frozen_last: int = 200) -> AggregationConfig:
    """Made-up evaluations under ``root``. Paths are relative, as in the real config, so the test runs from ``root``."""

    seeds = tuple(range(1, len(differences) + 1))
    for seed, difference in zip(seeds, differences):
        seed_dir = root / "seeds" / f"seed_{seed}"
        frozen = frozen_last if seed == seeds[-1] else 200
        _write_pair(evaluation=seed_dir / "evaluation", runs=seed_dir, log=seed_dir / "logs" / "evaluate.1.out", results=_results(h=500, delta=500 + difference, frozen=frozen))
    for name in ("original", "retrain"):
        _write_pair(evaluation=root / name / "evaluation", runs=root / name, log=root / name / "evaluate.2.out", results=_results(h=600, delta=500))
    write_json(root / "shipped.json", _results(h=590, delta=500))
    return AggregationConfig(
        seeds_dir=Path("seeds"),
        seeds=seeds,
        output_dir=Path("out"),
        original_evaluation=Path("original/evaluation"),
        original_runs=Path("original"),
        original_log="original/evaluate.*.out",
        retrain_evaluation=Path("retrain/evaluation"),
        retrain_runs=Path("retrain"),
        retrain_log="retrain/evaluate.*.out",
        shipped_results=Path("shipped.json"),
    )


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Globs in the config are relative to the folder commands run from."""

    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_summarize_matches_hand_computation() -> None:
    row = summarize([1, 2, 3, 4, 5])
    assert row["mean"] == 3 and row["min"] == 1 and row["max"] == 5 and row["n"] == 5
    assert row["sd"] == pytest.approx(2.5**0.5)
    half = 2.7764451 * 2.5**0.5 / 5**0.5  # t(0.975, 4) from a table
    assert row["ci95_t"] == pytest.approx([3 - half, 3 + half], abs=1e-5)


def test_paired_counts_each_direction_and_ties() -> None:
    row = summarize_paired([5, 5, 5, 5, 5], [1, 5, 9, 2, 3])
    assert row["by_seed"] == [4, 0, -4, 3, 2]
    assert (row["delta_above_h"], row["tied"], row["delta_below_h"]) == (3, 1, 1)


def test_corrected_interval_matches_hand_computation() -> None:
    row = summarize([1, 2, 3, 4, 5])
    half = 4.6040949 * 2.5**0.5 / 5**0.5  # t(0.995, 4) from a table, 0.05 shared over five rows
    assert corrected_interval(row, family=5) == pytest.approx([3 - half, 3 + half], abs=1e-5)


def test_rough_sampling_sd_matches_hand_computation() -> None:
    assert rough_sampling_sd(h_mean=500, delta_mean=500, num_docs=1000) == pytest.approx((1000 * 0.5) ** 0.5)


def _verdict(by_seed: list[float], *, move: int = 7) -> dict:
    row = summarize_paired(by_seed, [0.0] * len(by_seed))
    row["ci95_t_corrected"] = corrected_interval(row, family=5)
    return language_verdicts(paired_topic={"fr": row}, same_weights_move=move)["fr"]


def test_verdict_firm_needs_margin_and_corrected_interval() -> None:
    assert _verdict([100, 105, 110, 115, 120])["verdict"] == "firm"
    assert _verdict([-100, -105, -110, -115, -120])["verdict"] == "firm"


def test_verdict_small_margin_is_not_settled_even_with_a_tight_interval() -> None:
    verdict = _verdict([5, 5.5, 6, 6.5, 7])
    assert verdict["verdict"] == "not settled" and verdict["corrected_interval_clear_of_zero"] is True
    assert verdict["margin_beyond_same_weights_move"] is False and verdict["smallest_margin"] == 5


def test_verdict_wide_interval_is_not_settled_even_with_a_clear_margin() -> None:
    verdict = _verdict([45, 32, 80, 33, 8])
    assert verdict["verdict"] == "not settled" and verdict["margin_beyond_same_weights_move"] is True
    assert verdict["corrected_interval_clear_of_zero"] is False


def test_verdict_mixed_signs_is_no_clear_winner() -> None:
    assert _verdict([-2, 34, 64, 13, -12])["verdict"] == "no clear winner"


def test_pattern_holds_when_every_seed_agrees(in_tmp: Path) -> None:
    config = _config(in_tmp, differences=[10, 20, 30, 40, 50])
    summary = aggregate(config)
    assert summary["pattern"]["holds_in_every_seed"] is True
    assert summary["paired_delta_minus_h"]["topic"]["de"]["delta_above_h"] == 5
    assert summary["models"]["h"]["kept_epoch"]["counts"] == {"2": 5}
    assert summary["models"]["delta"]["delta_size"]["counts"] == {"0.05": 5}
    assert "ci95_t" not in summary["models"]["h"]["kept_epoch"] and "ci95_t" not in summary["models"]["delta"]["delta_size"]
    assert summary["models"]["h"]["delta_size"] is None


def test_pattern_names_the_seed_that_breaks_it(in_tmp: Path) -> None:
    config = _config(in_tmp, differences=[10, 20, -30, 40, 50])
    summary = aggregate(config)
    assert summary["pattern"]["holds_in_every_seed"] is False
    assert summary["pattern"]["seeds_where_it_fails"]["topic zh: Δh above h"] == [3]
    assert summary["pattern"]["seeds_where_it_fails"]["topic en: Δh below h"] == []


def test_references_stay_out_of_the_means(in_tmp: Path) -> None:
    config = _config(in_tmp, differences=[10, 20, 30, 40, 50])
    summary = aggregate(config)
    assert summary["models"]["h"]["topic"]["de"]["mean"] == 500
    assert summary["references"]["original paper weights"]["h"]["topic"]["de"] == 600
    assert "h topic de" in summary["references"]["seed 0 retrain"]["outside_seed_range"]
    assert summary["references"]["seed 0 retrain"]["delta_minus_h_topic"]["de"] == -100
    assert summary["references"]["seed 0 retrain"]["largest_topic_move_from_it"] == 2


def test_gaps_from_seed_zero_are_checked_against_the_range_of_gaps(in_tmp: Path) -> None:
    summary = aggregate(_config(in_tmp, differences=[10, 20, 30, 40, 50]))
    gaps = summary["seed_zero_gaps"]
    assert gaps["runs"]["shipped file"]["de"] == -90 and gaps["runs"]["seed 0 retrain"]["de"] == -100
    assert gaps["outside_seed_range"]["de"] == ["original paper weights", "seed 0 retrain", "shipped file"]
    assert gaps["outside_seed_range"]["en"] == []  # -50 in the seeds and in every seed 0 source


def test_base_rate_is_two_in_n_plus_one(in_tmp: Path) -> None:
    summary = aggregate(_config(in_tmp, differences=[10, 20, 30, 40, 50]))
    base = summary["outside_base_rate"]
    assert base["rows"] == 26 and base["chance_per_row"] == pytest.approx(1 / 3) and base["expected_outside"] == pytest.approx(26 / 3)
    assert base["outside"]["seed 0 retrain"] == len(summary["references"]["seed 0 retrain"]["outside_seed_range"])


def test_rerun_gap_sits_next_to_the_seed_range(in_tmp: Path) -> None:
    summary = aggregate(_config(in_tmp, differences=[10, 20, 30, 40, 50]))
    assert summary["rerun_against_seed"]["delta"]["de"] == {"rerun_gap": 0, "seed_range": 40}


def test_a_baseline_that_moves_is_refused(in_tmp: Path) -> None:
    config = _config(in_tmp, differences=[10, 20, 30], frozen_last=201)
    with pytest.raises(ValueError, match="other baseline numbers"):
        aggregate(config)


def test_a_missing_seed_is_an_error_not_a_smaller_n(in_tmp: Path) -> None:
    config = _config(in_tmp, differences=[10, 20, 30])
    (in_tmp / "seeds" / "seed_2" / "evaluation" / "results.json").unlink()
    with pytest.raises(FileNotFoundError):
        aggregate(config)


def test_a_smoke_evaluation_is_refused(in_tmp: Path) -> None:
    config = _config(in_tmp, differences=[10, 20, 30])
    path = in_tmp / "seeds" / "seed_1" / "evaluation" / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    write_json(path, {**results, "generation": {}})
    with pytest.raises(ValueError, match="no generation"):
        aggregate(config)


def test_run_writes_summaries_and_figures(in_tmp: Path) -> None:
    config = _config(in_tmp, differences=[10, 20, -30, 40, 50])
    run_aggregation(config=config)
    written = json.loads((in_tmp / "out" / "summary.json").read_text(encoding="utf-8"))
    assert written["seeds"] == [1, 2, 3, 4, 5]
    text = (in_tmp / "out" / "summary.md").read_text(encoding="utf-8")
    assert "does not hold in every seed" in text and "95% t interval, n = 5" in text
    assert "does not show seed 0 is unusual" in text and "2/(5 + 1) = 0.33" in text
    assert "no clear winner" in text and "rounding that may differ by GPU type" in text and "not a measurement" in text
    assert "kept epoch |" not in text.split("## Each model over the seeds")[1].split("## For scale")[0]
    assert "\u2013" not in text and "\u2014" not in text
    for name in ("seeds_topic_score", "seeds_paired_difference"):
        assert (in_tmp / "out" / "figures" / f"{name}.pdf").stat().st_size > 0
        assert (in_tmp / "out" / "figures" / f"{name}.png").stat().st_size > 0


def test_protected_output_folder_is_refused(tmp_path: Path) -> None:
    from experiments.data.paths import PROTECTED_DIRS

    with pytest.raises(ValueError):
        AggregationConfig(
            seeds_dir=tmp_path,
            seeds=(1, 2),
            output_dir=PROTECTED_DIRS[0] / "x",
            original_evaluation=tmp_path,
            original_runs=tmp_path,
            original_log="",
            retrain_evaluation=tmp_path,
            retrain_runs=tmp_path,
            retrain_log="",
            shipped_results=tmp_path,
        )


def test_config_file_parses() -> None:
    from conftest import MAIN_ROOT

    config = parse_config(["--config", str(MAIN_ROOT / "configs" / "seeds" / "aggregate.json")])
    assert config.seeds == (1, 2, 3, 4, 5)
    assert config.output_dir == Path("outputs/seeds")
