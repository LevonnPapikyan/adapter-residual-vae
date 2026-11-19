"""Aggregate the seed runs: what holds across seeds, and how far.

    python -m experiments.factorization.aggregate_seeds --config configs/seeds/aggregate.json

Writes ``summary.json``, ``summary.md`` and ``figures/`` under ``output_dir``. Needs no GPU
and loads no model.

Scores come from each seed's ``evaluation/results.json``. Two things the evaluation does
not put in that file come from next to it: the kept epoch from each run's ``test.json``, by
the rule ``evaluate.py`` prints (lowest held-out loss), and the size of Δh from the line the
evaluation prints in its log, which has two decimals and exists only for the Δh model. The
shipped ``results.json`` is read only to compare the Δh minus h gaps quoted from seed 0.

What belongs here
-----------------
- Reading finished evaluations, the statistics over seeds, the paired differences, the
  check of the pattern in every seed, and the summary files and figures.

What does NOT belong here
-------------------------
- Computing any score (``evaluate.py``, ``core/analysis``). Nothing here touches a model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as scipy_stats

from core.runs.io import parse_args_with_config, read_json, refuse_protected_path, write_json
from experiments.data.paths import PROTECTED_DIRS
from experiments.data.translations import LANG_NAMES, LANGS, NUM_DOCS
from experiments.factorization.figures import COLOR_DELTA, COLOR_H, plt
from experiments.factorization.output import (
    LABEL_DELTA,
    LABEL_FROZEN,
    LABEL_H,
    LABEL_LDA,
    LABEL_STATE_DELTA,
    LABEL_STATE_H,
    LABEL_STATE_OFF,
)

MODELS = {"h": (LABEL_H, LABEL_STATE_H), "delta": (LABEL_DELTA, LABEL_STATE_DELTA)}
MODEL_NAMES = {"h": "h", "delta": "Δh"}
OTHER_LANGS = LANGS[1:]  # finds_original looks for the English original, so it has no English entry
SCALARS = ("language_probe", "language_share", "document_share", "generation")
PAIRED_SCALAR_NOTES = {
    "language_probe": "language probe (chance 0.20, lower is better)",
    "language_share": "language variance share of the read vector",
    "document_share": "document variance share of the read vector",
    "generation": "generation (of 200)",
}
SIZE_LINE = re.compile(r"size of Δh relative to h_off: ([0-9.]+)")
COLOR_SCALE = "0.35"
SIZE_CELL = r"\|Δh\|/\|h_off\|"  # the pipes would otherwise end a Markdown table cell
ORIGINAL, RETRAIN, SHIPPED = "original paper weights", "seed 0 retrain", "shipped file"


def coverage_notes(*, seeds: Sequence[int], paired_topic: Mapping[str, Any], same_weights_move: int, frozen_move: int) -> dict[str, str]:
    """What the spread means. Only the sizes come from the numbers."""

    n = len(seeds)
    corrected_t = float(scipy_stats.t.ppf(1 - 0.05 / (2 * len(LANGS)), n - 1))
    inside = [f"{LANG_NAMES[lang]} seed {seed} at {value:+.0f}" for lang in LANGS for seed, value in zip(seeds, paired_topic[lang]["by_seed"]) if abs(value) <= same_weights_move]
    sampling = [row["rough_sampling_sd"] for row in paired_topic.values()]
    in_sds = ", ".join(f"{lang} {abs(paired_topic[lang]['mean']) / paired_topic[lang]['rough_sampling_sd']:.1f}" for lang in LANGS)
    return {
        "covers": "The spread between seeds covers random initialisation, the held-out split, batch order and noise in training.",
        "does_not_cover": (
            "It does not cover a different LDA teacher, different translations, or sampling of test documents: "
            "every seed is scored against the same teacher on the same 1,000 documents and the same machine translations."
        ),
        "gpu_rounding": (
            f"It does not cover rounding that may differ by GPU type: every seed ran on an A100. The same weights differ by up to {same_weights_move} documents on a trained-model score, "
            f"and the frozen baseline by up to {frozen_move}, between `evaluation_original` and the shipped `../figures/paper/results.json`. A comment in `slurm/evaluate.sbatch` puts this down to GPU type; "
            "the H200 run that would test it, `outputs/checkpointed/evaluation_original_h200`, had not finished when this was written, so the cause is unconfirmed. "
            f"Seed margins inside that difference could change sign on another GPU type: {', '.join(inside) or 'none'}."
        ),
        "test_documents": (
            f"Averaging seeds does not reduce the sampling of test documents, because every seed uses the same {NUM_DOCS:,}. A rough size, not a measurement: "
            f"the binomial sd of an unpaired difference of two scores on {NUM_DOCS:,} documents is about {min(sampling):.0f} to {max(sampling):.0f} documents. "
            f"The true paired value should be smaller and needs per-document predictions, which `results.json` does not hold. In units of that rough size, the mean differences are {in_sds}."
        ),
        "small_n": (
            f"n = {n} gives a wide interval. Each interval is for the mean over seeds, not for where one new run lands, which is wider. "
            f"The plain interval has no correction for looking at several rows; the topic rows also show one corrected (Bonferroni) over the {len(LANGS)} languages, t = {corrected_t:.3f}. "
            f"Both lean on the differences being roughly normal. With {n} seeds even {n} of {n} in one direction is only p = {2 * 0.5**n:.4g} on a two-sided sign test."
        ),
        "generation": (
            "Generation uses evaluation seed 0 in every evaluation, so each model has one sampling draw, "
            "and its spread mixes training with sampling."
        ),
    }


@dataclass(frozen=True)
class AggregationConfig:
    """One aggregation of finished seed pairs.

    Attributes
    ----------
    seeds_dir:
        Folder holding ``seed_{k}/evaluation``, ``seed_{k}/{h,delta}`` and ``seed_{k}/logs``. Read only.
    seeds:
        The seeds, all of them. A missing one is an error, never a smaller n.
    output_dir:
        ``summary.json``, ``summary.md`` and ``figures/`` go here.
    original_evaluation, original_runs, original_log:
        The original paper weights: their evaluation, ``h/`` and ``delta/`` runs, and evaluation
        log (a glob with one match). Reported apart and kept out of every mean.
    retrain_evaluation, retrain_runs, retrain_log:
        The same for the seed 0 retrain.
    shipped_results:
        The shipped ``results.json``, read only for its Δh minus h gaps.
    """

    seeds_dir: Path
    seeds: tuple[int, ...]
    output_dir: Path
    original_evaluation: Path
    original_runs: Path
    original_log: str
    retrain_evaluation: Path
    retrain_runs: Path
    retrain_log: str
    shipped_results: Path

    def __post_init__(self) -> None:
        for name in ("seeds_dir", "output_dir", "original_evaluation", "original_runs", "retrain_evaluation", "retrain_runs", "shipped_results"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be at least two different seeds, got {self.seeds!r}.")
        refuse_protected_path(self.output_dir, protected_dirs=PROTECTED_DIRS)


def build_parser() -> argparse.ArgumentParser:
    """Every aggregation setting as an option."""

    parser = argparse.ArgumentParser(description="Mean, spread and paired differences of the seed runs.")
    parser.add_argument("--seeds-dir", default="outputs/seeds")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", default="outputs/seeds")
    parser.add_argument("--original-evaluation", default="outputs/checkpointed/evaluation_original")
    parser.add_argument("--original-runs", default="../models/final_0.5b")
    parser.add_argument("--original-log", default="outputs/checkpointed/logs/evaluate_original.*.out")
    parser.add_argument("--retrain-evaluation", default="outputs/checkpointed/evaluation")
    parser.add_argument("--retrain-runs", default="outputs/checkpointed")
    parser.add_argument("--retrain-log", default="outputs/checkpointed/logs/evaluate.*.out")
    parser.add_argument("--shipped-results", default="../figures/paper/results.json")
    return parser


def parse_config(argv: list[str] | None = None) -> AggregationConfig:
    """Defaults, then ``--config FILE``, then the command line."""

    values = vars(parse_args_with_config(build_parser(), argv))
    values.pop("config")
    return AggregationConfig(**values)


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """Mean, sample standard deviation, range and the 95 percent t interval for the mean."""

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = float(scipy_stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "min": float(values.min()), "max": float(values.max()), "ci95_t": [mean - half, mean + half]}


def summarize_paired(delta: Sequence[float], h: Sequence[float]) -> dict[str, Any]:
    """The statistics of Δh minus h by seed, each seed's difference, and how many seeds go which way."""

    differences = np.asarray(delta, dtype=float) - np.asarray(h, dtype=float)
    return {
        **summarize(differences),
        "by_seed": [float(value) for value in differences],
        "delta_above_h": int((differences > 0).sum()),
        "tied": int((differences == 0).sum()),
        "delta_below_h": int((differences < 0).sum()),
    }


def summarize_counts(values: Sequence[float]) -> dict[str, Any]:
    """Counts and range, for numbers where a mean and a t interval carry no meaning."""

    values = list(values)
    return {"n": len(values), "min": min(values), "max": max(values), "counts": {str(value): values.count(value) for value in sorted(set(values))}}


def corrected_interval(row: Mapping[str, Any], *, family: int) -> list[float]:
    """The 95 percent t interval for the mean, with a Bonferroni correction over ``family`` rows."""

    half = float(scipy_stats.t.ppf(1 - 0.05 / (2 * family), row["n"] - 1) * row["sd"] / np.sqrt(row["n"]))
    return [row["mean"] - half, row["mean"] + half]


def rough_sampling_sd(*, h_mean: float, delta_mean: float, num_docs: int) -> float:
    """Binomial sd of an unpaired difference of two scores, in documents. A rough size, not a measurement."""

    p, q = h_mean / num_docs, delta_mean / num_docs
    return float(np.sqrt(num_docs * (p * (1 - p) + q * (1 - q))))


def kept_epoch(*, run_dir: Path) -> int:
    """The epoch of the lowest held-out loss. Row 0 of ``test.json`` is before training."""

    return min(read_json(run_dir / "test.json")[1:], key=lambda row: row["val"])["epoch"]


def delta_size(*, log: Path) -> float:
    """|Δh| / |h_off| as the evaluation printed it: two decimals, for the Δh model only."""

    found = SIZE_LINE.findall(log.read_text(encoding="utf-8", errors="replace"))
    if len(found) != 1:
        raise ValueError(f"{log} should print the size of Δh once, it does so {len(found)} times.")
    return float(found[0])


def one_log(pattern: str) -> Path:
    """The one log a glob names; a glob with several would hide which one was read."""

    matches = sorted(Path().glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{pattern} should match one log, it matches {[str(path) for path in matches]}.")
    return matches[0]


def read_results(evaluation_dir: Path) -> dict[str, Any]:
    """A full evaluation's ``results.json``. A smoke test's has no generation or layers and is refused."""

    results = read_json(evaluation_dir / "results.json")
    for label in (LABEL_H, LABEL_DELTA):
        if not results["generation"].get(label):
            raise ValueError(f"{evaluation_dir} has no generation scores for {label}.")
    if not results["layers"]:
        raise ValueError(f"{evaluation_dir} has no layer scan, so it is not a full evaluation.")
    return results


def pair_metrics(*, results: Mapping[str, Any], runs_dir: Path, log: Path) -> dict[str, dict[str, Any]]:
    """The numbers of one pair, per model."""

    pair = {}
    for model, (label, state) in MODELS.items():
        inside = results["inside"][state]
        pair[model] = {
            "topic": {lang: results["topic"][label][lang] for lang in LANGS},
            "retrieval": dict(zip(OTHER_LANGS, inside["finds_original"], strict=True)),
            "language_probe": results["language"][label],
            "language_share": inside["language"],
            "document_share": inside["document"],
            "generation": sum(results["generation"][label].values()),
            "delta_size": delta_size(log=log) if model == "delta" else None,
            "kept_epoch": kept_epoch(run_dir=runs_dir / model),
        }
    return pair


def scale_numbers(results: Mapping[str, Any]) -> dict[str, Any]:
    """What involves no trained model, so should be the same in every evaluation."""

    return {
        "frozen": {"topic": results["topic"][LABEL_FROZEN], "language_probe": results["language"][LABEL_FROZEN]},
        "lda": {"topic": results["topic"][LABEL_LDA]},
        "h_off": {
            "retrieval": dict(zip(OTHER_LANGS, results["inside"][LABEL_STATE_OFF]["finds_original"], strict=True)),
            "language_share": results["inside"][LABEL_STATE_OFF]["language"],
            "document_share": results["inside"][LABEL_STATE_OFF]["document"],
        },
        "majority": results["majority"],
    }


def read_seeds(config: AggregationConfig) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Every seed's numbers, and the scale numbers, which must agree across seeds."""

    per_seed, scales = {}, {}
    for seed in config.seeds:
        seed_dir = config.seeds_dir / f"seed_{seed}"
        results = read_results(seed_dir / "evaluation")
        per_seed[seed] = pair_metrics(results=results, runs_dir=seed_dir, log=one_log(str(seed_dir / "logs" / "evaluate.*.out")))
        scales[seed] = scale_numbers(results)
    first = scales[config.seeds[0]]
    for seed, scale in scales.items():
        if scale != first:
            raise ValueError(f"Seed {seed} has other baseline numbers than seed {config.seeds[0]}, so one row cannot stand for all.")
    return per_seed, first


def _column(per_seed: Mapping[int, Mapping[str, Any]], model: str, metric: str, key: str | None = None) -> list[float]:
    return [row[model][metric] if key is None else row[model][metric][key] for row in per_seed.values()]


def model_statistics(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean, spread and interval of every number, per model."""

    summary = {}
    for model in MODELS:
        summary[model] = {
            "topic": {lang: summarize(_column(per_seed, model, "topic", lang)) for lang in LANGS},
            "retrieval": {lang: summarize(_column(per_seed, model, "retrieval", lang)) for lang in OTHER_LANGS},
            **{name: summarize(_column(per_seed, model, name)) for name in SCALARS},
            "delta_size": summarize_counts(_column(per_seed, model, "delta_size")) if model == "delta" else None,
            "kept_epoch": summarize_counts(_column(per_seed, model, "kept_epoch")),
        }
    return summary


def paired_statistics(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Δh minus h by seed, for everything both models have and where the sign means something."""

    topic = {lang: summarize_paired(_column(per_seed, "delta", "topic", lang), _column(per_seed, "h", "topic", lang)) for lang in LANGS}
    for lang, row in topic.items():
        row["ci95_t_corrected"] = corrected_interval(row, family=len(LANGS))
        row["rough_sampling_sd"] = rough_sampling_sd(
            h_mean=float(np.mean(_column(per_seed, "h", "topic", lang))), delta_mean=float(np.mean(_column(per_seed, "delta", "topic", lang))), num_docs=NUM_DOCS
        )
    return {
        "topic": topic,
        "retrieval": {
            lang: summarize_paired(_column(per_seed, "delta", "retrieval", lang), _column(per_seed, "h", "retrieval", lang)) for lang in OTHER_LANGS
        },
        **{name: summarize_paired(_column(per_seed, "delta", name), _column(per_seed, "h", name)) for name in SCALARS},
    }


def reference_rows(*, config: AggregationConfig, summary: Mapping[str, Any]) -> dict[str, Any]:
    """The runs seed 0 already showed, apart from the means, and where they fall outside the seeds' range."""

    rows = {}
    for name, evaluation, runs, log in (
        ("original paper weights", config.original_evaluation, config.original_runs, config.original_log),
        ("seed 0 retrain", config.retrain_evaluation, config.retrain_runs, config.retrain_log),
    ):
        pair = pair_metrics(results=read_results(evaluation), runs_dir=runs, log=one_log(log))
        outside = []
        for model in MODELS:
            for lang in LANGS:
                if not summary[model]["topic"][lang]["min"] <= pair[model]["topic"][lang] <= summary[model]["topic"][lang]["max"]:
                    outside.append(f"{MODEL_NAMES[model]} topic {lang}")
            for lang in OTHER_LANGS:
                if not summary[model]["retrieval"][lang]["min"] <= pair[model]["retrieval"][lang] <= summary[model]["retrieval"][lang]["max"]:
                    outside.append(f"{MODEL_NAMES[model]} retrieval {lang}")
            for metric in SCALARS:
                if not summary[model][metric]["min"] <= pair[model][metric] <= summary[model][metric]["max"]:
                    outside.append(f"{MODEL_NAMES[model]} {metric.replace('_', ' ')}")
        shipped = read_json(evaluation / "comparison.json")
        moved = shipped["difference"]["topic"]
        rows[name] = {
            "evaluation": str(evaluation),
            "kept_out_of_means": True,
            "compared_with": shipped["reference"],
            "largest_topic_move_from_it": max(abs(moved[label][lang]) for label in (LABEL_H, LABEL_DELTA) for lang in LANGS),
            "largest_frozen_move_from_it": max(abs(moved[LABEL_FROZEN][lang]) for lang in LANGS),
            **pair,
            "delta_minus_h_topic": {lang: pair["delta"]["topic"][lang] - pair["h"]["topic"][lang] for lang in LANGS},
            "outside_seed_range": outside,
        }
    return rows


def seed_zero_gaps(*, config: AggregationConfig, references: Mapping[str, Any], paired_topic: Mapping[str, Any]) -> dict[str, Any]:
    """Δh minus h from each seed 0 source against the seeds' range of that same gap, which ``outside_seed_range`` does not test."""

    shipped = read_json(config.shipped_results)["topic"]
    runs = {name: dict(references[name]["delta_minus_h_topic"]) for name in (ORIGINAL, RETRAIN)}
    runs[SHIPPED] = {lang: shipped[LABEL_DELTA][lang] - shipped[LABEL_H][lang] for lang in LANGS}
    outside = {lang: [name for name, gaps in runs.items() if not paired_topic[lang]["min"] <= gaps[lang] <= paired_topic[lang]["max"]] for lang in LANGS}
    return {"shipped_file": str(config.shipped_results), "runs": runs, "outside_seed_range": outside}


def outside_base_rate(*, n: int, references: Mapping[str, Any]) -> dict[str, Any]:
    """How often an ordinary run falls outside the range of ``n`` runs, next to how often seed 0 did."""

    rows = len(MODELS) * (len(LANGS) + len(OTHER_LANGS) + len(SCALARS))
    rate = 2 / (n + 1)  # the new run is equally likely to hold any of the n + 1 ranks, and two ranks are outside
    return {"rows": rows, "chance_per_row": rate, "expected_outside": rate * rows, "outside": {name: len(row["outside_seed_range"]) for name, row in references.items()}}


def rerun_against_seed(*, references: Mapping[str, Any], models: Mapping[str, Any]) -> dict[str, Any]:
    """The one rerun pair's gap per topic score, next to the whole range over the seeds."""

    return {
        model: {
            lang: {"rerun_gap": abs(references[ORIGINAL][model]["topic"][lang] - references[RETRAIN][model]["topic"][lang]), "seed_range": models[model]["topic"][lang]["max"] - models[model]["topic"][lang]["min"]}
            for lang in LANGS
        }
        for model in MODELS
    }


def language_verdicts(*, paired_topic: Mapping[str, Any], same_weights_move: int) -> dict[str, Any]:
    """Firm, not settled or no clear winner, per language.

    Firm needs every seed on one side, the smallest margin beyond what the same weights already
    moved by, and the interval corrected over the languages clear of 0.
    """

    verdicts = {}
    for lang, row in paired_topic.items():
        low, high = row["ci95_t_corrected"]
        one_side = row["n"] in (row["delta_above_h"], row["delta_below_h"])
        smallest = min(row["by_seed"], key=abs)
        margin_clear, interval_clear = abs(smallest) > same_weights_move, low > 0 or high < 0
        verdict = "no clear winner" if not one_side else "firm" if margin_clear and interval_clear else "not settled"
        verdicts[lang] = {"verdict": verdict, "smallest_margin": smallest, "margin_beyond_same_weights_move": margin_clear, "corrected_interval_clear_of_zero": interval_clear}
    return verdicts


def check_pattern(*, seeds: Sequence[int], paired: Mapping[str, Any]) -> dict[str, Any]:
    """In every seed: Δh above h on each non-English language, English lower, less language."""

    def seeds_where(row: Mapping[str, Any], *, holds: Any) -> list[int]:
        return [seed for seed, value in zip(seeds, row["by_seed"]) if not holds(value)]

    failures = {f"topic {lang}: Δh above h": seeds_where(paired["topic"][lang], holds=lambda v: v > 0) for lang in OTHER_LANGS}
    failures["topic en: Δh below h"] = seeds_where(paired["topic"]["en"], holds=lambda v: v < 0)
    failures["language probe: Δh below h"] = seeds_where(paired["language_probe"], holds=lambda v: v < 0)
    return {
        "holds_in_every_seed": not any(failures.values()),
        "seeds_where_it_fails": failures,
        "english_difference_range": [paired["topic"]["en"]["min"], paired["topic"]["en"]["max"]],
    }


def aggregate(config: AggregationConfig) -> dict[str, Any]:
    """Everything ``summary.json`` holds."""

    per_seed, scale = read_seeds(config)
    summary = model_statistics(per_seed)
    paired = paired_statistics(per_seed)
    n = len(config.seeds)
    t = float(scipy_stats.t.ppf(0.975, n - 1))
    references = reference_rows(config=config, summary=summary)
    same_weights_move = references[ORIGINAL]["largest_topic_move_from_it"]
    frozen_move = references[ORIGINAL]["largest_frozen_move_from_it"]
    return {
        "seeds": list(config.seeds),
        "num_test_docs": NUM_DOCS,
        "interval": f"95 percent t interval for the mean over seeds, n = {n}, {n - 1} degrees of freedom, t = {t:.3f}",
        "per_seed": {str(seed): row for seed, row in per_seed.items()},
        "models": summary,
        "paired_delta_minus_h": paired,
        "scale": scale,
        "references": references,
        "seed_zero_gaps": seed_zero_gaps(config=config, references=references, paired_topic=paired["topic"]),
        "outside_base_rate": outside_base_rate(n=n, references=references),
        "rerun_against_seed": rerun_against_seed(references=references, models=summary),
        "same_weights_move": same_weights_move,
        "verdicts": language_verdicts(paired_topic=paired["topic"], same_weights_move=same_weights_move),
        "pattern": check_pattern(seeds=config.seeds, paired=paired),
        "coverage": coverage_notes(seeds=config.seeds, paired_topic=paired["topic"], same_weights_move=same_weights_move, frozen_move=frozen_move),
        "sources": {
            "scores": "seed_k/evaluation/results.json",
            "kept_epoch": "seed_k/{h,delta}/test.json, lowest held-out loss, the rule evaluate.py prints",
            "delta_size": "the line 'size of Δh relative to h_off' in seed_k/logs/evaluate.*.out; two decimals, Δh model only",
            "code": "code_used.txt in this folder: the snapshot of what ran, and what changed in the tree after the last seed job",
        },
    }


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]


def _number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _stat_cells(row: Mapping[str, Any], digits: int) -> list[str]:
    low, high = row["ci95_t"]
    return [_number(row["mean"], digits), _number(row["sd"], digits), _number(row["min"], digits), _number(row["max"], digits), f"[{_number(low, digits)}, {_number(high, digits)}]"]


def _paired_row(name: str, row: Mapping[str, Any], digits: int, n: int) -> list[str]:
    low, high = row["ci95_t"]
    side = "above 0" if low > 0 else "below 0" if high < 0 else "includes 0"
    by_seed_digits = 3 if digits == 3 else 0  # scores of 3 decimals; the rest are counts
    by_seed = ", ".join(f"{value:+.{by_seed_digits}f}" for value in row["by_seed"])
    tied = f" ({row['tied']} tied)" if row["tied"] else ""
    corrected = f"[{row['ci95_t_corrected'][0]:.1f}, {row['ci95_t_corrected'][1]:.1f}]" if "ci95_t_corrected" in row else ""
    return [name, *_stat_cells(row, digits), side, corrected, f"{row['delta_above_h']} of {n}{tied}", by_seed]


def _seed_list(seeds: Sequence[int]) -> str:
    return ", ".join(str(seed) for seed in seeds)


def _language_line(summary: Mapping[str, Any], lang: str) -> str:
    """One language's verdict and the numbers behind it."""

    n, row, verdict, move = len(summary["seeds"]), summary["paired_delta_minus_h"]["topic"][lang], summary["verdicts"][lang], summary["same_weights_move"]
    low, high = row["ci95_t_corrected"]
    side = f"Δh below h in {row['delta_below_h']} of {n} seeds" if row["delta_below_h"] > row["delta_above_h"] else f"Δh above h in {row['delta_above_h']} of {n} seeds"
    numbers = f"{side}, Δh minus h mean {row['mean']:+.1f} of {summary['num_test_docs']:,}, from {row['min']:+.0f} to {row['max']:+.0f}"
    corrected = f"interval corrected for {len(LANGS)} languages [{low:.1f}, {high:.1f}]"
    if verdict["verdict"] == "firm":
        return f"- {LANG_NAMES[lang]}: firm. {numbers}, {corrected}."
    if verdict["verdict"] == "no clear winner":
        return f"- {LANG_NAMES[lang]}: no clear winner. {numbers}, {corrected}."
    reasons = []
    if not verdict["margin_beyond_same_weights_move"]:
        reasons.append(f"the smallest margin, {verdict['smallest_margin']:+.0f}, is inside the up to {move} document difference seen for the same weights")
    if not verdict["corrected_interval_clear_of_zero"]:
        reasons.append(f"the {corrected} includes 0")
    return f"- {LANG_NAMES[lang]}: same direction in all {n}, but small and not settled: {' and '.join(reasons)}. {numbers}."


def _verdict_lines(summary: Mapping[str, Any]) -> list[str]:
    """Written from the numbers, so it cannot say more than they do."""

    n, paired, pattern = len(summary["seeds"]), summary["paired_delta_minus_h"], summary["pattern"]
    failures = pattern["seeds_where_it_fails"]
    if pattern["holds_in_every_seed"]:
        lines = [f"**The pattern holds in every one of the {n} seeds:** Δh above h on every non-English language, English lower, less language."]
    else:
        lines = ["**The pattern does not hold in every seed.** Where it fails: " + "; ".join(f"{name} fails in seed {_seed_list(seeds)}" for name, seeds in failures.items() if seeds) + "."]
    lines.append("")
    lines += [_language_line(summary, lang) for lang in (*OTHER_LANGS, "en")]
    probe = paired["language_probe"]
    lines.append(
        f"- Language probe: Δh lower than h in {probe['delta_below_h']} of {n} seeds, mean {summary['models']['delta']['language_probe']['mean']:.3f} "
        f"against {summary['models']['h']['language_probe']['mean']:.3f}, chance 0.20."
    )
    generation = paired["generation"]
    lines.append(
        f"- Not part of the pattern, generation: Δh above h in {generation['delta_above_h']} of {n} seeds, Δh minus h mean {generation['mean']:+.1f} of 200, "
        f"from {generation['min']:+.0f} to {generation['max']:+.0f}."
    )
    for name, reference in summary["references"].items():
        differences = ", ".join(f"{lang} {reference['delta_minus_h_topic'][lang]:+d}" for lang in LANGS)
        lines.append(f"- Kept out of all of the above, {name}: Δh minus h is {differences}.")
    return lines


def _coverage_lines(summary: Mapping[str, Any]) -> list[str]:
    return ["## What the spread does and does not cover", "", *[note + "\n" for note in summary["coverage"].values()]]


def _gap_section(summary: Mapping[str, Any]) -> list[str]:
    """Δh minus h from seed 0 against the seeds' range of the same gap."""

    n, paired, gaps = len(summary["seeds"]), summary["paired_delta_minus_h"]["topic"], summary["seed_zero_gaps"]
    names = list(gaps["runs"])
    rows, lines = [], []
    for lang in LANGS:
        row = paired[lang]
        values = [gaps["runs"][name][lang] for name in names]
        low, high = row["ci95_t"]
        rows.append([LANG_NAMES[lang], f"{row['mean']:+.1f}", f"[{low:.1f}, {high:.1f}]", f"{row['min']:+.0f} to {row['max']:+.0f}", *[f"{value:+d}" for value in values], f"{len(gaps['outside_seed_range'][lang])} of {len(names)}"])
        if len(gaps["outside_seed_range"][lang]) == len(names):
            where = "above" if min(values) > row["max"] else "below"
            sign = ", with the sign opposite to the mean" if where == "below" and row["mean"] > 0 else ""
            lines.append(
                f"- {LANG_NAMES[lang]}: the seed 0 gaps ({', '.join(f'{value:+d}' for value in values)}) are {where} the {n}-seed range{sign}. Quote {row['mean']:+.1f} with its interval instead."
            )
    return [
        "## Δh minus h quoted from seed 0",
        "",
        *_table(["", f"{n}-seed mean", f"95% t interval, n = {n}", f"{n}-seed range", *names, "outside the range"], rows),
        "",
        *lines,
        "",
        f"An ordinary run falls outside the range on a given row about {summary['outside_base_rate']['chance_per_row']:.2f} of the time, so this does not prove seed 0 is odd, but a gap quoted from seed 0 on these languages is not a fair stand-in for the {n}-seed mean.",
        "",
    ]


def _paired_section(summary: Mapping[str, Any]) -> list[str]:
    n, paired = len(summary["seeds"]), summary["paired_delta_minus_h"]
    header = ["", "mean", "sd", "min", "max", f"95% t interval, n = {n}", "interval against 0", f"corrected over {len(LANGS)} languages", "Δh above h", "by seed, " + _seed_list(summary["seeds"])]
    rows = [_paired_row(f"topic, {LANG_NAMES[lang]}", paired["topic"][lang], 1, n) for lang in LANGS]
    rows += [_paired_row(f"retrieval, {LANG_NAMES[lang]}", paired["retrieval"][lang], 1, n) for lang in OTHER_LANGS]
    rows += [_paired_row(PAIRED_SCALAR_NOTES[name], paired[name], 1 if name == "generation" else 3, n) for name in SCALARS]
    return ["## Paired by seed: Δh minus h", "", *_table(header, rows), "", "For the language probe and language share a negative difference favours Δh.", ""]


def _count_text(row: Mapping[str, Any]) -> str:
    return ", ".join(f"{value} in {count} of {row['n']}" for value, count in row["counts"].items())


def _model_section(summary: Mapping[str, Any]) -> list[str]:
    models, size = summary["models"], summary["models"]["delta"]["delta_size"]
    common_size = max(size["counts"], key=size["counts"].get)
    rows = []
    for model in MODELS:
        name = MODEL_NAMES[model]
        rows += [[f"{name}, topic, {LANG_NAMES[lang]}", *_stat_cells(models[model]["topic"][lang], 1)] for lang in LANGS]
        rows += [[f"{name}, retrieval, {LANG_NAMES[lang]}", *_stat_cells(models[model]["retrieval"][lang], 1)] for lang in OTHER_LANGS]
        rows += [[f"{name}, {PAIRED_SCALAR_NOTES[metric]}", *_stat_cells(models[model][metric], 1 if metric == "generation" else 3)] for metric in SCALARS]
    return [
        "## Each model over the seeds",
        "",
        *_table(["", "mean", "sd", "min", "max", f"95% t interval, n = {len(summary['seeds'])}"], rows),
        "",
        "Topic is how many of the 1,000 documents have the teacher's top topic on top. Retrieval is how many of the 1,000 translations have their own English original as nearest neighbour, "
        "on the vector each model reads (h_on for the h model, Δh for the Δh model), so it has no English row. Variance shares are of that same vector.",
        "",
        "Kept epoch, chosen by a rule, so given as counts: " + "; ".join(f"{MODEL_NAMES[model]} {_count_text(models[model]['kept_epoch'])}" for model in MODELS) + ".",
        "",
        f"|Δh| / |h_off| for the Δh model: about {common_size}, printed to 2 decimals ({_count_text(size)}). The evaluation neither saves it nor prints it for the h model.",
        "",
    ]


def _per_seed_section(summary: Mapping[str, Any]) -> list[str]:
    header = ["seed", "model", *LANGS, "language", "generation", "kept epoch", *[f"retrieval {lang}" for lang in OTHER_LANGS], "language share", "document share", SIZE_CELL]
    rows = []
    for seed, pair in summary["per_seed"].items():
        for model in MODELS:
            row = pair[model]
            rows.append([
                seed, MODEL_NAMES[model], *[row["topic"][lang] for lang in LANGS], _number(row["language_probe"], 3), row["generation"], row["kept_epoch"],
                *[row["retrieval"][lang] for lang in OTHER_LANGS], _number(row["language_share"], 3), _number(row["document_share"], 3),
                "" if row["delta_size"] is None else row["delta_size"],
            ])
    return ["## Every seed", "", *_table(header, rows), ""]


def _scale_section(summary: Mapping[str, Any]) -> list[str]:
    scale = summary["scale"]
    rows = [
        ["LDA on this text", *[scale["lda"]["topic"][lang] for lang in LANGS], ""],
        ["frozen Qwen + linear map", *[scale["frozen"]["topic"][lang] for lang in LANGS], _number(scale["frozen"]["language_probe"], 3)],
        ["always the most common topic", *[scale["majority"]] * len(LANGS), ""],
    ]
    off = scale["h_off"]
    return [
        "## For scale",
        "",
        "The same in every evaluation, because each used evaluation seed 0 and no trained model enters them.",
        "",
        *_table(["", *LANGS, "language"], rows),
        "",
        "English LDA scores 1,000 by construction: its reading of the English text is the answer.",
        "",
        "Plain Qwen's h_off: retrieval " + ", ".join(f"{lang} {off['retrieval'][lang]}" for lang in OTHER_LANGS) + f"; language share {off['language_share']:.3f}, document share {off['document_share']:.3f}.",
        "",
    ]


def _reference_section(summary: Mapping[str, Any]) -> list[str]:
    rows = []
    outside_lines = []
    for name, reference in summary["references"].items():
        for model in MODELS:
            row = reference[model]
            rows.append([
                name, MODEL_NAMES[model], *[row["topic"][lang] for lang in LANGS], _number(row["language_probe"], 3), row["generation"], row["kept_epoch"],
                *[row["retrieval"][lang] for lang in OTHER_LANGS], _number(row["language_share"], 3), _number(row["document_share"], 3),
                "" if row["delta_size"] is None else row["delta_size"],
            ])
        rows.append([name, "Δh minus h", *[f"{reference['delta_minus_h_topic'][lang]:+d}" for lang in LANGS], *[""] * 10])
        outside_lines.append(f"- {name}: outside the seeds' range on {', '.join(reference['outside_seed_range']) or 'nothing'}.")
        outside_lines.append(f"- {name}: this evaluation's trained-model topic scores differ from `{reference['compared_with']}` by up to {reference['largest_topic_move_from_it']} documents.")
    header = ["run", "model", *LANGS, "language", "generation", "kept epoch", *[f"retrieval {lang}" for lang in OTHER_LANGS], "language share", "document share", SIZE_CELL]
    n, base = len(summary["seeds"]), summary["outside_base_rate"]
    counts = sorted(base["outside"].values())
    base_lines = [
        f"A new run that behaves like the {n} seeds falls outside their range with probability 2/({n} + 1) = {base['chance_per_row']:.2f} on each row, so on about {base['expected_outside']:.1f} of these {base['rows']} rows. "
        f"Rows are correlated, and the two runs share a seed, so they count as about one draw. Seed 0 falls outside on {counts[0]} to {counts[-1]} of {base['rows']} rows, which does not show seed 0 is unusual. "
        f"The {n}-seed means are the numbers to use because they rest on more runs and seed 0 was seen first, not because seed 0 is atypical.",
        "",
    ]
    rerun = summary["rerun_against_seed"]
    rerun_rows = [[MODEL_NAMES[model], *[f"{rerun[model][lang]['rerun_gap']} / {rerun[model][lang]['seed_range']:.0f}" for lang in LANGS]] for model in MODELS]
    as_large = [f"{MODEL_NAMES[model]} {LANG_NAMES[lang]}" for model in MODELS for lang in LANGS if rerun[model][lang]["rerun_gap"] >= rerun[model][lang]["seed_range"]]
    rerun_lines = [
        f"One rerun against the seeds, topic score: gap between the two runs above / whole range over the {n} seeds. From a single rerun pair, so a rough comparison.",
        "",
        *_table(["", *LANGS], rerun_rows),
        "",
        f"The rerun gap is as large as the whole seed range on: {', '.join(as_large) or 'nothing'}.",
        "",
    ]
    return ["## Already seen, kept out of every mean", "", "Both are seed 0, which was seen before the seeds were chosen.", "", *_table(header, rows), "", *outside_lines, "", *base_lines, *rerun_lines]


def write_markdown(*, summary: Mapping[str, Any], path: Path) -> None:
    """``summary.md``: the answer first, then the tables behind it."""

    n, num_docs = len(summary["seeds"]), summary["num_test_docs"]
    lines = [
        f"# h and Δh over {n} seeds",
        "",
        f"Seeds {_seed_list(summary['seeds'])}, chosen in advance; every seed kept. {num_docs:,} test documents in each language. Interval: {summary['interval']}.",
        "",
        "## Does it hold",
        "",
        *_verdict_lines(summary),
        "",
        *_coverage_lines(summary),
        *_gap_section(summary),
        *_paired_section(summary),
        *_model_section(summary),
        *_scale_section(summary),
        *_reference_section(summary),
        *_per_seed_section(summary),
        "## Where the numbers come from",
        "",
        *[f"- {name}: {source}" for name, source in summary["sources"].items()],
        "",
        "Figures: `figures/seeds_topic_score` and `figures/seeds_paired_difference`, as PDF and PNG.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def _save(fig: Any, *, output_dir: Path, name: str) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved {figures_dir / name}.pdf and .png")


def plot_topic_dots(*, summary: Mapping[str, Any], output_dir: Path) -> None:
    """Every seed's topic score per language, with the mean, the frozen baseline and LDA."""

    n, num_docs = len(summary["seeds"]), summary["num_test_docs"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    spread = np.linspace(-0.1, 0.1, n)
    styles = {"h": (COLOR_H, "o", -0.2), "delta": (COLOR_DELTA, "s", 0.2)}
    for i, lang in enumerate(LANGS):
        for model, (color, marker, shift) in styles.items():
            values = [summary["per_seed"][str(seed)][model]["topic"][lang] for seed in summary["seeds"]]
            mean = summary["models"][model]["topic"][lang]["mean"]
            ax.scatter(i + shift + spread, values, s=34, color=color, marker=marker, edgecolors="white", linewidths=0.6, zorder=3)
            ax.hlines(mean, i + shift - 0.15, i + shift + 0.15, color=color, lw=2.5, zorder=4)
            ax.text(i + shift, max(values) + 14, f"{mean:.0f}", ha="center", va="bottom", fontsize=9, color="0.15")
        ax.scatter(i, summary["scale"]["frozen"]["topic"][lang], s=60, marker="X", color=COLOR_SCALE, edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter(i, summary["scale"]["lda"]["topic"][lang], s=70, marker="D", facecolors="white", edgecolors="black", linewidths=1.2, zorder=3)
    ax.axhline(summary["scale"]["majority"], color="0.45", lw=0.8, ls="--", zorder=1)
    ax.set(xticks=range(len(LANGS)), xticklabels=[LANG_NAMES[lang] for lang in LANGS], ylim=(0, num_docs * 1.06), xlim=(-0.5, len(LANGS) - 0.5), ylabel=f"top topic right (of {num_docs:,})")
    ax.set_title(f"Topic score: {n} seeds, {num_docs:,} test documents per language", loc="left")
    handles = [
        plt.Line2D([], [], color=COLOR_H, marker="o", ls="", markersize=6, label="μ from h, one dot per seed"),
        plt.Line2D([], [], color=COLOR_DELTA, marker="s", ls="", markersize=6, label="μ from Δh, one dot per seed"),
        plt.Line2D([], [], color="0.15", lw=2.5, label="mean over seeds (number above)"),
        plt.Line2D([], [], color=COLOR_SCALE, marker="X", ls="", markersize=7, label="frozen Qwen + linear map"),
        plt.Line2D([], [], markerfacecolor="white", markeredgecolor="black", marker="D", ls="", markersize=7, label="LDA on this text"),
        plt.Line2D([], [], color="0.45", lw=0.8, ls="--", label="always the most common topic"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _save(fig, output_dir=output_dir, name="seeds_topic_score")


def plot_paired_differences(*, summary: Mapping[str, Any], output_dir: Path) -> None:
    """Δh minus h per seed and language, with the mean and its interval. Above zero, Δh is ahead."""

    n, num_docs = len(summary["seeds"]), summary["num_test_docs"]
    paired = summary["paired_delta_minus_h"]["topic"]
    fig, ax = plt.subplots(figsize=(8, 4))
    spread = np.linspace(-0.2, 0.2, n)
    ax.axhline(0, color="0.45", lw=1, zorder=1)
    top = max(row["max"] for row in paired.values())
    for i, lang in enumerate(LANGS):
        row = paired[lang]
        ax.scatter(i - 0.08 + spread, row["by_seed"], s=34, color=COLOR_DELTA, marker="s", edgecolors="white", linewidths=0.6, zorder=3)
        ax.errorbar(i + 0.24, row["mean"], yerr=[[row["mean"] - row["ci95_t"][0]], [row["ci95_t"][1] - row["mean"]]], color="0.15", capsize=3, lw=1.2, marker="_", markersize=12, zorder=4)
        ax.text(i, top + 22, f"{row['delta_above_h']} of {n} above", ha="center", fontsize=9, color="0.15")
    ax.set(xticks=range(len(LANGS)), xticklabels=[LANG_NAMES[lang] for lang in LANGS], xlim=(-0.5, len(LANGS) - 0.5), ylabel=f"Δh minus h (documents of {num_docs:,})")
    ax.set_ylim(top=top + 50)
    ax.set_title(f"Paired by seed, {n} seeds: above zero, Δh is ahead", loc="left")
    handles = [
        plt.Line2D([], [], color=COLOR_DELTA, marker="s", ls="", markersize=6, label="one seed, 1 to 5 from left to right"),
        plt.Line2D([], [], color="0.15", marker="_", markersize=12, lw=1.2, label=f"mean and 95 percent t interval, n = {n}"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _save(fig, output_dir=output_dir, name="seeds_paired_difference")


def run_aggregation(*, config: AggregationConfig) -> dict[str, Any]:
    """Read the evaluations, write the summaries and the figures. Returns the summary."""

    summary = aggregate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "summary.json", summary, indent=1)
    print(f"wrote {config.output_dir / 'summary.json'}")
    write_markdown(summary=summary, path=config.output_dir / "summary.md")
    plot_topic_dots(summary=summary, output_dir=config.output_dir)
    plot_paired_differences(summary=summary, output_dir=config.output_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    config = parse_config(argv)
    print(asdict(config))
    run_aggregation(config=config)


if __name__ == "__main__":
    main()
