"""This run's results against a reference ``results.json``.

What belongs here
-----------------
- Differences between two results dictionaries of the same schema.

What does NOT belong here
-------------------------
- Reading or writing them.

The topic and language differences of the two trained models, and of LDA on the
text, are expected to be 0 on the same hardware and software: they involve no
random draw. The frozen map starts from seeded random weights, and generation
samples, so those also move with hardware and software versions.
"""

from __future__ import annotations

from typing import Any, Mapping


def compare_results(*, results: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``results`` minus ``reference`` for every number both hold."""

    comparison: dict[str, Any] = {
        "majority": results["majority"] - reference["majority"],
        "topic": {
            name: {lang: scores[lang] - reference["topic"][name][lang] for lang in scores}
            for name, scores in results["topic"].items()
            if name in reference["topic"]
        },
        "language": {
            name: value - reference["language"][name]
            for name, value in results["language"].items()
            if name in reference["language"]
        },
        "generation_total": {
            name: sum(scores.values()) - sum(reference["generation"][name].values())
            for name, scores in results["generation"].items()
            if name in reference.get("generation", {})
        },
        "inside": {
            name: {
                "finds_original": [a - b for a, b in zip(values["finds_original"], reference["inside"][name]["finds_original"])],
                "language": values["language"] - reference["inside"][name]["language"],
                "document": values["document"] - reference["inside"][name]["document"],
            }
            for name, values in results["inside"].items()
            if name in reference.get("inside", {})
        },
    }
    if results.get("layers") and reference.get("layers"):
        comparison["layers"] = [
            {
                "topic": {lang: row["topic"][lang] - ref["topic"][lang] for lang in row["topic"]},
                "language": row["language"] - ref["language"],
                "finds_original": row["finds_original"] - ref["finds_original"],
            }
            for row, ref in zip(results["layers"], reference["layers"])
        ]
    return comparison
