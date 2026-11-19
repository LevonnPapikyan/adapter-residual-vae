"""How one checkpoint scores, and how its states sit on a map.

What belongs here
-----------------
- The row of one checkpoint: topic score, language probe, retrieval, the variance of
  each state vector, the size of Delta h, all with the evaluation's own formulas.
- Distances of W and of the LoRA products BA to the final checkpoint.
- PCA, Procrustes, t-SNE, and the choice of steps and of the training sample.

What does NOT belong here
-------------------------
- Files, configs and the drawing (``experiments/factorization/trajectory_*.py``).

A vector that is the same for every text (Delta h at step 0, where every LoRA B is 0,
and the Delta model's mu there) has no variance to split, no probe and no PCA. It gets
``defined: False`` and ``None``, never a number made of noise.

The batches are not the evaluation's. These states come from batches per language, while
``evaluate.look_inside`` encodes all texts as one stacked list. Topic score and probe come
out the same, but bfloat16 Delta h moves with its batch: retrieval on the same weights can
differ by a few documents from what ``evaluate`` reports.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
from sklearn.manifold import TSNE
import torch

from core.analysis.language_probe import language_score
from core.analysis.representation import TextIndex, finds_original, variance_split
from core.analysis.topic_score import majority_count, topic_score

STATE_VECTORS: tuple[str, ...] = ("h_off", "h_on", "delta", "mu")
_LORA_A = re.compile(r"^(enc|dec)\..*layers\.(\d+)\.self_attn\.(\w+)\.lora_A\.default\.weight$")


def is_constant(features: np.ndarray) -> bool:
    """Whether every row is the same."""

    return bool(np.ptp(features, axis=0).max() == 0)


def read_vector_name(mu_from: str) -> str:
    """Which state vector ``mu`` is read from."""

    return {"h": "h_on", "delta": "delta"}[mu_from]


def state_vectors(*, on: np.ndarray, off: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
    """The four vectors of ``STATE_VECTORS``, by name."""

    return {"h_off": off, "h_on": on, "delta": on - off, "mu": mu}


def theta_from_mu(mu: np.ndarray) -> np.ndarray:
    """``theta_hat = softmax(mu)``, in the float format of ``mu``, as the evaluation computes it."""

    return torch.softmax(torch.from_numpy(mu), -1).numpy()


def per_language(*, features: np.ndarray, index: TextIndex, langs: Sequence[str]) -> dict[str, np.ndarray]:
    """Split a stacked ``[num_langs * num_docs, ...]`` array into its language blocks."""

    return {lang: features[j * index.num_docs : (j + 1) * index.num_docs] for j, lang in enumerate(langs)}


def vector_row(*, features: np.ndarray, index: TextIndex, langs: Sequence[str]) -> dict[str, object]:
    """Variance split, absolute variance per text, and retrieval of one state vector."""

    row: dict[str, object] = {"defined": not is_constant(features), "value_max": float(np.abs(features).max())}
    if not row["defined"]:
        return row
    split = variance_split(features=features, index=index)
    total = float(((features - features.mean(0)) ** 2).sum(1).mean())
    row.update(
        language_share=split.language,
        document_share=split.document,
        variance_total=total,
        variance_language=split.language * total,
        variance_document=split.document * total,
        finds_original={lang: finds_original(features=features, lang_index=j, index=index) for j, lang in enumerate(langs) if j},
    )
    return row


def checkpoint_row(
    *,
    on: np.ndarray,
    off: np.ndarray,
    mu: np.ndarray,
    target: np.ndarray,
    index: TextIndex,
    langs: Sequence[str],
    read: str,
) -> dict[str, object]:
    """Every score of one checkpoint in one float format, on the stacked test texts."""

    vectors = state_vectors(on=on, off=off, mu=mu)
    delta = vectors["delta"]
    thetas = per_language(features=theta_from_mu(mu), index=index, langs=langs)
    rows = {name: vector_row(features=vectors[name], index=index, langs=langs) for name in STATE_VECTORS}
    return {
        "topic": {lang: topic_score(theta_pred=thetas[lang], target=target) for lang in langs},
        "language_probe": None if is_constant(mu) else language_score(thetas=thetas),
        "read": read,
        "retrieval": rows[read].get("finds_original"),
        "vectors": rows,
        "delta_ratio": float(np.mean(np.linalg.norm(delta, axis=1) / np.linalg.norm(off, axis=1))),
        "delta_max": float(np.abs(delta).max()),
        "delta_zero_fraction": float((delta == 0).mean()),
    }


def training_row(*, mu: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """Topic score of a checkpoint on the English training sample, against the teacher on those posts."""

    return {
        "right": topic_score(theta_pred=theta_from_mu(mu), target=target),
        "total": len(target),
        "majority": majority_count(target=target),
    }


def _lora_factors(weights: Mapping[str, torch.Tensor], part: str) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    factors = {}
    for name, a in weights.items():
        match = _LORA_A.match(name)
        if match and match.group(1) == part:
            factors[int(match.group(2)), match.group(3)] = (weights[name.replace("lora_A", "lora_B")].double(), a.double())
    return factors


def _ba_inner(p: Mapping, q: Mapping) -> float:
    """``<B1 A1, B2 A2>`` summed over projections, in the small factors, without forming the products."""

    return float(sum(((p[k][0].T @ q[k][0]) * (q[k][1] @ p[k][1].T).T).sum() for k in p))


def lora_distance(*, weights: Mapping[str, torch.Tensor], final: Mapping[str, torch.Tensor], part: str) -> dict[str, float | None]:
    """``|BA|`` of one LoRA (``part`` is ``enc`` or ``dec``) and its distance to the final one, without the factor alpha / rank = 2."""

    now, last = _lora_factors(weights, part), _lora_factors(final, part)
    now_sq, last_sq, inner = _ba_inner(now, now), _ba_inner(last, last), _ba_inner(now, last)
    distance = float(np.sqrt(max(now_sq + last_sq - 2 * inner, 0.0)))
    return {"norm": float(np.sqrt(now_sq)), "distance": distance, "relative": distance / np.sqrt(last_sq) if last_sq > 0 else None}


def weights_row(*, weights: Mapping[str, torch.Tensor], final: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """``|W|``, the LoRAs' ``|BA|``, and each one's distance to the final checkpoint. Saved tensor names."""

    w, w_final = weights["mu_head.weight"].double(), final["mu_head.weight"].double()
    distance = float((w - w_final).norm())
    return {
        "W": {"norm": float(w.norm()), "distance": distance, "relative": distance / float(w_final.norm())},
        "encoder_BA": lora_distance(weights=weights, final=final, part="enc"),
        "decoder_BA": lora_distance(weights=weights, final=final, part="dec"),
    }


def pca_basis(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean, and the principal directions as rows."""

    features = features.astype(np.float64)
    mean = features.mean(0)
    _, _, components = np.linalg.svd(features - mean, full_matrices=False)
    return mean, components


def project(features: np.ndarray, *, mean: np.ndarray, components: np.ndarray, dims: int = 2) -> np.ndarray:
    """Coordinates of ``features`` on the first ``dims`` directions of a basis."""

    return (features.astype(np.float64) - mean) @ components[:dims].T


def procrustes_to(coords: np.ndarray, *, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """``coords`` rotated or reflected onto ``reference``, no scaling, and the normalized disparity in [0, 1]. Both centered."""

    u, s, vt = np.linalg.svd(coords.T @ reference)
    disparity = max(0.0, 1 - s.sum() ** 2 / ((coords**2).sum() * (reference**2).sum()))  # rounding can dip below 0
    return coords @ (u @ vt), float(disparity)


def variance_in_plane(features: np.ndarray, *, mean: np.ndarray, components: np.ndarray) -> float:
    """The share of the variance of ``features`` that the plane of a basis's first two directions shows."""

    features = features.astype(np.float64)
    coords = project(features, mean=mean, components=components)
    return float(((coords - coords.mean(0)) ** 2).sum() / ((features - features.mean(0)) ** 2).sum())


def effective_perplexity(*, num_points: int, requested: float) -> float:
    """The requested perplexity, or a third of the points when there are too few (smoke tests)."""

    return float(min(requested, max(1.0, (num_points - 1) / 3)))


def tsne_map(*, features: np.ndarray, perplexity: float, seed: int, pca_dims: int) -> np.ndarray:
    """t-SNE on the first PCA features, from a PCA start. Axes and distances are not comparable between calls."""

    mean, components = pca_basis(features)
    reduced = project(features, mean=mean, components=components, dims=pca_dims)
    return TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed).fit_transform(reduced)


def nearest_saved_steps(*, wanted: Sequence[int], saved: Sequence[int]) -> list[int]:
    """The saved step nearest each wanted one, the lower on a tie, without repeats, in order."""

    return sorted({min(saved, key=lambda step: (abs(step - target), step)) for target in wanted})


def training_sample_indices(*, num_texts: int, size: int, seed: int) -> np.ndarray:
    """A fixed random sample of positions in the training split, sorted. All of them if there are fewer."""

    return np.sort(np.random.default_rng(seed).permutation(num_texts)[: min(size, num_texts)])


REFERENCE_TOLERANCES = {"topic": 15, "language_probe": 0.03, "retrieval": 20, "variance_share": 0.02}


def _difference(value: float, reference: float, tolerance: float) -> dict[str, object]:
    return {"this": value, "reference": reference, "difference": value - reference, "within_tolerance": bool(abs(value - reference) <= tolerance)}


def compare_row_to_reference(
    *,
    row: Mapping[str, object],
    reference_topic: Mapping[str, int],
    reference_language: float,
    reference_inside: Mapping[str, object],
    langs: Sequence[str],
) -> dict[str, object]:
    """One checkpoint's row minus a shipped model's numbers, with the agreed tolerances."""

    read = row["vectors"][row["read"]]
    other_langs = list(langs)[1:]
    return {
        "tolerances": REFERENCE_TOLERANCES,
        "batching": "this row is batched per language; the shipped retrieval and shares come from one stacked list, as evaluate.look_inside does",
        "topic": {lang: _difference(row["topic"][lang], reference_topic[lang], REFERENCE_TOLERANCES["topic"]) for lang in langs},
        "language_probe": _difference(row["language_probe"], reference_language, REFERENCE_TOLERANCES["language_probe"]),
        "retrieval": {
            lang: _difference(read["finds_original"][lang], found, REFERENCE_TOLERANCES["retrieval"])
            for lang, found in zip(other_langs, reference_inside["finds_original"])
        },
        "language_share": _difference(read["language_share"], reference_inside["language"], REFERENCE_TOLERANCES["variance_share"]),
        "document_share": _difference(read["document_share"], reference_inside["document"], REFERENCE_TOLERANCES["variance_share"]),
    }
