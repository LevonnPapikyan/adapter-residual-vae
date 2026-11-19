"""The arithmetic: objective, schedule, key mapping, scores. No language model involved."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from core.analysis.comparison import compare_results
from core.analysis.language_probe import language_score
from core.analysis.linear_map import fit_linear_map
from core.analysis.representation import TextIndex, finds_original, variance_split
from core.analysis.topic_matching import agreement_counts, map_onto, match_topics
from core.analysis.topic_score import majority_count, most_common_floor, topic_score, topk_counts
from core.checkpoints.io import load_model_file, save_model_file
from core.factorization.checkpointing import (
    load_trained_run,
    module_name_to_saved_name,
    saved_name_to_module_name,
)
from core.factorization.config import ObjectiveConfig
from core.factorization.objective import compute_objective
from core.factorization.posterior import last_token_state
from core.runs.io import refuse_protected_path, write_json
from core.train.loop import count_steps_per_epoch, split_held_out
from core.train.lr_schedules import warmup_cosine_multiplier, warmup_steps_for


def test_objective_matches_hand_computation() -> None:
    mu = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    logvar = torch.zeros(2, 2)
    theta = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    nll = torch.tensor([10.0, 30.0])
    num_tokens = torch.tensor([10.0, 30.0])
    output = compute_objective(nll=nll, num_tokens=num_tokens, mu=mu, logvar=logvar, theta=theta, config=ObjectiveConfig(lam=2.0, beta=0.5))

    rate = 0.5 * np.array([1.0, 4.0])
    log_softmax_0 = np.log(np.array([math.e, 1.0]) / (math.e + 1))
    log_softmax_1 = np.log(np.array([1.0, math.e**2]) / (1 + math.e**2))
    align = np.array([-log_softmax_0[0], -0.5 * log_softmax_1.sum()])
    expected = (np.array([10.0, 30.0]) + 0.5 * rate).mean() / 20.0 + 2.0 * align.mean()
    assert output.loss.item() == pytest.approx(expected)
    assert output.rec == pytest.approx(20.0 / 20.0)
    assert output.align == pytest.approx(align.mean())
    assert output.rate == pytest.approx(rate.mean())
    assert list(output.stats()) == ["rec", "align", "rate"]


def test_objective_refuses_mismatched_theta() -> None:
    with pytest.raises(ValueError, match="theta"):
        compute_objective(
            nll=torch.zeros(2),
            num_tokens=torch.ones(2),
            mu=torch.zeros(2, 3),
            logvar=torch.zeros(2, 3),
            theta=torch.zeros(2, 4),
            config=ObjectiveConfig(),
        )


def test_last_token_state_picks_the_last_real_token() -> None:
    hidden = torch.arange(12.0).reshape(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert last_token_state(hidden_states=hidden, attention_mask=mask).tolist() == [[2.0, 3.0], [10.0, 11.0]]


def test_schedule_is_the_final_runs_schedule() -> None:
    assert warmup_steps_for(total_steps=2200) == 110
    assert warmup_steps_for(total_steps=3) == 1
    for step in (0, 1, 109, 110, 1100, 2199):
        warm, total = 110, 2200
        flat = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
        assert warmup_cosine_multiplier(step, warmup_steps=warm, total_steps=total) == flat


def test_loop_counts_and_split() -> None:
    assert count_steps_per_epoch(num_texts=8793, batch_size=16) == 550
    split = split_held_out(rng=np.random.default_rng(0), num_texts=20, val_frac=0.1)
    assert len(split.held_out) == 2 and len(split.kept) == 18 and not set(split.held_out) & set(split.kept)
    with pytest.raises(ValueError, match="holds out nothing"):
        split_held_out(rng=np.random.default_rng(0), num_texts=5, val_frac=0.1)


@pytest.mark.parametrize(
    ("saved", "module"),
    [
        ("enc.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", "posterior_backbone.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"),
        ("dec.base_model.model.model.layers.1.self_attn.o_proj.lora_B.default.weight", "generator.base_model.model.model.layers.1.self_attn.o_proj.lora_B.default.weight"),
        ("mu_head.weight", "mu_head.weight"),
        ("logvar_head.bias", "logvar_head.bias"),
        ("prefix.weight", "z_to_prefix.weight"),
    ],
)
def test_key_mapping_round_trip(saved: str, module: str) -> None:
    assert saved_name_to_module_name(saved) == module
    assert module_name_to_saved_name(module) == saved


def test_unknown_saved_key_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown saved tensor"):
        saved_name_to_module_name("norm.weight")


def test_trained_with_layernorm_is_refused(tmp_path: Path) -> None:
    save_model_file(tmp_path / "model.pt", config={"no_norm": False}, weights={"norm.weight": torch.zeros(1), "norm.bias": torch.zeros(1)})
    with pytest.raises(ValueError, match="no_norm=False"):
        load_trained_run(run_dir=tmp_path, device="cpu")


def test_model_file_is_never_overwritten(tmp_path: Path) -> None:
    save_model_file(tmp_path / "model.pt", config={}, weights={})
    with pytest.raises(FileExistsError):
        save_model_file(tmp_path / "model.pt", config={}, weights={})
    saved = load_model_file(tmp_path / "model.pt")
    assert saved.config == {} and saved.weights == {}
    with pytest.raises(FileNotFoundError):
        load_model_file(tmp_path / "none.pt")


def test_existing_results_are_not_replaced_unless_asked(tmp_path: Path) -> None:
    from experiments.factorization.output import refuse_existing_results

    refuse_existing_results(output_dir=tmp_path, overwrite=False)  # nothing there yet
    write_json(tmp_path / "results.json", {})
    with pytest.raises(FileExistsError, match="overwrite"):
        refuse_existing_results(output_dir=tmp_path, overwrite=False)
    refuse_existing_results(output_dir=tmp_path, overwrite=True)


def test_protected_path_guard(tmp_path: Path) -> None:
    protected = tmp_path / "runs"
    with pytest.raises(ValueError, match="read-only"):
        refuse_protected_path(protected / "final" / "x", protected_dirs=[protected])
    assert refuse_protected_path(tmp_path / "outputs", protected_dirs=[protected]) == (tmp_path / "outputs").resolve()


def test_write_json_keeps_non_ascii_and_order(tmp_path: Path) -> None:
    write_json(tmp_path / "a" / "x.json", {"μ from Δh": 1, "b": {"c": 2}})
    assert (tmp_path / "a" / "x.json").read_text(encoding="utf-8") == '{\n "μ from Δh": 1,\n "b": {\n  "c": 2\n }\n}'


def test_topic_scores() -> None:
    target = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.2, 0.1, 0.7]])
    pred = np.array([[0.5, 0.4, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    assert topic_score(theta_pred=pred, target=target) == 2
    assert topk_counts(theta_pred=pred, target=target) == [2, 3, 4]
    assert majority_count(target=target) == 2
    assert most_common_floor(target=target) == [2, 3, 4]
    with pytest.raises(ValueError, match="shape"):
        topic_score(theta_pred=pred[:2], target=target)


def test_text_index_and_representation_probes() -> None:
    index = TextIndex(num_langs=2, num_docs=3)
    assert index.text_lang.tolist() == [0, 0, 0, 1, 1, 1]
    assert index.text_doc.tolist() == [0, 1, 2, 0, 1, 2]
    docs = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
    features = np.concatenate([docs, docs + np.array([0.1, 0.1])], dtype=np.float32)
    assert finds_original(features=features, lang_index=1, index=index) == 3
    split = variance_split(features=features, index=index)
    assert split.document > 0.99 and split.language < 0.01
    with pytest.raises(ValueError, match="rows"):
        finds_original(features=features[:3], lang_index=1, index=index)


def test_language_probe_sees_a_planted_signal() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(40, 4))
    separable = {"a": noise[:20] + 5.0, "b": noise[20:] - 5.0}
    assert language_score(thetas=separable) == 1.0


def test_linear_map_learns_a_simple_rule() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3)).astype(np.float32)
    theta = np.eye(2, dtype=np.float32)[(features[:, 0] > 0).astype(int)]
    fitted = fit_linear_map(features=features, theta=theta, device="cpu", steps=200)
    assert (fitted.predict(features=features).argmax(1) == theta.argmax(1)).mean() > 0.95


def test_topic_matching_finds_a_permutation() -> None:
    rng = np.random.default_rng(0)
    theta_a = rng.dirichlet(np.ones(4), size=200)
    order = np.array([2, 0, 3, 1])
    theta_b = theta_a[:, order]
    match = match_topics(theta_a=theta_a, theta_b=theta_b)
    assert match.to_a.tolist() == order.tolist()
    assert np.allclose(match.correlation, 1.0)
    mapped = map_onto(theta_b=theta_b, to_a=match.to_a)
    assert np.allclose(mapped, theta_a)
    counts = agreement_counts(theta_a=theta_a, theta_b_mapped=mapped)
    assert counts["top_topic"] == 200 and counts["same_pair"] == 200


def test_comparison_is_a_difference() -> None:
    base = {
        "topic": {"m": {"en": 10, "de": 5}},
        "language": {"m": 0.5},
        "generation": {"m": {"0": 3, "1": 4}},
        "inside": {"r": {"finds_original": [1, 2], "language": 0.1, "document": 0.2}},
        "majority": 7,
        "layers": [{"topic": {"en": 1}, "language": 0.3, "finds_original": 2.0}],
    }
    assert compare_results(results=base, reference=base)["topic"]["m"] == {"en": 0, "de": 0}
    moved = {**base, "topic": {"m": {"en": 12, "de": 4}}, "generation": {}}
    comparison = compare_results(results=moved, reference=base)
    assert comparison["topic"]["m"] == {"en": 2, "de": -1}
    assert comparison["generation_total"] == {}
