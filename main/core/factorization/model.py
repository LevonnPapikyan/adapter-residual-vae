"""The topic-aligned VAE: a posterior, a fixed router, and a generator.

    document y --> posterior backbone --> readout u --> mu, log sigma^2 --> z --> prefix --> generator

There is no prompt: the model sees the document alone.

- **Posterior** ``q(z | y) = N(mu(u), diag sigma^2(u))``: a frozen language model
  with a LoRA reads the document, and two linear maps of the readout ``u`` give the
  moments. ``theta_hat = softmax(mu)`` is pulled to LDA's ``theta``, so coordinate
  ``k`` of the latent stands for LDA topic ``k``.
- **Router** ``N(0, I)``: fixed, with no parameters, so it does not appear here as a
  module; only the rate term in ``objective.py`` refers to it.
- **Generator** ``p(y | z)``: a second frozen copy with its own LoRA writes the
  document after one prefix vector made from ``z``.

The readout ``mu_from`` is ``"h"``, the backbone state at the last token with its
LoRA on, or ``"delta"``, ``h_on - h_off``, what the LoRA adds to the plain model.

What belongs here
-----------------
- Building the networks and running them.

What does NOT belong here
-------------------------
- Loss arithmetic (``objective.py``), saved-file formats (``checkpointing.py``),
  data, and optimizers.

The order of the statements in ``__init__`` fixes the order of every random draw:
tokenizer, backbone, its LoRA, generator, its LoRA, ``mu_head``, ``logvar_head``,
``z_to_prefix``. Do not reorder them, and do not reorder the module registrations,
since parameter order reaches the optimizer and the gradient clip.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from core.factorization.adaptation import apply_lora
from core.factorization.config import TopicVAEConfig
from core.factorization.objective import ObjectiveOutput, compute_objective
from core.factorization.posterior import PosteriorOutput, last_token_state, sample_z

LOGVAR_INIT = -2.0
LOGVAR_MIN, LOGVAR_MAX = -8, 4


@dataclass(frozen=True)
class ReconstructionOutput:
    """The generator's negative log-likelihood per text and real tokens per text."""

    nll: torch.Tensor
    num_tokens: torch.Tensor


class TopicVAE(nn.Module):
    """Document-only VAE whose latent is LDA's topic mixture.

    Parameters
    ----------
    config:
        Model settings. A trained model takes them from its own ``model.pt``.
    """

    def __init__(self, *, config: TopicVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.tokenizer.padding_side = "right"  # so the last real token of a text sits at mask.sum(1) - 1

        backbone = AutoModel.from_pretrained(config.base_model, dtype=torch.bfloat16).requires_grad_(False)
        self.posterior_backbone = apply_lora(backbone, lora=config.lora) if config.train_encoder_lora else backbone
        generator = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.bfloat16).requires_grad_(False)
        self.generator = apply_lora(generator, lora=config.lora, task_type="CAUSAL_LM")

        self.hidden_size = int(generator.config.hidden_size)
        self.mu_head = nn.Linear(self.hidden_size, config.num_topics)
        self.logvar_head = nn.Linear(self.hidden_size, config.num_topics)
        self.z_to_prefix = nn.Linear(config.num_topics, self.hidden_size)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, LOGVAR_INIT)
        token_std = generator.get_input_embeddings().weight.float().std().item()
        nn.init.normal_(self.z_to_prefix.weight, std=token_std / math.sqrt(config.num_topics))
        nn.init.zeros_(self.z_to_prefix.bias)
        if not config.use_generator:  # built above all the same, so the random draws match the full model
            self.generator.requires_grad_(False)
            self.z_to_prefix.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Where the trained linear maps live."""

        return self.mu_head.weight.device

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Trained parameters by learning-rate group, in optimizer order."""

        groups: dict[str, list[nn.Parameter]] = {"encoder_lora": [], "decoder_lora": [], "linear_maps": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "lora_" not in name:
                groups["linear_maps"].append(parameter)
            elif name.startswith("posterior_backbone."):
                groups["encoder_lora"].append(parameter)
            else:
                groups["decoder_lora"].append(parameter)
        return groups

    def tokenize(self, *, texts: Sequence[str], max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Token ids and attention mask, right-padded. An empty text becomes one space."""

        batch = self.tokenizer(
            [text or " " for text in texts],
            truncation=True,
            max_length=max_tokens,
            padding=True,
            return_tensors="pt",
        )
        return batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

    def backbone_state(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The backbone's last-layer state at the last real token, in float32."""

        hidden = self.posterior_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        return last_token_state(hidden_states=hidden, attention_mask=attention_mask)

    def readout(self, *, texts: Sequence[str]) -> torch.Tensor:
        """What ``mu`` is read from: ``h`` or ``h_on - h_off``, shape ``[batch, hidden]``."""

        input_ids, attention_mask = self.tokenize(texts=texts, max_tokens=self.config.encoder_max_tokens)
        with torch.set_grad_enabled(self.training and self.config.train_encoder_lora):
            state = self.backbone_state(input_ids=input_ids, attention_mask=attention_mask)
        if self.config.mu_from == "delta":
            with torch.no_grad(), self.posterior_backbone.disable_adapter():
                state_off = self.backbone_state(input_ids=input_ids, attention_mask=attention_mask)
            state = state - state_off  # outside no_grad, so the LoRA gets a gradient
        return state

    def posterior(self, *, texts: Sequence[str]) -> PosteriorOutput:
        """Moments of ``q(z | y)``: linear maps of the readout as it is, no normalization."""

        readout = self.readout(texts=texts)
        return PosteriorOutput(
            mu=self.mu_head(readout),
            logvar=self.logvar_head(readout).clamp(LOGVAR_MIN, LOGVAR_MAX),
        )

    def reconstruction_nll(self, *, z: torch.Tensor, texts: Sequence[str]) -> ReconstructionOutput:
        """``-log p(y | z)`` summed over each text's tokens, with position ``t`` predicting token ``t``."""

        if z.shape[0] != len(texts):
            raise ValueError(f"z has {z.shape[0]} rows but there are {len(texts)} texts.")
        input_ids, attention_mask = self.tokenize(texts=texts, max_tokens=self.config.decoder_max_tokens)
        embeddings = self.generator.get_input_embeddings()(input_ids)
        inputs = torch.cat([self.z_to_prefix(z)[:, None].to(embeddings.dtype), embeddings], 1)  # the prefix is one position
        mask = torch.cat([attention_mask.new_ones(len(input_ids), 1), attention_mask], 1)
        logits = self.generator(inputs_embeds=inputs, attention_mask=mask).logits[:, :-1].float()
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        return ReconstructionOutput(nll=(token_nll * attention_mask).sum(1), num_tokens=attention_mask.sum(1).float())

    def training_objective(self, *, texts: Sequence[str], theta: torch.Tensor) -> ObjectiveOutput:
        """The training loss of one batch, with one sampled ``z`` per text."""

        posterior = self.posterior(texts=texts)
        if theta.shape != posterior.mu.shape:
            raise ValueError(f"theta {tuple(theta.shape)} must have the shape of mu {tuple(posterior.mu.shape)}.")
        z = sample_z(mu=posterior.mu, logvar=posterior.logvar)
        if self.config.use_generator:
            reconstruction = self.reconstruction_nll(z=z, texts=texts)
        else:  # no reconstruction term; T still counts the generator's tokens, so beta weighs the rate as before
            _, attention_mask = self.tokenize(texts=texts, max_tokens=self.config.decoder_max_tokens)
            num_tokens = attention_mask.sum(1).float()
            reconstruction = ReconstructionOutput(nll=torch.zeros_like(num_tokens), num_tokens=num_tokens)
        return compute_objective(
            nll=reconstruction.nll,
            num_tokens=reconstruction.num_tokens,
            mu=posterior.mu,
            logvar=posterior.logvar,
            theta=theta,
            config=self.config.objective,
        )

    @torch.no_grad()
    def write_from_z(self, *, z: torch.Tensor, max_new_tokens: int = 80) -> list[str]:
        """Let the generator write from the prefix made of each row of ``z`` (sampling, top-k 50)."""

        prefix = self.z_to_prefix(z)[:, None].to(torch.bfloat16)
        output = self.generator.generate(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
