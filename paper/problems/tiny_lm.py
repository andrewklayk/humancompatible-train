"""
A small decoder LM with the attribute layout :mod:`paper.problems.sparse_lm` expects.

``transformers`` is installed neither in this project's environment nor as a module on
the cluster, and the real E3 model (``Qwen/Qwen2.5-0.5B``) is not in the local
HuggingFace cache. This stand-in exists so that everything *except* the pretrained
weights — the gates, the density constraints, the dual optimizers, the distributed
reduction, the timing instrumentation and the artifact writing — can be exercised and
unit-tested on a CPU-only machine.

It deliberately mirrors the Llama/Qwen layout rather than being a generic transformer:
``model.model.layers[i].mlp.{gate_proj,up_proj,down_proj}`` and
``model.model.layers[i].self_attn.{q_proj,k_proj,v_proj,o_proj}``, a ``.config`` carrying
``hidden_size`` / ``intermediate_size`` / ``num_attention_heads`` / ``num_key_value_heads``
/ ``num_hidden_layers``, and a ``forward(input_ids=..., labels=...)`` returning an object
with a ``.loss``. Anything relying on those names works identically here and on the real
model, which is the point.

This is a test fixture, not a model anyone should train: no rotary embeddings, no KV
cache, no attention mask beyond the causal one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TinyConfig:
    """The subset of a HuggingFace config the gate code reads."""

    vocab_size: int = 256
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class CausalLMOutput:
    """Stands in for ``transformers.modeling_outputs.CausalLMOutputWithPast``."""

    loss: Optional[Tensor]
    logits: Tensor


class TinyMLP(nn.Module):
    """SwiGLU MLP, named as Llama/Qwen name it."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyAttention(nn.Module):
    """Grouped-query self-attention, named as Llama/Qwen name it.

    ``num_key_value_heads < num_attention_heads`` on purpose: it is what makes gating
    *query* heads the only structurally sound choice, and this fixture should exercise
    that rather than hide it.
    """

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv, self.head_dim).transpose(1, 2)
        repeat = self.n_heads // self.n_kv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # (batch, heads, seq, head_dim) -> (batch, seq, heads*head_dim). The head index is
        # the outer one here, which is exactly why a per-head gate expands by
        # repeat_interleave(head_dim) in sparse_lm.
        out = out.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TinyBlock(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.self_attn = TinyAttention(config)
        self.mlp = TinyMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyDecoder(nn.Module):
    """The ``.model`` attribute: embeddings, blocks, final norm."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layers = nn.ModuleList(
            TinyBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        seq = input_ids.shape[1]
        positions = torch.arange(seq, device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyCausalLM(nn.Module):
    """``AutoModelForCausalLM``-shaped wrapper: ``.model``, ``.config``, ``.lm_head``."""

    def __init__(self, config: Optional[TinyConfig] = None):
        super().__init__()
        self.config = config or TinyConfig()
        self.model = TinyDecoder(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        **_unused,
    ) -> CausalLMOutput:
        logits = self.lm_head(self.model(input_ids))
        loss = None
        if labels is not None:
            # Same shift HuggingFace applies: predict token t+1 from tokens <= t.
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)


def tiny_causal_lm(seed: int = 0, **config_kwargs) -> TinyCausalLM:
    """A seeded :class:`TinyCausalLM`, so tests and smoke runs are reproducible."""
    torch.manual_seed(seed)
    return TinyCausalLM(TinyConfig(**config_kwargs))
