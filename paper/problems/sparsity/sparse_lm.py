"""
Structured L0 gates and the density constraints of arXiv:2208.04425.

Implements Eq. (3) of Gallego-Posada, Ramirez, Erraqabi, Bengio & Lacoste-Julien,
*Controlled Sparsity via Constrained Optimization* (NeurIPS 2022):

    min_{theta,phi}  E_{z|phi}[ L_D(theta * z) ]
    s.t.             E_{z_g|phi_g}[ ||z_g||_0 ] / #(theta_g)  <=  eps_g,   g in [1:G]

with the hard-concrete gates of Louizos, Welling & Kingma (2018). Three things here are
easy to get subtly wrong, so each is stated once and then enforced by code.

**Gates are structured, not per-weight.** Their section 2 ("Parameter grouping") puts one
gate per *input neuron* of a fully connected layer. For a decoder LM that is one gate per
MLP intermediate channel and one per attention head — order 1e5 gate parameters, against
the 1e9 a per-weight variant would need, which their section 2 notes would double the
trainable parameter count.

**The density denominator counts parameters, not gates.** ``#(theta_g)`` is a parameter
count, so a gate must be weighted by how many parameters it controls: an MLP-channel gate
covers its rows of ``gate_proj``/``up_proj`` and its column of ``down_proj``
(``3 * hidden``), an attention-head gate covers its rows of ``q_proj`` and its columns of
``o_proj`` (``2 * head_dim * hidden``). Only weight matrices are counted; the per-head
slice of an attention bias is ``head_dim`` parameters and is ignored. For equally sized
gates the weighting collapses to the mean open probability, which is the degenerate case
:meth:`GateSet.densities` is unit-tested against.

**The constraint is a closed form in the gate parameters — no data enters it.**
``P(z_j != 0) = sigmoid(log_alpha_j - beta*log(-gamma/zeta))`` exactly, so the constraint
vector needs no sampling and carries no minibatch noise. That is a real property of this
problem rather than an approximation, but it also means a data-parallel reduction over
this constraint has nothing to pool — see the E3 section of ``paper/README.md``.

The sampled gates *are* used in the objective, so :meth:`GateSet.resample` must be called
before every forward pass whose gradient will be taken. Reusing one sample across two
backward passes raises from autograd, which is the intended failure mode rather than a
silent wrong answer.

Gate parameters are kept in float32 even when the model runs in bfloat16: there are only
about 1e5 of them, and ``sigmoid``/``logit`` round badly in bf16. The cast to the
activation dtype happens where the gate is applied.
"""

import torch
from torch import nn
from typing import Optional
from .sparsity_gates import GateParamGroup, GateSet, HardConcreteGate


# --------------------------------------------------------------------------- #
# attaching gates to a decoder LM
# --------------------------------------------------------------------------- #


def decoder_blocks(model: nn.Module):
    """The transformer blocks of a HuggingFace-shaped decoder LM.

    Resolved by attribute layout rather than by importing ``transformers``, so this works
    for Llama/Qwen/Mistral-shaped models and for the local stand-in in
    :mod:`paper.problems.tiny_lm` — which is what lets the gates, the constraints and the
    whole training loop be tested without ``transformers`` installed.
    """
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise TypeError(
            f"{type(model).__name__} has no `.model.layers`; expected a "
            f"HuggingFace-shaped decoder LM"
        )
    return layers

def attach_gates(
    model: nn.Module,
    *,
    gate_mlp: bool = True,
    gate_heads: bool = True,
    init_open: float = 0.95,
    init_std: float = 0.01,
    seed: int = 0,
    holder: str = "l0_gates",
    device=None,
    dtype=torch.float32,
) -> GateSet:
    """Install structured hard-concrete gates on a decoder LM.

    Gates are applied by ``register_forward_pre_hook`` on ``mlp.down_proj`` and
    ``self_attn.o_proj``, scaling their *input*. That needs no model surgery and works for
    any Llama/Qwen-shaped model, at the cost of one elementwise multiply per block.

    Attention gating covers **query heads only**. Qwen2.5-0.5B has 14 query heads to 2
    key/value heads, so a key/value head is shared 7:1 and is not a per-head structured
    unit; gating a query head drops its slice of ``q_proj``'s output and the matching
    columns of ``o_proj``.
    """
    if not (gate_mlp or gate_heads):
        raise ValueError("at least one of gate_mlp / gate_heads must be True")

    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model has no `.config`; cannot determine the head layout")
    hidden = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // n_heads)

    generator = torch.Generator().manual_seed(seed)
    groups: list[GateParamGroup] = []

    def make(name, layer, kind, module, n_gates, params_per_gate, repeat=1) -> None:
        groups.append(
            GateParamGroup(
                name=name,
                layer=layer,
                kind=kind,
                module=module,
                gate=HardConcreteGate(
                    n_gates,
                    init_open=init_open,
                    init_std=init_std,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                ),
                params_per_gate=params_per_gate,
                repeat=repeat,
            )
        )

    for index, block in enumerate(decoder_blocks(model)):
        if gate_mlp:
            down = block.mlp.down_proj
            # One gate per intermediate channel; it owns a gate_proj row, an up_proj row
            # and a down_proj column.
            make(f"layer{index:02d}.mlp", index, "mlp", down,
                 down.in_features, 3 * hidden)
        if gate_heads:
            out_proj = block.self_attn.o_proj
            if out_proj.in_features != n_heads * head_dim:
                raise ValueError(
                    f"layer {index}: o_proj.in_features={out_proj.in_features} does not "
                    f"match num_attention_heads*head_dim={n_heads * head_dim}"
                )
            # One gate per query head; it owns q_proj rows and o_proj columns.
            make(f"layer{index:02d}.attn", index, "attn", out_proj,
                 n_heads, 2 * head_dim * hidden, repeat=head_dim)

    if not groups:
        raise ValueError("no gates were attached; the model has no decoder blocks")
    return GateSet(model, groups, holder)


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #


def trainable_bytes(model: nn.Module) -> int:
    """Bytes a gradient all-reduce moves per step, i.e. the ``O(n)`` term."""
    return sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )


def describe(model: nn.Module, gate_set: Optional[GateSet] = None) -> dict:
    """Parameter and gate counts, for the record kept alongside every run."""
    total = sum(p.numel() for p in model.parameters())
    gate_params = 0 if gate_set is None else sum(len(g.gate) for g in gate_set.groups)
    gated = 0 if gate_set is None else sum(g.params_total for g in gate_set.groups)
    return {
        "params_total": total,
        "params_gated": gated,
        # The share of the model the density constraint can actually reach. The rest
        # (embeddings, norms) is why the density denominator runs over gated groups only:
        # a whole-model denominator would put a hard floor under every target. For
        # Qwen2.5-0.5B the tied 151936x896 embedding alone is 27.5% of the parameters.
        "gated_fraction": gated / total if total else 0.0,
        "gate_params": gate_params,
    }
