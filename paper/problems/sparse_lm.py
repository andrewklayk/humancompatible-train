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

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

# Hard-concrete stretch parameters: Louizos et al. section 4, and the values
# arXiv:2208.04425 uses.
BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1

# ``P(z != 0) = sigmoid(log_alpha + _OPEN_SHIFT)``. Derivation: ``z != 0`` iff
# ``s > -gamma/(zeta-gamma)``, and ``logit(-gamma/(zeta-gamma)) = log(-gamma/zeta)``, so
# the threshold on the logistic variable ``log u - log(1-u)`` is
# ``beta*log(-gamma/zeta) - log_alpha``. Hence the shift is ``-beta*log(-gamma/zeta)``,
# which is the ``-beta*log(-gamma/zeta)`` term in Louizos et al. Eq. (12).
_OPEN_SHIFT = -BETA * math.log(-GAMMA / ZETA)  # = (2/3)*log(11) = +1.5985968...

GRANULARITIES = ("model", "layer", "layer_split")


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


class HardConcreteGate(nn.Module):
    """A vector of independent hard-concrete gates.

    :param n_gates: Number of gates.
    :param init_open: Target ``P(z != 0)`` at initialisation. Gates start nearly open so
        the model begins dense and the constraint begins violated, which is the regime
        arXiv:2208.04425 trains in.
    :param init_std: Standard deviation of the initial ``log_alpha`` jitter. Nonzero only
        to break exact ties between gates; the per-channel loss gradient breaks them
        anyway.
    :param generator: Seeds the initialisation. Always a **CPU** generator: ``log_alpha``
        is drawn on the host and then moved, so initialisation is bitwise identical
        whatever device the model ends up on.
    """

    def __init__(
        self,
        n_gates: int,
        *,
        init_open: float = 0.95,
        init_std: float = 0.01,
        generator: Optional[torch.Generator] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        if n_gates <= 0:
            raise ValueError(f"n_gates must be positive; got {n_gates}")
        if not 0.0 < init_open < 1.0:
            raise ValueError(f"init_open must lie in (0, 1); got {init_open}")
        mean = math.log(init_open / (1.0 - init_open)) - _OPEN_SHIFT
        log_alpha = torch.empty(n_gates, dtype=dtype)
        with torch.no_grad():
            log_alpha.normal_(mean=mean, std=init_std, generator=generator)
        self.log_alpha = nn.Parameter(log_alpha.to(device))

    def __len__(self) -> int:
        return self.log_alpha.numel()

    def open_prob(self) -> Tensor:
        """``P(z != 0)``, in closed form. This is what the constraint is built from."""
        return torch.sigmoid(self.log_alpha + _OPEN_SHIFT)

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        """One reparameterised draw of ``z``, differentiable in ``log_alpha``.

        ``generator`` must live on the same device as ``log_alpha`` (use
        ``torch.Generator(device=...)``); ``None`` uses the global RNG.
        """
        u = torch.rand(
            self.log_alpha.shape,
            generator=generator,
            device=self.log_alpha.device,
            dtype=self.log_alpha.dtype,
        )
        # torch.rand samples [0, 1); the open interval is what the logit needs.
        eps = torch.finfo(self.log_alpha.dtype).eps
        u = u.clamp(eps, 1.0 - eps)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / BETA)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)

    def median(self) -> Tensor:
        """The median gate, which is the test-time model.

        ``median(logit(u)) = 0`` for ``u ~ U(0,1)`` and the stretch is monotone, so the
        median of ``s`` is ``sigmoid(log_alpha / beta)``. Note Louizos et al. Eq. (13)
        writes its test-time estimator *without* the ``/beta``; arXiv:2208.04425 says
        **medians** (their Appendix A.1), which is what this returns.
        """
        s = torch.sigmoid(self.log_alpha / BETA)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #


@dataclass
class GateGroup:
    """One gate vector, and the bookkeeping the density constraint needs.

    :param params_per_gate: ``n_j``, how many model parameters a single gate in this
        group controls. This is the weight in the parameter-counted density.
    :param repeat: How many consecutive input features of the hooked ``nn.Linear`` a
        single gate covers — 1 for an MLP channel, ``head_dim`` for an attention head.
    :param z: The gate values the forward hook will apply. Owned by :class:`GateSet`,
        which rewrites it on every ``resample`` / mode switch. Lives on the group rather
        than in a side table so the hook is a plain closure over the object it belongs
        to.
    """

    name: str
    layer: int
    kind: str  # "mlp" | "attn"
    gate: HardConcreteGate
    params_per_gate: int
    repeat: int
    z: Optional[Tensor] = field(default=None, repr=False)

    @property
    def n_gates(self) -> int:
        return len(self.gate)

    @property
    def params_total(self) -> int:
        return self.n_gates * self.params_per_gate

    def expanded_z(self) -> Tensor:
        """``z`` broadcast to the hooked layer's input width."""
        if self.z is None:
            raise RuntimeError(
                f"gate group {self.name!r} has no current sample; call "
                f"GateSet.resample() (or use_median()/use_open()) before the forward pass"
            )
        return self.z.repeat_interleave(self.repeat) if self.repeat > 1 else self.z


def _partition(groups: Sequence[GateGroup], granularity: str):
    """Group the gate groups into constraint cells; returns ``[(name, [group, ...])]``.

    ``model`` gives ``m = 1`` and ``layer``/``layer_split`` give ``m = n_layers`` and
    ``m = 2*n_layers`` — the model-wise and layer-wise granularities of their Fig. 1.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {GRANULARITIES}; got {granularity!r}"
        )
    if granularity == "model":
        return [("model", list(groups))]
    if granularity == "layer":
        cells: dict[int, list[GateGroup]] = {}
        for group in groups:
            cells.setdefault(group.layer, []).append(group)
        return [(f"layer{layer:02d}", cells[layer]) for layer in sorted(cells)]
    ordered = sorted(groups, key=lambda g: (g.layer, g.kind))
    return [(group.name, [group]) for group in ordered]


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


def _make_hook(group: GateGroup):
    """A forward *pre*-hook scaling the layer's input by this group's gates."""

    def pre_hook(module, args):
        inputs = args[0]
        z = group.expanded_z()
        return (inputs * z.to(inputs.dtype),) + tuple(args[1:])

    return pre_hook


class GateSet:
    """The gates attached to a model, plus the density constraints over them.

    The gate modules are registered as a submodule of ``model`` (default name
    ``l0_gates``), so they appear in ``model.parameters()`` and are synchronised by
    ``DistributedDataParallel`` like any other parameter. That is what makes every rank
    hold identical gate parameters, and hence identical constraint values.
    """

    def __init__(self, model: nn.Module, groups: list[GateGroup], handles, holder: str):
        self.model = model
        self.groups = groups
        self.holder = holder
        self._handles = list(handles)
        self.mode = "open"
        self.use_open()

    # -- lifecycle ---------------------------------------------------------- #

    def remove(self) -> None:
        """Detach the hooks. The gate parameters stay on the model."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # -- gate values -------------------------------------------------------- #

    def resample(self, generator: Optional[torch.Generator] = None) -> None:
        """Draw a fresh ``z`` for every group. Call once before each training forward.

        Pass a generator seeded identically on every rank to make the objective's gate
        noise shared; pass rank-dependent seeds to average over ``world_size`` draws
        instead. Those are two different estimators, not one right and one wrong — the
        shared one is what makes a multi-rank run comparable to a single-rank run at the
        pooled batch.
        """
        for group in self.groups:
            group.z = group.gate.sample(generator)
        self.mode = "sample"

    def use_median(self, *, detach: bool = True) -> None:
        """Switch to the median (test-time) gates."""
        for group in self.groups:
            z = group.gate.median()
            group.z = z.detach() if detach else z
        self.mode = "median"

    def use_open(self) -> None:
        """All-ones gates: the model with the hooks installed but inert.

        The timing reference — it pays the hook cost without applying any sparsity.
        """
        for group in self.groups:
            group.z = torch.ones(
                group.n_gates,
                device=group.gate.log_alpha.device,
                dtype=group.gate.log_alpha.dtype,
            )
        self.mode = "open"

    # -- density and constraints -------------------------------------------- #

    def densities(self, granularity: str = "layer") -> Tensor:
        """Parameter-counted expected density per constraint cell.

        ``density_g = sum_j n_j P(z_j != 0) / sum_j n_j``, the left-hand side of Eq. (3).
        Differentiable in the gate parameters, and independent of the data and of the
        current gate sample.
        """
        out = []
        for _, cell in _partition(self.groups, granularity):
            numerator = None
            denominator = 0
            for group in cell:
                weighted = group.gate.open_prob().sum() * group.params_per_gate
                numerator = weighted if numerator is None else numerator + weighted
                denominator += group.params_total
            out.append(numerator / denominator)
        return torch.stack(out)

    def constraint_names(self, granularity: str = "layer") -> list[str]:
        return [name for name, _ in _partition(self.groups, granularity)]

    def constraints(self, eps, granularity: str = "layer") -> Tensor:
        """``c = density - eps``, in the package's ``c <= 0`` convention.

        :param eps: A scalar target density, or one per constraint cell. Their section
            3.1 notes ``eps >= 1`` is a vacuous constraint, since density never exceeds 1.
        """
        density = self.densities(granularity)
        if isinstance(eps, (int, float)):
            return density - float(eps)
        target = torch.as_tensor(eps, device=density.device, dtype=density.dtype)
        if target.numel() != density.numel():
            raise ValueError(
                f"eps has {target.numel()} entries but granularity {granularity!r} "
                f"gives {density.numel()} constraints"
            )
        return density - target

    def m(self, granularity: str = "layer") -> int:
        return len(_partition(self.groups, granularity))

    # -- reporting ---------------------------------------------------------- #

    def median_report(self) -> list[dict]:
        """Per-group achieved sparsity at the median gates.

        ``expected_density`` is the constraint's own quantity; ``median_density`` is what
        the test-time model actually realises, and the two differ because the median of a
        gate is not its open probability. ``params_active`` is what a purged network would
        keep — reported rather than realised, since a perplexity number does not require
        physically slicing the weights (their Appendix D does, for a latency claim this
        experiment does not make).
        """
        rows = []
        for group in self.groups:
            with torch.no_grad():
                open_prob = group.gate.open_prob()
                median = group.gate.median()
                active = int((median > 0).sum())
            rows.append(
                {
                    "group": group.name,
                    "layer": group.layer,
                    "kind": group.kind,
                    "n_gates": group.n_gates,
                    "params_per_gate": group.params_per_gate,
                    "expected_density": float(open_prob.mean()),
                    "median_density": active / group.n_gates,
                    "params_total": group.params_total,
                    "params_active": active * group.params_per_gate,
                }
            )
        return rows

    def gate_parameters(self) -> list[nn.Parameter]:
        return [group.gate.log_alpha for group in self.groups]


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
    if hasattr(model, holder):
        raise ValueError(f"{type(model).__name__} already has an attribute {holder!r}")

    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model has no `.config`; cannot determine the head layout")
    hidden = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // n_heads)

    generator = torch.Generator().manual_seed(seed)
    groups: list[GateGroup] = []
    modules = nn.ModuleDict()
    handles = []

    def make(name, layer, kind, n_gates, params_per_gate, repeat) -> GateGroup:
        gate = HardConcreteGate(
            n_gates,
            init_open=init_open,
            init_std=init_std,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        group = GateGroup(
            name=name,
            layer=layer,
            kind=kind,
            gate=gate,
            params_per_gate=params_per_gate,
            repeat=repeat,
        )
        groups.append(group)
        modules[name.replace(".", "_")] = gate  # ModuleDict keys cannot contain "."
        return group

    for index, block in enumerate(decoder_blocks(model)):
        if gate_mlp:
            down = block.mlp.down_proj
            group = make(
                f"layer{index:02d}.mlp",
                index,
                "mlp",
                down.in_features,  # intermediate_size
                3 * hidden,  # gate_proj row + up_proj row + down_proj column
                1,
            )
            handles.append(down.register_forward_pre_hook(_make_hook(group)))
        if gate_heads:
            out_proj = block.self_attn.o_proj
            if out_proj.in_features != n_heads * head_dim:
                raise ValueError(
                    f"layer {index}: o_proj.in_features={out_proj.in_features} does not "
                    f"match num_attention_heads*head_dim={n_heads * head_dim}"
                )
            group = make(
                f"layer{index:02d}.attn",
                index,
                "attn",
                n_heads,
                2 * head_dim * hidden,  # q_proj rows + o_proj columns
                head_dim,
            )
            handles.append(out_proj.register_forward_pre_hook(_make_hook(group)))

    if not groups:
        raise ValueError("no gates were attached; the model has no decoder blocks")

    model.add_module(holder, modules)
    return GateSet(model, groups, handles, holder)


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
