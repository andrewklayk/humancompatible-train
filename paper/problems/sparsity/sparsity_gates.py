
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

    def open_proba(self) -> Tensor:
        """``P(z != 0)``, in closed form. This is what the constraint is built from."""
        return torch.sigmoid(self.log_alpha + _OPEN_SHIFT)

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        """One draw of ``z``.

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
class GateParamGroup:
    """One gate vector, the module it gates, and the bookkeeping the density needs.

    :param module: The module whose *input* this group scales. :class:`GateSet` registers
        :meth:`pre_hook` on it; nothing else reads it.
    :param params_per_gate: ``n_j``, how many model parameters a single gate in this
        group controls. This is the weight in the parameter-counted density.
    :param repeat: How many consecutive input features of the hooked module a single gate
        covers — 1 for an MLP or convolution channel, ``head_dim`` for an attention head.
    :param dim: The activation axis the gates index: ``-1`` for a channels-last ``Linear``
        input, ``1`` for an ``NCHW`` convolution input.
    :param z: The gate values the forward hook will apply. Owned by :class:`GateSet`,
        which rewrites it on every ``resample`` / mode switch.
    """

    name: str
    layer: int
    kind: str  # "mlp" | "attn" | "conv<n>"
    gate: HardConcreteGate
    module: nn.Module = field(repr=False)
    params_per_gate: int
    repeat: int = 1
    dim: int = -1
    z: Optional[Tensor] = field(default=None, repr=False)

    @property
    def n_gates(self) -> int:
        return len(self.gate)

    @property
    def params_total(self) -> int:
        return self.n_gates * self.params_per_gate

    def expanded_z(self) -> Tensor:
        """``z`` broadcast to the hooked module's input width."""
        if self.z is None:
            raise RuntimeError(
                f"gate group {self.name!r} has no current sample; call "
                f"GateSet.resample() (or use_median()/use_open()) before the forward pass"
            )
        return self.z.repeat_interleave(self.repeat) if self.repeat > 1 else self.z

    def pre_hook(self, module: nn.Module, args):
        """Scale the hooked module's input by the current gates.
        """
        x = args[0]
        shape = [1] * x.ndim
        shape[self.dim] = -1
        return (x * self.expanded_z().to(x.dtype).view(shape),) + tuple(args[1:])


def _partition(groups: Sequence[GateParamGroup], granularity: str):
    """Group the gate groups into constraints; returns ``[(name, [group, ...])]``.

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
        cells: dict[int, list[GateParamGroup]] = {}
        for group in groups:
            cells.setdefault(group.layer, []).append(group)
        return [(f"layer{layer:02d}", cells[layer]) for layer in sorted(cells)]
    ordered = sorted(groups, key=lambda g: (g.layer, g.kind))
    return [(group.name, [group]) for group in ordered]



class GateSet:
    """All gates attached to a model, plus the constraint calculation over them.

    The gate modules are registered as a submodule of ``model`` (default name
    ``l0_gates``), so they appear in ``model.parameters()`` and are synchronised by
    ``DistributedDataParallel`` like any other parameter. That is what makes every rank
    hold identical gate parameters, and hence identical constraint values.
    """

    def __init__(self, model: nn.Module, groups: list[GateParamGroup],
                 holder: str = "l0_gates"):
        if hasattr(model, holder):
            raise ValueError(f"{type(model).__name__} already has an attribute {holder!r}")
        self.model = model
        self.groups = groups
        self.holder = holder
        model.add_module(  # ModuleDict keys cannot contain "."
            holder,
            nn.ModuleDict({g.name.replace(".", "_"): g.gate for g in groups}),
        )
        self._handles = [
            group.module.register_forward_pre_hook(group.pre_hook) for group in groups
        ]
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
                weighted = group.gate.open_proba().sum() * group.params_per_gate
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
                open_prob = group.gate.open_proba()
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

