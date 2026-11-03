"""
Shared machinery for the dual optimizers.

Every algorithm in this module solves the same problem template

.. math::
    \\min_{\\theta} \\; \\mathbb{E}[f(\\theta, \\xi)]
    \\quad \\text{s.t.} \\quad \\mathbb{E}[c(\\theta, \\zeta)] \\le 0,

by forming a scalar surrogate whose gradient with respect to :math:`\\theta` costs
a single backward pass, and driving auxiliary variables (multipliers, and for some
methods penalties) from constraint *values* alone. What distinguishes the methods
is essentially one component: the map that updates those auxiliary variables.

:class:`DualOptimizer` factors out everything else -- constraint-group
bookkeeping, input validation, dual clamping, the three public entry points
(:meth:`~DualOptimizer.forward`, :meth:`~DualOptimizer.update`,
:meth:`~DualOptimizer.forward_update`), checkpointing, and the data-parallel
reduction -- leaving each subclass to implement two required hooks and, where
needed, four optional ones.

Subclass contract
-----------------

Required:

``_dual_update(group, c)``
    Advance this group's auxiliary variables in place, given the group's slice of
    the constraint vector. The base clamps the duals to the group's range
    immediately afterwards.
``_add_constraint_contributions(lagrangian, group, snapshot, c)``
    Add this group's constraint-dependent terms to the surrogate, in place.

Optional:

``_snapshot(group)``
    What :meth:`_add_constraint_contributions` is entitled to use. Defaults to the live
    dual tensor, i.e. the surrogate sees the *post*-update multipliers, which is
    what the augmented-Lagrangian recursions prescribe. :class:`~.pbm.PBM`
    overrides it to take a pre-update copy instead, so that the surrogate and the
    dual update do not share a random constraint estimate.
``_should_update(group)``
    Whether this step performs a dual update at all. Lets a method take several
    primal steps per dual step.
``_post_update(group, c)``
    Side effects that must happen after the duals are updated *and clamped* --
    penalty updates, or a buffer whose recursion needs the previous value.
``_add_global_terms(lagrangian, constraints)``
    Terms over the whole constraint vector rather than per group, e.g. a single
    quadratic penalty shared by all groups.
``_end_of_step()``
    Bookkeeping that must run once per step, after all groups, and only on steps
    that updated -- geometric penalty schedules, epoch counters.

The ordering inside :meth:`_walk` is deliberate and load-bearing: a buffer whose
recursion needs the *previous* constraint estimate belongs in ``_post_update``
(after the dual update), whereas one that feeds the current dual step belongs
inside ``_dual_update`` itself. Getting this wrong silently changes an algorithm.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, clamp_
from torch.nn import Parameter
from torch.optim import Optimizer


def _flat(c) -> Tensor:
    """One group's constraint values as a 1-D tensor.

    A list of tensors is concatenated rather than passed to ``torch.as_tensor``,
    which would copy the values and drop their autograd graph.
    """
    if isinstance(c, (list, tuple)):
        return torch.cat([_flat(x) for x in c])
    return torch.as_tensor(c).reshape(-1)


class DualOptimizer(Optimizer, abc.ABC):
    """Base class for the dual (multiplier / penalty) optimizers.

    A "constraint group" is a :class:`torch.optim.Optimizer` param group whose
    ``params`` entry holds this group's dual tensor (and, for penalty-barrier
    methods, its penalty tensor), with the group's hyperparameters as sibling
    keys. Groups are matched to their constraints by position in the constraint
    vector, in registration order; see :meth:`_gather_constraints` for the
    accepted input forms.

    :param param_groups: List of group dictionaries, each with a ``params`` key.
    :param defaults: Scalar hyperparameters inherited by groups added later.
    :param process_group: Distributed process group. When set, constraint values
        are averaged across workers before each dual update, so replicas keep
        identical multipliers. The surrogate still uses the *local* constraint
        tensor, so autograd flows through :math:`\\partial c / \\partial \\theta`.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        defaults: dict[str, Any],
        *,
        process_group: Optional["dist.ProcessGroup"] = None,
    ) -> None:
        self.process_group = process_group
        super().__init__(param_groups, defaults)

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _base_group(
        m: Optional[int],
        init_duals: float | Tensor | None,
        dual_range: Optional[Tuple[float, float]],
        is_ineq: bool,
        device=None,
    ) -> Tuple[Parameter, dict[str, Any]]:
        """Validate a group's inputs and build its dual tensor and settings.

        Shared by every subclass: the scalar-to-tensor promotion of
        ``init_duals``, the ``is_ineq``-dependent defaulting of ``dual_range``,
        and the derived clamp bounds.
        """
        if init_duals is None and m is None:
            raise ValueError("At least one of m, init_duals must be set")

        if not isinstance(is_ineq, bool):
            raise ValueError(
                f"Expected a Boolean value for is_ineq, got {type(is_ineq)}"
            )

        init = None if init_duals is None else torch.as_tensor(init_duals)
        m = m if m is not None else init.numel()
        if init is not None and init.ndim > 0 and init.numel() != m:
            raise ValueError(f"init_duals has {init.numel()} entries, expected m={m}")

        # Always a fresh tensor: wrapping the caller's would alias it, so e.g. a
        # sweep reusing one `init_duals` would start each run from the last run's duals.
        duals = torch.zeros(
            m,
            device=device if device is not None else getattr(init, "device", None),
            # Only a torch tensor sets the dtype; Python/NumPy scalars keep the default.
            dtype=init_duals.dtype
            if isinstance(init_duals, Tensor) and init_duals.is_floating_point()
            else None,
        )
        if init is not None:
            duals += init.to(duals)
        duals = Parameter(duals, requires_grad=False)

        # Kept as given, before the inequality clip below, for later groups to
        # inherit: a clipped lower bound would wrongly carry over to equality groups.
        raw_range = dual_range
        if dual_range is None:
            dual_range = (0, None) if is_ineq else (None, None)

        lower = dual_range[0]
        if is_ineq:
            lower = 0 if lower is None else max(lower, 0)

        settings = {
            "lower_bound": lower,
            "upper_bound": dual_range[1],
            "is_ineq": is_ineq,
            "dual_range": raw_range,
        }
        return duals, settings

    @staticmethod
    def _drop_none(settings: dict[str, Any]) -> dict[str, Any]:
        """Drop unset entries so groups added later inherit them from defaults."""
        return {k: v for k, v in settings.items() if v is not None}

    @staticmethod
    def _scalar_defaults(settings: dict[str, Any]) -> dict[str, Any]:
        """The subset of a group's settings that later groups may inherit.

        Per-group *state* must never be inherited: ``Optimizer.add_param_group``
        fills missing keys from ``self.defaults`` by reference, so leaving a
        tensor buffer in the defaults would make two groups share one buffer.
        Names are per-group by definition, and so are the clamp bounds, which
        depend on the group's own ``is_ineq``; later groups re-derive them from
        the inherited ``dual_range`` instead.
        """
        return {
            k: v
            for k, v in settings.items()
            if k not in ("name", "lower_bound", "upper_bound")
            and not isinstance(v, torch.Tensor)
        }

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Register a group, giving it a default name if it has none."""
        param_group.setdefault("name", f"group{len(self.param_groups)}")
        super().add_param_group(param_group)

    # ------------------------------------------------------------------ #
    # constraint plumbing
    # ------------------------------------------------------------------ #

    def _sizes(self) -> list[int]:
        """Number of constraints per group, in registration order."""
        return [len(group["params"][0]) for group in self.param_groups]

    @property
    def m(self) -> int:
        """Total number of constraints across all groups."""
        return sum(self._sizes())

    @property
    def names(self) -> list[str]:
        """Constraint-group names, in registration order."""
        return [group["name"] for group in self.param_groups]

    @property
    def _device(self) -> torch.device:
        """Where groups added later put their tensors: alongside the first group's."""
        return self.param_groups[0]["params"][0].device

    def _gather_constraints(self, constraints) -> Tensor:
        """Normalize and validate the caller's constraint values.

        Three input forms are accepted:

        * a **flat tensor** with ``m`` entries (any shape, e.g. ``(m, 1)``; it is
          flattened), whose entries are matched to groups by position in
          registration order;
        * a **sequence** of per-group tensors, which additionally catches arity
          mistakes the flat form cannot see;
        * a **mapping** from group name to tensor, which removes the coupling
          between registration order and the order the caller happens to build
          its constraint values in.

        Every part is flattened with ``reshape(-1)``, a view wherever possible,
        so the returned tensor keeps the caller's autograd graph.
        """
        sizes = self._sizes()
        total = sum(sizes)

        def _check(parts, labels):
            for part, size, label in zip(parts, sizes, labels):
                if part.numel() != size:
                    raise ValueError(
                        f"constraint group {label!r} has {size} dual variable(s) "
                        f"but was given {part.numel()} constraint value(s)"
                    )

        if isinstance(constraints, Mapping):
            names = self.names
            missing = [n for n in names if n not in constraints]
            unknown = [k for k in constraints if k not in names]
            if missing or unknown:
                raise ValueError(
                    "constraint mapping must have exactly one entry per group; "
                    f"registered groups {names}, missing {missing}, "
                    f"unknown {unknown}"
                )
            parts = [_flat(constraints[n]) for n in names]
            _check(parts, names)
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)):
            if len(constraints) != len(sizes):
                raise ValueError(
                    f"expected one constraint tensor per group ({len(sizes)}), "
                    f"got {len(constraints)}"
                )
            parts = [_flat(c) for c in constraints]
            _check(parts, self.names)
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        flat = constraints.reshape(-1)
        if flat.numel() != total:
            raise ValueError(
                f"expected {total} constraint value(s) for group sizes {sizes}, "
                f"got {flat.numel()}"
            )
        return flat

    # TODO: deprecated, used only in quadratic ineq constraints
    def _penalty_constraints(self, constraints: Tensor) -> Tensor:
        """The constraint vector as a quadratic penalty term should see it.

        Returns the argument itself when no group is an inequality group, so the
        all-equality path is unchanged down to the last bit.
        """
        if not any(group.get("is_ineq") for group in self.param_groups):
            return constraints

        parts = []
        offset = 0
        for group, n in zip(self.param_groups, self._sizes()):
            chunk = constraints[offset : offset + n]
            parts.append(chunk.clamp(min=0.0) if group.get("is_ineq") else chunk)
            offset += n
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    # ------------------------------------------------------------------ #
    # the shared step
    # ------------------------------------------------------------------ #

    def _walk(self, loss: Optional[Tensor], constraints, *, update: bool):
        """Single pass over the constraint groups: either build the Lagrangian function ('surrogate') or update the duals, or both."""
        constraints = self._gather_constraints(constraints)

        # Reduced values drive the dual update so replicas stay consistent; the
        # local tensor is what enters the surrogate, so autograd still sees
        # this rank's dependence on the parameters.
        if update and self.process_group is not None:
            with torch.no_grad():
                constraints_for_update = constraints.detach().clone()
                dist.all_reduce(
                    constraints_for_update,
                    op=dist.ReduceOp.AVG,
                    group=self.process_group,
                )
        else:
            constraints_for_update = constraints

        lagrangian = None
        if loss is not None:
            base = self._initial_surrogate(loss, constraints, constraints_for_update)
            lagrangian = torch.zeros_like(base)
            lagrangian.add_(base)

        offset = 0
        for group in self.param_groups:
            n = len(group["params"][0])
            c_local = constraints[offset : offset + n]
            c_update = constraints_for_update[offset : offset + n]

            snapshot = self._snapshot(group)

            if update and self._should_update(group):
                with torch.no_grad():
                    self._dual_update(group, c_update)
                    clamp_(
                        group["params"][0],
                        min=group.get("lower_bound"),
                        max=group.get("upper_bound"),
                    )
                    self._post_update(group, c_update)

            if lagrangian is not None:
                self._add_constraint_contributions(lagrangian, group, snapshot, c_local)

            offset += n

        if lagrangian is not None:
            self._add_global_terms(lagrangian, constraints)

        if update:
            self._end_of_step()

        return lagrangian

    # ------------------------------------------------------------------ #
    # public entry points
    # ------------------------------------------------------------------ #

    def forward(self, loss: Tensor, constraints) -> Tensor:
        """Build the surrogate at the current multipliers, changing no state.

        :param loss: Objective value.
        :param constraints: Constraint values; flat tensor,
            or name-keyed mapping.
        :return: The surrogate (Lagrangian) to call ``.backward()`` on.

        .. warning::
            Call ``.backward()`` on the returned surrogate **before**
            :meth:`update`. The linear term is built from the live dual tensor,
            which autograd keeps for its backward pass, and :meth:`update`
            modifies that tensor in place -- so ``forward`` → ``update`` →
            ``backward`` raises ``RuntimeError: one of the variables needed for
            gradient computation has been modified by an inplace operation``.
            :meth:`forward_update` is not subject to this, and is the recommended
            entry point. Where the primal optimizer's ``step()`` goes relative to
            :meth:`update` makes no difference: the constraint tensor already
            holds the values from the current iterate.
        """
        return self._walk(loss, constraints, update=False)

    def update(self, constraints) -> None:
        """Update the auxiliary variables only.

        :param constraints: Constraint values, in any of the accepted forms.

        .. note::
            When paired with :meth:`forward`, this must come after the
            ``.backward()`` call -- see the warning there.
        """
        self._walk(None, constraints, update=True)

    step = update

    def forward_update(self, loss: Tensor, constraints) -> Tensor:
        """Update the auxiliary variables and build the surrogate in one pass.

        Marginally cheaper than :meth:`forward` followed by :meth:`update`, free
        of that pairing's ordering constraint, and the recommended entry point for
        the training loop::

            optimizer.zero_grad()
            dual.forward_update(loss, constraints).backward()
            optimizer.step()

        The surrogate is formed with the **post**-update multipliers, which is what
        the augmented-Lagrangian recursions prescribe
        (:math:`\\mathcal{L}_{t+1} = f_t + \\pmb{\\lambda}_{t+1}^T \\mathbf{c}_t`).
        :class:`~.pbm.PBM` is the exception and deliberately so: it overrides
        ``_snapshot`` to take a pre-update copy, so that the surrogate and the dual
        update do not share a random constraint estimate.

        :param loss: Objective value.
        :param constraints: Constraint values, in any of the accepted forms.
        :return: The surrogate (Lagrangian) to call ``.backward()`` on.
        """
        return self._walk(loss, constraints, update=True)

    @property
    def duals(self) -> Tensor:
        """
        :return: Dual variables, concatenated across groups.
        :rtype: Tensor
        """
        return torch.cat([group["params"][0] for group in self.param_groups])

    @property
    def duals_by_group(self) -> dict[str, Tensor]:
        """Each group's dual tensor by group name. These are the live tensors, not copies."""
        return {group["name"]: group["params"][0] for group in self.param_groups}

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #

    def _extra_state(self) -> dict[str, Any]:
        """Optimizer-level scalars to store alongside the param groups."""
        return {}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        """Restore whatever :meth:`_extra_state` saved."""

    def state_dict(self) -> dict[str, Any]:
        """"""
        state_dict = super().state_dict()
        state_dict["state"].update(self._extra_state())
        # Store the dual tensors themselves rather than the param IDs PyTorch
        # substitutes, so a checkpoint round-trips without a matching optimizer.
        # Note the IDs PyTorch assigns are global across groups, so they cannot
        # be used to index into a single group's param list.
        for id_pg, pg in enumerate(state_dict["param_groups"]):
            pg["params"] = list(self.param_groups[id_pg]["params"])
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """"""
        self._load_extra_state(state_dict["state"])
        self.param_groups = list(state_dict["param_groups"])

    # ------------------------------------------------------------------ #
    # subclass hooks
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def _dual_update(self, group: dict[str, Any], c: Tensor) -> None:
        """Advance this group's auxiliary variables in place."""

    @abc.abstractmethod
    def _add_constraint_contributions(
        self, lagrangian: Tensor, group: dict[str, Any], snapshot: Any, c: Tensor
    ) -> None:
        """Add this group's constraint-dependent surrogate terms, in place."""

    def _initial_surrogate(
        self, loss: Tensor, constraints: Tensor, constraints_for_update: Tensor
    ) -> Tensor:
        """The term the surrogate starts from, before per-group terms are added.

        Defaults to the objective, which is what every multiplier method wants.
        Switching methods, whose surrogate *replaces* the objective rather than
        augmenting it, override this instead of adding per-group terms.

        Both constraint tensors are supplied because a switching method needs
        each for a different purpose: any *decision* must be taken on
        ``constraints_for_update`` so that data-parallel replicas agree on it,
        while the returned tensor must be built from the local ``constraints`` so
        that autograd still flows through this rank's data. Outside a process
        group the two are the same tensor.
        """
        return loss

    def _snapshot(self, group: dict[str, Any]) -> Any:
        """What the surrogate may use; by default the live (post-update) duals."""
        return group["params"][0]

    def _should_update(self, group: dict[str, Any]) -> bool:
        """Whether this step updates this group's auxiliary variables."""
        return True

    def _post_update(self, group: dict[str, Any], c: Tensor) -> None:
        """Side effects to run after the duals are updated and clamped."""

    def _add_global_terms(self, lagrangian: Tensor, constraints: Tensor) -> None:
        """Add surrogate terms defined over the whole constraint vector, such as the flat quadratic penalty."""

    def _end_of_step(self) -> None:
        """Once-per-step bookkeeping, after all groups, on updating steps only."""
