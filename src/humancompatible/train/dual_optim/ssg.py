import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple
from torch import Tensor

from .base import DualOptimizer

# cite: Oracle Complexity of Single-Loop Switching Subgradient Methods for
# Non-Smooth Weakly Convex Functional Constrained Optimization (NeurIPS 2023)
# https://arxiv.org/abs/2301.13314


class SSG(DualOptimizer):
    r"""
    Stochastic switching subgradient method: maintains no multipliers at all and
    instead switches which objective the primal step descends. Reference:
    https://arxiv.org/abs/2301.13314

    At each iteration, with :math:`\tau` the constraint tolerance,

    .. math::
        \text{minimize} \quad
        \begin{cases}
            \max_i c_i(\theta) & \text{if } \max_i c_i(\theta) > \tau \\
            f(\theta)          & \text{otherwise.}
        \end{cases}

    So :meth:`forward_update` returns the *switched* objective rather than an
    augmented Lagrangian, and the caller's training loop is otherwise unchanged.
    There is nothing to update on the dual side; :attr:`duals` stays at zero and
    is kept only so that code reading multipliers for diagnostics (a KKT
    stationarity residual, say) sees the correct value of zero for this method.

    :param m: Number of constraints.
    :type m: int
    :param constraint_tol: Switch to the constraint branch when the largest
        violation exceeds this. Defaults to `0.`
    :type constraint_tol: float
    :param constraint_scale: Multiplier applied to the constraint branch. See the
        warning below before using anything other than `1.`
    :type constraint_scale: float
    :param process_group: Distributed process group for DDP. When set, constraint
        values are averaged across workers before the switch is evaluated, so all
        replicas take the same branch. Defaults to ``None``.
    :type process_group: dist.ProcessGroup, optional

    .. warning::
        The reference algorithm uses **separate step sizes** for the objective and
        the constraint branch. A single surrogate cannot express that with one
        primal optimizer. ``constraint_scale`` scales the constraint branch, which
        reproduces two step sizes exactly for **SGD-like** primal optimizers
        (:math:`k \nabla \max_i c_i` under step size :math:`\eta` equals
        :math:`\nabla \max_i c_i` under :math:`k\eta`) but **not** for adaptive
        ones such as Adam, which normalise away the scale. With an adaptive primal
        optimizer and genuinely different branch step sizes, drive two primal
        optimizers over the same parameters instead of using this class, and
        select the branch with :meth:`switched_to_constraints`.

    .. note::
        Constraint values may be passed to :meth:`forward` / :meth:`update` /
        :meth:`forward_update` as a flat tensor, as one tensor per constraint
        group, or as a mapping from group name to tensor. See
        :meth:`~humancompatible.train.dual_optim.base.DualOptimizer._gather_constraints`.
    """

    def __init__(
        self,
        m: int = None,
        constraint_tol: float = 0.0,
        *,
        constraint_scale: float = 1.0,
        device=None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:

        self.constraint_tol = constraint_tol
        self.constraint_scale = constraint_scale
        self._switched = False

        params, settings = self._make_group(m, device)
        super().__init__(
            [{"params": params, **settings}],
            self._scalar_defaults(settings),
            process_group=process_group,
        )

    @classmethod
    def _make_group(cls, m: int = None, device=None, *, bound: float = None):
        # The duals exist only so that the group bookkeeping and the `duals`
        # property behave like every other dual optimizer; they never move.
        duals, settings = cls._base_group(
            m, None, (0.0, 0.0), is_ineq=True, device=device, bound=bound
        )
        return [duals], cls._drop_none(settings)

    def add_constraint_group(
        self,
        m: int,
        device=None,
        *,
        name: str = None,
        bound: float = None,
    ) -> None:
        """
        Adds a further group of constraints.

        Groups exist here only to name and bound subsets of the constraint vector;
        the switch is always taken on the largest violation across all groups.

        :param m: Size of group (number of constraints to add)
        :type m: int
        :param name: Name for this group, used when passing constraints as a mapping. Defaults to `group<k>`.
        :type name: str
        :param bound: Right-hand side of this group's constraints, if any. Only used by :meth:`violation`.
        :type bound: float
        """
        params, settings = self._make_group(m, device, bound=bound)
        if name is not None:
            settings["name"] = name
        self.add_param_group({"params": params, **settings})

    def switched_to_constraints(self, constraints) -> bool:
        """Whether the given constraint values would select the constraint branch.

        Useful both for logging which branch a step took and for driving two
        primal optimizers by hand when the branches need different step sizes.
        """
        flat = self._gather_constraints(constraints).detach()
        return bool(flat.max() > self.constraint_tol)

    @property
    def last_switched_to_constraints(self) -> bool:
        """Whether the most recent surrogate used the constraint branch."""
        return self._switched

    # ------------------------------------------------------------------ #
    # hooks
    # ------------------------------------------------------------------ #

    def _initial_surrogate(
        self, loss: Tensor, constraints: Tensor, constraints_for_update: Tensor
    ) -> Tensor:
        # Decide on the reduced values so every replica takes the same branch;
        # differentiate the local ones so autograd sees this rank's data. Each
        # rank therefore contributes the subgradient of its own largest
        # violation, which is what a data-parallel subgradient step should do.
        self._switched = bool(constraints_for_update.detach().max() > self.constraint_tol)
        if not self._switched:
            return loss
        max_c = constraints.max()
        if self.constraint_scale == 1.0:
            return max_c
        return self.constraint_scale * max_c

    def _dual_update(self, group: dict[str, Any], c: Tensor) -> None:
        """No multipliers to update: the method switches objectives instead."""

    def _add_surrogate_terms(
        self, lagrangian: Tensor, group: dict[str, Any], snapshot: Any, c: Tensor
    ) -> None:
        """No per-group terms: :meth:`_initial_surrogate` is the whole surrogate."""

    def _extra_state(self) -> dict[str, Any]:
        return {
            "constraint_tol": self.constraint_tol,
            "constraint_scale": self.constraint_scale,
        }

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        self.constraint_tol = state["constraint_tol"]
        self.constraint_scale = state.get("constraint_scale", self.constraint_scale)
