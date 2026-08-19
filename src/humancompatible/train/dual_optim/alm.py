import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple
from torch import Tensor

from .base import DualOptimizer

# cite: Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints
# https://arxiv.org/pdf/2504.07607

AUGMENTATIONS = ("quadratic", "hpr")


class ALM(DualOptimizer):
    def __init__(
        self,
        m: int = None,
        lr: float = 0.01,
        init_duals: float | Tensor = None,
        penalty: float = 1.0,
        *,
        dual_range: Tuple[float, float] = (-100.0, 100.0),
        momentum: float = 0.0,
        dampening: Optional[float] = None,
        is_ineq: bool = False,
        restart: bool = False,
        augmentation: str = "quadratic",
        device=None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:

        if augmentation not in AUGMENTATIONS:
            raise ValueError(
                f"Unknown augmentation: {augmentation!r}; expected one of "
                f"{AUGMENTATIONS}"
            )
        if augmentation == "hpr" and penalty <= 0:
            raise ValueError(
                f"The 'hpr' augmentation divides by the penalty, so it requires "
                f"penalty > 0; got {penalty}"
            )

        self.penalty = penalty
        self.augmentation = augmentation
        params, settings = self._make_group(
            m, lr, momentum, dampening, init_duals, dual_range, is_ineq, restart, device
        )
        super().__init__(
            [{"params": params, **settings}],
            self._scalar_defaults(settings),
            process_group=process_group,
        )

    @classmethod
    def _make_group(
        cls,
        m: int = None,
        lr: float = None,
        momentum: float = None,
        dampening: float = None,
        init_duals: float | Tensor = None,
        dual_range: Tuple[float, float] = None,
        is_ineq: bool = None,
        restart: bool = None,
        device=None,
    ):
        if momentum is not None and (momentum < 0 or momentum > 1):
            raise ValueError(f"momentum must be within [0,1]; got {momentum}")

        # Default dampening to momentum (EMA) when unset and momentum > 0; else 0.
        if dampening is None:
            dampening = momentum if (momentum is not None and momentum > 0) else 0.0

        if not isinstance(restart, bool):
            raise ValueError(
                f"Expected a Boolean value for restart, got {type(restart)}"
            )

        duals, settings = cls._base_group(
            m, init_duals, dual_range, is_ineq, device
        )
        settings.update(
            {
                "lr": lr,
                "momentum": momentum,
                "dampening": dampening,
                "momentum_buffer": torch.zeros_like(
                    duals.data, requires_grad=False, device=device
                ),
                "restart": restart,
            }
        )
        return [duals], cls._drop_none(settings)

    def add_constraint_group(
        self,
        m: int,
        lr: float = None,
        momentum: float = None,
        dampening: Optional[float] = None,
        init_duals: Tensor = None,
        dual_range: tuple[float, float] = None,
        is_ineq: bool = False,
        restart: bool = False,
        device = None,
        *,
        name: str = None,
        bound: float = None,
    ) -> None:
        """
        Allows to add a group of dual variables with separate initial values and learning rates.

        :param m: Size of group (number of dual variables to add)
        :type m: int
        :param lr: Dual variable update rate.
        :type lr: float
        :param momentum: Momentum/Smoothing factor for dual variables. Equivalent to SGD momentum. Set to `0` to disable.
        :type momentum: float
        :param dampening: Dampening for momentum. Equivalent to SGD dampening. Set to `0` to disable.
        :type dampening: float
        :param init_duals: Initial values for the new dual variables. Defaults to the value set when creating the optimizer.
        :type init_duals: Tensor
        :param dual_range: After each dual update, the dual variables will be clamped to this range.
        :type dual_range: Tuple[float, float]
        :param is_ineq: Whether to treat the constraints as equality or inequality. If`True`, dual variables will be relaxed on strict satisfaction and lower-bounded by `max(dual_range[0], 0)`.
        :type is_ineq: bool
        :param restart: Whether to set the dual variables to zero immediately on strict satisfaction of corresponding constraints. Not recommended for stochastic constraints.
        :type restart: bool
        :param name: Name for this group, used when passing constraints as a mapping. Defaults to `group<k>`.
        :type name: str
        :param bound: Right-hand side of this group's constraints, if any. Only used by :meth:`violation`.
        :type bound: float

        .. note::
            Parameters here will default to values set when initializing the dual optimizer.

        """
        params, settings = self._make_group(
            m, lr, momentum, dampening, init_duals, dual_range, is_ineq, restart, device
        )
        if bound is not None:
            settings["bound"] = bound
        if name is not None:
            settings["name"] = name
        self.add_param_group({"params": params, **settings})

    # ------------------------------------------------------------------ #
    # hooks
    # ------------------------------------------------------------------ #

    def _ascent_direction(self, group: dict[str, Any], c: Tensor) -> Tensor:
        """This group's dual ascent direction, i.e. the surrogate's gradient in the duals.
        """
        if self.augmentation == "quadratic" or not group.get("is_ineq"):
            return c
        duals = group["params"][0]
        return (torch.clamp(duals + self.penalty * c, min=0.0) - duals) / self.penalty

    def _dual_update(self, group: dict[str, Any], c: Tensor) -> None:
        momentum = group.get("momentum", 0.0)
        buffer = group["momentum_buffer"]

        d = self._ascent_direction(group, c)

        if momentum > 0:
            _update_c_buffers(d, momentum, group.get("dampening", 0.0), buffer)

        _update_duals(
            group["params"][0],
            buffer if momentum > 0 else d,
            group["lr"],
            group.get("restart"),
            raw_constraints=c,
        )

    def _add_surrogate_terms(
        self, lagrangian: Tensor, group: dict[str, Any], snapshot: Any, c: Tensor
    ) -> None:
        # naive augmentation
        if self.augmentation == "quadratic":
            lagrangian.add_(snapshot @ c)
            return
        
        # HPR augmentation
        rho = self.penalty
        if group.get("is_ineq"):
            with torch.no_grad():
                lam_star = torch.clamp(snapshot + rho * c, min=0.0)
                step = lam_star - snapshot
            lagrangian.add_(lam_star @ c)
            lagrangian.add_(step.dot(step), alpha=-1.0 / (2 * rho))
        else:
            lagrangian.add_(snapshot @ c)
            lagrangian.add_(0.5 * rho * torch.dot(c, c))

    def _add_global_terms(self, lagrangian: Tensor, constraints: Tensor) -> None:
        # Under HPR the quadratic already sits inside each group's term.
        if self.augmentation == "hpr" or self.penalty == 0:
            return
        # Violations only for inequality groups; see _penalty_constraints.
        c = self._penalty_constraints(constraints)
        lagrangian.add_(0.5 * self.penalty * torch.dot(c, c))

    def _extra_state(self) -> dict[str, Any]:
        return {"penalty": self.penalty, "augmentation": self.augmentation}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        self.penalty = state["penalty"]
        # Checkpoints written before the augmentation option lack the key.
        self.augmentation = state.get("augmentation", "quadratic")


def _update_c_buffers(
    constraints: Tensor,
    momentum: float,
    dampening: float,
    buffer: Tensor,
) -> None:
    """Update the constraint buffer with momentum."""
    buffer.mul_(momentum).add_(constraints, alpha=1 - dampening)


def _update_duals(
    duals: Tensor,
    buffer: Tensor,
    lr: float,
    restart: bool,
    raw_constraints: Tensor = None,
) -> None:
    """Update duals using the buffered constraint gradients."""
    duals.add_(buffer, alpha=lr)
    if restart:
        # Use raw constraints (not the EMA buffer) to check satisfaction.
        check = raw_constraints if raw_constraints is not None else buffer
        duals.masked_fill_(check < 0, 0.0)



ALM.__doc__ = (

        # \textbf{input}: \gamma \text{ (lr) }, \pmb{\lambda}_t \text{ (dual variables, created by method) }, \\
        # \mathbf{c}(\theta) \text{ (constraints) }, f(\theta) \text{ (objective) }, \rho \text{ (penalty coefficient) } \\
    r"""
    A Dual Optimizer that works on the dual maximization tasks according to the Augmented Lagrangian rule. Creates and updates dual variables. Reference: https://doi.org/10.48550/arXiv.2504.07607

    .. math::

        \pmb{\lambda}_{t+1} & \leftarrow \pmb{\lambda}_t + \gamma \mathbf{c}_t(\theta_{t})

        \mathcal{L}_{t+1} & \leftarrow f_t(\theta_{t}) + \pmb{\lambda}_{t+1}^T \mathbf{c}_t(\theta_{t}) + \frac{\rho}{2} \| \mathbf{c}_t(\theta_{t}) \|^2_2

    For constraint groups registered with ``is_ineq=True`` the quadratic term acts
    on the violation, :math:`\frac{\rho}{2} \| [\mathbf{c}_t(\theta_t)]_+ \|^2_2`,
    since penalising the raw value of an inequality constraint would also penalise
    being strictly feasible. The linear term and the dual update always use the raw
    values.

    Setting ``augmentation="hpr"`` switches to the Hestenes--Powell--Rockafellar
    augmentation, in which the linear and quadratic terms are replaced by a single
    expression per group,

    .. math::

        \mathcal{L}_{t+1} \leftarrow f_t(\theta_t) + \frac{1}{2\rho} \left[
            \big\| \big[ \pmb{\lambda}_{t+1} + \rho \, \mathbf{c}_t(\theta_t) \big]_+ \big\|^2_2
            - \| \pmb{\lambda}_{t+1} \|^2_2 \right]

        \pmb{\lambda}_{t+1} \leftarrow \left( 1 - \tfrac{\gamma}{\rho} \right) \pmb{\lambda}_t
            + \tfrac{\gamma}{\rho} \big[ \pmb{\lambda}_t + \rho \, \mathbf{c}_t(\theta_t) \big]_+

    for inequality groups, the dual update again being gradient ascent on the
    surrogate. Writing :math:`\sigma = \pmb{\lambda} + \rho \mathbf{c}` for the trial
    multiplier, the whole difference from the default is where the clamp sits: the
    weight the surrogate puts on :math:`\partial \mathbf{c} / \partial \theta` is
    :math:`\max(\pmb{\lambda}, \sigma)` for ``"quadratic"`` but
    :math:`\max(0, \sigma)` for ``"hpr"``. The default therefore never lets a
    multiplier's pull on the primal step fall below :math:`\pmb{\lambda}` however
    feasible the constraint is, whereas HPR switches that pull off entirely once
    :math:`\sigma \le 0`. Three practical consequences:

    * HPR is a **no-op for equality groups**, where
      :math:`\frac{1}{2\rho}(\|\pmb{\mu} + \rho \mathbf{h}\|^2 - \|\pmb{\mu}\|^2)
      = \pmb{\mu}^T \mathbf{h} + \frac{\rho}{2}\|\mathbf{h}\|^2` identically and the
      dual update is unchanged.
    * At ``lr == penalty`` both modes perform the *same* dual update
      :math:`\pmb{\lambda}_{t+1} = [\pmb{\lambda}_t + \rho \mathbf{c}_t]_+` -- the
      default reaches it via the non-negativity clamp -- so at that setting the choice
      is purely a primal-side one. ``restart=True`` is also less necessary under HPR,
      since an inactive multiplier stops influencing the primal step while still
      nonzero.
    * The clamp inside the dual update makes it a nonlinear function of the constraint
      estimate, so with *stochastic* constraints HPR biases the multipliers upward
      (Jensen), i.e. toward feasibility. The default's dual update is linear in the
      estimate and carries no such bias.

    :param m: Number of constraints (determines the number of dual variables to create)
    :type m: int
    :param lr: Dual variable update rate.
    :type lr: float
    :param init_duals: Initial values for the new dual variables. Defaults to 0 for all.
    :type init_duals: float | Tensor
    :param penalty: Augmented Lagrangian penalty parameter. Defaults to`1.`
    :type penalty: float
    :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
    :type dual_range: Tuple[float, float]
    :param momentum: Momentum/Smoothing factor for dual variables. Equivalent to SGD momentum. Set to `0` to disable.
    :type momentum: float
    :param dampening: Dampening for momentum. Equivalent to SGD dampening. Set to `0` to disable.
    :type dampening: float
    :param is_ineq: Whether to treat the constraints as equality or inequality. If`True`, dual variables will be decreased on strict satisfaction and lower-bounded by `max(dual_range[0], 0)`.
    :type is_ineq: bool
    :param restart: Whether to set the dual variables to zero immediately on strict satisfaction of corresponding constraints. Not recommended for stochastic constraints.
    :type restart: bool
    :param ctol: Reserved for a constraint tolerance allowing tiny violations to account for noise. Accepted for API stability but **currently unused** by the dual update.
    :type ctol: float
    :param augmentation: Which augmentation to form, ``"quadratic"`` (default) or ``"hpr"``. The latter requires `penalty > 0` and, on inequality groups, replaces the linear-plus-quadratic terms by the Hestenes-Powell-Rockafellar expression above.
    :type augmentation: str
    :param process_group: Distributed process group for DDP. When set, constraint values are averaged across all workers via ``dist.all_reduce`` before each dual update, keeping dual variables consistent across replicas. Defaults to ``None`` (no synchronization).
    :type process_group: dist.ProcessGroup, optional

    .. note::
        Constraint values may be passed to :meth:`forward` / :meth:`update` /
        :meth:`forward_update` as a flat tensor, as one tensor per constraint
        group, or as a mapping from group name to tensor. See
        :meth:`~humancompatible.train.dual_optim.base.DualOptimizer._gather_constraints`.

    .. note::
        The HPR term also exists in this module as
        :func:`~humancompatible.train.dual_optim.barrier.augmented_lagrangian`, which
        :class:`~.pbm.PBM` reproduces with ``pbf="augmented_lagrangian"`` and
        per-coordinate penalties :math:`p = \lambda / \rho`. That parametrisation is not
        reused here because it divides by :math:`\lambda` and so breaks at
        :math:`\lambda = 0`.
    """
)
