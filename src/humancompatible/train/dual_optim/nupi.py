import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple
from torch import Tensor

from .base import DualOptimizer

# cite: On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization
# https://arxiv.org/pdf/2406.04558v1


class nuPI(DualOptimizer):
    r"""
    A Dual Optimizer that updates the dual variables with a proportional-integral
    (PI) controller on the constraint violation, which damps the oscillation and
    overshoot of plain dual gradient ascent. Creates and updates dual variables.
    Reference: https://doi.org/10.48550/arXiv.2406.04558

    With error :math:`\mathbf{c}_t` and error buffer :math:`\pmb{\xi}_t`, the first
    step applies (Lemma 2, eq. 15a)

    .. math::
        \pmb{\lambda}_{1} \leftarrow \pmb{\lambda}_0 + \kappa_i \mathbf{c}_0 + \kappa_p \pmb{\xi}_0

    and every later step the general recursion (Lemma 2, eq. 15c)

    .. math::
        \pmb{\lambda}_{t+1} & \leftarrow \pmb{\lambda}_t + \left( \kappa_i + \kappa_p (1 - \nu) \right) \mathbf{c}_t - \kappa_p (1 - \nu) \pmb{\xi}_t

        \pmb{\xi}_{t+1} & \leftarrow \nu \pmb{\xi}_t + (1 - \nu) \mathbf{c}_t

        \mathcal{L}_{t+1} & \leftarrow f_t(\theta_{t}) + \pmb{\lambda}_{t+1}^T \mathbf{c}_t(\theta_{t}) + \frac{\rho}{2} \| \mathbf{c}_t(\theta_{t}) \|^2_2

    Note that :math:`\nu = 0, \kappa_p = 0` recovers plain dual gradient ascent.

    The reference method defines a multiplier update, not an augmented surrogate,
    so ``penalty`` defaults to 0 and the quadratic term above is absent unless it is
    set explicitly. When it is set, and for groups registered with ``is_ineq=True``,
    the term acts on the violation :math:`[\mathbf{c}_t(\theta_t)]_+`.

    :param m: Number of constraints (determines the number of dual variables to create)
    :type m: int
    :param nu: Error-buffer decay of the PI controller.
    :type nu: float
    :param init_duals: Initial values for the new dual variables. Defaults to 0 for all.
    :type init_duals: float | Tensor
    :param penalty: Augmented Lagrangian penalty parameter. Defaults to`0.`
    :type penalty: float
    :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
    :type dual_range: Tuple[float, float]
    :param ki: Integral gain (the dual step size of plain gradient ascent).
    :type ki: float
    :param kp: Proportional gain.
    :type kp: float
    :param is_ineq: Whether to treat the constraints as equality or inequality. If`True`, dual variables will be decreased on strict satisfaction and lower-bounded by `max(dual_range[0], 0)`.
    :type is_ineq: bool
    :param process_group: Distributed process group for DDP. When set, constraint values are averaged across all workers via ``dist.all_reduce`` before each dual update, keeping dual variables consistent across replicas. Defaults to ``None`` (no synchronization).
    :type process_group: dist.ProcessGroup, optional

    .. note::
        Constraint values may be passed to :meth:`forward` / :meth:`update` /
        :meth:`forward_update` as a flat tensor, as one tensor per constraint
        group, or as a mapping from group name to tensor. See
        :meth:`~humancompatible.train.dual_optim.base.DualOptimizer._gather_constraints`.
    """

    def __init__(
        self,
        m: int = None,
        nu: float = 0.01,
        init_duals: float | Tensor = None,
        penalty: float = 0.,
        *,
        dual_range: Tuple[float, float] = (-100.0, 100.0),
        ki: float = 0.01,
        kp: float = 1.0,
        is_ineq: bool = False,
        device=None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:

        self.penalty = penalty
        params, settings = self._make_group(
            m, nu, ki, kp, init_duals, dual_range, is_ineq, device
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
        nu: float = None,
        ki: float = None,
        kp: float = None,
        init_duals: float | Tensor = None,
        dual_range: Tuple[float, float] = None,
        is_ineq: bool = None,
        device=None,
    ):
        duals, settings = cls._base_group(
            m, init_duals, dual_range, is_ineq, device
        )
        settings.update(
            {
                "nu": nu,
                "ki": ki,
                "kp": kp,
                "momentum_buffer": torch.zeros_like(
                    duals.data, requires_grad=False, device=device
                ),
            }
        )
        settings = cls._drop_none(settings)
        # False must survive the drop-None filter, so it is set afterwards.
        settings["_momentum_initialized"] = False
        return [duals], settings

    def add_constraint_group(
        self,
        m: int,
        nu: float = None,
        ki: float = None,
        kp: float = None,
        init_duals: Tensor = None,
        dual_range: tuple[float, float] = None,
        is_ineq: bool = False,
        device = None,
        *,
        name: str = None,
        bound: float = None,
    ) -> None:
        """
        Allows to add a group of dual variables with separate initial values and learning rates.

        :param m: Size of group (number of dual variables to add)
        :type m: int
        :param nu: Error-buffer decay of the PI controller.
        :type nu: float
        :param ki: Integral gain.
        :type ki: float
        :param kp: Proportional gain.
        :type kp: float
        :param init_duals: Initial values for the new dual variables. Defaults to the value set when creating the optimizer.
        :type init_duals: Tensor
        :param dual_range: After each dual update, the dual variables will be clamped to this range.
        :type dual_range: Tuple[float, float]
        :param is_ineq: Whether to treat the constraints as equality or inequality. If`True`, dual variables will be relaxed on strict satisfaction and lower-bounded by `max(dual_range[0], 0)`.
        :type is_ineq: bool
        :param name: Name for this group, used when passing constraints as a mapping. Defaults to `group<k>`.
        :type name: str
        :param bound: Right-hand side of this group's constraints, if any. Only used by :meth:`violation`.
        :type bound: float

        .. note::
            Parameters here will default to values set when initializing the dual optimizer.

        """
        params, settings = self._make_group(
            m, nu, ki, kp, init_duals, dual_range, is_ineq, device
        )
        if bound is not None:
            settings["bound"] = bound
        if name is not None:
            settings["name"] = name
        self.add_param_group({"params": params, **settings})

    # ------------------------------------------------------------------ #
    # hooks
    # ------------------------------------------------------------------ #

    def _dual_update(self, group: dict[str, Any], c: Tensor) -> None:
        duals = group["params"][0]
        buffer = group["momentum_buffer"]
        ki = group.get("ki", 0.0)
        kp = group.get("kp", 0.0)

        if not group.get("_momentum_initialized", False):
            # t=0: lambda_1 = lambda_0 + ki*c_0 + kp*xi_0  (paper Lemma 2, eq. 15a)
            duals.add_(c, alpha=ki).add_(buffer, alpha=kp)
            group["_momentum_initialized"] = True
        else:
            # t>=1: general recursion (paper Lemma 2, eq. 15c)
            _update_duals(duals, buffer, c, group["nu"], ki, kp)

    def _post_update(self, group: dict[str, Any], c: Tensor) -> None:
        # The controller's recursion consumes xi_t and only then advances it, so
        # the buffer update must follow the dual step (and its clamp).
        _update_c_buffers(c, group["nu"], group["momentum_buffer"])

    def _add_surrogate_terms(
        self, lagrangian: Tensor, group: dict[str, Any], snapshot: Any, c: Tensor
    ) -> None:
        lagrangian.add_(snapshot @ c)

    def _add_global_terms(self, lagrangian: Tensor, constraints: Tensor) -> None:
        if self.penalty == 0:
            return
        # Violations only for inequality groups; see _penalty_constraints.
        c = self._penalty_constraints(constraints)
        lagrangian.add_(0.5 * self.penalty * torch.dot(c, c))

    def _extra_state(self) -> dict[str, Any]:
        return {"penalty": self.penalty}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        self.penalty = state["penalty"]


def _update_c_buffers(
    constraints: Tensor,
    nu: float,
    buffer: Tensor,
) -> None:
    """Update the error buffer of the PI controller."""
    buffer.mul_(nu).add_(constraints, alpha=1 - nu)


def _update_duals(
    duals: Tensor,
    buffer: Tensor,
    constraints: Tensor,
    nu: float,
    ki: float,
    kp: float
) -> None:
    """Update duals with the PI controller recursion."""
    duals.add_( constraints, alpha=ki + kp * (1-nu) ).add_( buffer, alpha = -kp * (1-nu) )
