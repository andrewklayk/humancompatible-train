import torch
import torch.distributed as dist
from torch.nn import Parameter
from typing import Any, Optional, Tuple
from torch import Tensor

from .base import DualOptimizer

# cite: Stochastic inexact augmented Lagrangian method for nonconvex expectation constrained optimization
# https://link.springer.com/content/pdf/10.1007/s10589-023-00521-z.pdf


class iALM(DualOptimizer):
    def __init__(
        self,
        m: int = None,
        beta: float = 1.0,
        sigma: float = 1.0,
        gamma: float = 1.0,
        init_duals: float | Tensor = None,
        penalty: float = 1.0,
        *,
        dual_range: Tuple[float, float] = (-100., 100.),
        momentum: float = 0.0,
        dampening: Optional[float] = None,
        is_ineq: bool = False,
        device=None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:

        self.penalty = penalty
        params, settings = self._make_group(
            m, beta, sigma, gamma, momentum, dampening, init_duals, dual_range,
            is_ineq, device,
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
        beta: float = None,
        sigma: float = None,
        gamma: float = None,
        momentum: float = None,
        dampening: float = None,
        init_duals: float | Tensor = None,
        dual_range: Tuple[float, float] = None,
        is_ineq: bool = None,
        device=None,
    ):
        if momentum is not None and (momentum < 0 or momentum > 1):
            raise ValueError(f"`momentum`must be within [0,1]; got {momentum}")

        # Default dampening to momentum (EMA) when unset and momentum > 0; else 0.
        if dampening is None:
            dampening = momentum if (momentum is not None and momentum > 0) else 0.0

        duals, settings = cls._base_group(
            m, init_duals, dual_range, is_ineq, device
        )
        settings.update(
            {
                # beta is advanced in place by the sigma schedule, so it is a
                # tensor rather than a plain float.
                "beta": Parameter(torch.tensor(beta), requires_grad=False),
                "sigma": Parameter(torch.tensor(sigma), requires_grad=False),
                "gamma": Parameter(torch.tensor(gamma), requires_grad=False),
                "momentum": momentum,
                "dampening": dampening,
                "momentum_buffer": torch.zeros_like(
                    duals.data, requires_grad=False, device=device
                ),
            }
        )
        return [duals], cls._drop_none(settings)

    def add_constraint_group(
        self,
        m: int = None,
        beta: float = 1.0,
        sigma: float = 1.0,
        gamma: float = 1.0,
        momentum: float = None,
        dampening: Optional[float] = None,
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
        :param beta: Dual variable update rate
        :type beta: float
        :param sigma: Multiplier for increasing `beta`
        :type sigma: float
        :param gamma: Penalty update parameter
        :type gamma: float
        :param momentum: Momentum for dual variable updates
        :type momentum: float
        :param dampening: Dampening for momentum
        :type dampening: float
        :param init_duals: Initial values for the new dual variables
        :type init_duals: Tensor
        :param dual_range: After each dual update, the dual variables will be clamped to this range.
        :type dual_range: Tuple[float, float]
        :param is_ineq: Whether to treat the constraints as equality or inequality. If`True`, dual variables will be relaxed on strict satisfaction and lower-bounded by `max(dual_range[0], 0)`.
        :type is_ineq: bool
        :param name: Name for this group, used when passing constraints as a mapping. Defaults to `group<k>`.
        :type name: str
        :param bound: Right-hand side of this group's constraints, if any. Only used by :meth:`violation`.
        :type bound: float
        """
        params, settings = self._make_group(
            m, beta, sigma, gamma, momentum, dampening, init_duals, dual_range,
            is_ineq, device,
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
        momentum = group.get("momentum", 0.0)
        buffer = group.get("momentum_buffer")

        if momentum > 0:
            _update_c_buffers(c, momentum, group.get("dampening", 0.0), buffer)

        _update_duals(
            group["params"][0],
            group.get("beta"),
            group.get("gamma"),
            buffer if momentum > 0 else c,
        )

    def _add_surrogate_terms(
        self, lagrangian: Tensor, group: dict[str, Any], snapshot: Any, c: Tensor
    ) -> None:
        # The quadratic term uses this group's current beta, i.e. the value
        # before _end_of_step applies the sigma schedule.
        lagrangian.add_(snapshot @ c)
        lagrangian.add_(0.5 * group.get("beta") * torch.dot(c, c))

    def _end_of_step(self) -> None:
        # Advanced once per step, after every group's surrogate term has been
        # accumulated with the pre-update beta.
        for group in self.param_groups:
            group["beta"].mul_(group["sigma"])

    def _extra_state(self) -> dict[str, Any]:
        return {"penalty": self.penalty}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        self.penalty = state["penalty"]


def _update_c_buffers(
    constraints: Tensor,
    momentum: float,
    dampening: float,
    buffer: Tensor,
) -> None:
    buffer.mul_(momentum).add_(constraints, alpha=1 - dampening)


def _update_duals(
    duals: Tensor,
    beta: float,
    gamma: float,
    buffer: Tensor,
) -> None:

    update_mult = torch.min(beta, gamma / torch.linalg.norm(buffer))
    duals.add_(buffer, alpha=update_mult)


iALM.__doc__ = (

        # \textbf{input}: \gamma \text{ (lr) }, \pmb{\lambda}_t \text{ (dual variables, created by method) }, \\
        # \mathbf{c}(\theta) \text{ (constraints) }, f(\theta) \text{ (objective) }, \rho \text{ (penalty coefficient) } \\
    r"""
    A Dual Optimizer that works on the dual maximization tasks according to the Augmented Lagrangian rule, with adaptive stepsize based on https://doi.org/10.1007/s10589-023-00521-z, Algorithm 1. Creates and updates dual variables.

    .. math::

        \pmb{\lambda}_{t+1} & \leftarrow \pmb{\lambda}_t + \min\left\{ \beta_k, \frac{\gamma_k}{\|\mathbf{c}_t(\theta_t)\|} \right\} \mathbf{c}_t(\theta_{t})

        \mathcal{L}_{t+1} & \leftarrow f_t(\theta_{t}) + \pmb{\lambda}_{t+1}^T \mathbf{c}_t(\theta_{t}) + \frac{\beta_k}{2} \| \mathbf{c}_t(\theta_{t}) \|^2_2

    After each update, every group's :math:`\beta` is multiplied by its
    :math:`\sigma`, giving the geometric penalty schedule of the inexact ALM.

    :param m: Number of constraints (determines the number of dual variables to create)
    :type m: int
    :param beta: Dual variable update rate; also the coefficient of the quadratic penalty term.
    :type beta: float
    :param sigma: Multiplier for increasing`beta`.
    :type sigma: float
    :param gamma: Penalty update parameter.
    :type gamma: float
    :param init_duals: Initial values for the new dual variables. Defaults to 0 for all.
    :type init_duals: float | Tensor
    :param penalty: Accepted for API stability and stored in the state dict, but **not used**: the quadratic term is scaled by the per-group `beta`.
    :type penalty: float
    :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
    :type dual_range: Tuple[float, float]
    :param momentum: Momentum/Smoothing factor for dual variables. Equivalent to SGD momentum. Set to `0` to disable.
    :type momentum: float
    :param dampening: Dampening for momentum. Equivalent to SGD dampening. Set to `0` to disable. Defaults to `momentum` (EMA) when unset and momentum > 0.
    :type dampening: float
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
)
