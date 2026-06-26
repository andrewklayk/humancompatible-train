import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from typing import Any, Tuple
from torch import clamp_, Tensor

# cite: On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization
# https://arxiv.org/pdf/2406.04558v1


class nuPI(Optimizer):
    r"""
    A Dual Optimizer that works on the dual maximization tasks according to the Augmented Lagrangian rule. Creates and updates dual variables. Reference: https://doi.org/10.48550/arXiv.2504.07607


    :param m: Number of constraints (determines the number of dual variables to create)
    :type m: int
    :param nu: Momentum parameter.
    :type nu: float
    :param init_duals: Initial values for the new dual variables. Defaults to 0 for all.
    :type init_duals: float | Tensor
    :param penalty: Augmented Lagrangian penalty parameter. Defaults to`1.`
    :type penalty: float
    :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
    :type dual_range: Tuple[float, float]
    :param ki: Momentum parameter.
    :type ki: float
    :param kp: Momentum parameter.
    :type kp: float
    :param is_ineq: Whether to treat the constraints as equality or inequality. If`True`, dual variables will be decreased on strict satisfaction and lower-bounded by `max(dual_range[0], 0)`.
    :type is_ineq: bool
    :param ctol: Constraint tolerance; allows tiny violations of constraints to account for noise.
    :type ctol: float
    """
    def __init__(
        self,
        m: int = None,
        nu: float = 0.01,
        init_duals: float | Tensor = None,
        penalty: float = 1.0,
        *,
        dual_range: Tuple[float, float] = (-100.0, 100.0),
        ki: float = 0.01,
        kp: float = 1.0,
        is_ineq: bool = False,
        device=None,
    ) -> None:

        # self.dual_range = dual_range
        # self.ctol = ctol

        self.penalty = penalty
        self._is_initialized = False
        duals, defaults = _init_constraint_group(
            m, nu, ki, kp, init_duals, dual_range, is_ineq, device
        )

        super().__init__(duals, defaults)

    @property
    def duals(self) -> Tensor:
        """
        :return: Dual variables, concatenated into a single tensor.
        :rtype: Tensor
        """
        return torch.cat([group["params"][0] for group in self.param_groups])

    def add_constraint_group(
        self,
        m: int,
        nu: float = None,
        ki: float = None,
        kp: float = None,
        init_duals: Tensor = None,
        dual_range: tuple[float, float] = None,
        is_ineq: bool = False,
        device = None
    ) -> None:
        """
        Allows to add a group of dual variables with separate initial values and learning rates.

        :param m: Size of group (number of dual variables to add)
        :type m: int
        :param nu: Momentum parameter.
        :type nu: float
        :param ki: Momentum parameter.
        :type ki: float
        :param kp: Momentum parameter.
        :type kp: float
        :param init_duals: Initial values for the new dual variables. Defaults to the value set when creating the optimizer.
        :type init_duals: Tensor
        :param dual_range: After each dual update, the dual variables will be clamped to this range.
        :type dual_range: Tuple[float, float]
        :param is_ineq: Whether to treat the constraints as equality or inequality. If`True`, dual variables will be relaxed on strict satisfaction and lower-bounded by `max(dual_range[0], 0)`.
        :type is_ineq: bool

        .. note::
            Parameters here will default to values set when initializing the dual optimizer.

        """
        duals, settings_dict = _init_constraint_group(
            m, nu, ki, kp, init_duals, dual_range, is_ineq, device
        )
        param_group_dict = {"params": duals, **settings_dict}
        self.add_param_group(param_group_dict)

    def _add_penalty_term(self, lagrangian: Tensor, constraints: Tensor) -> None:
        """Add penalty term to lagrangian in-place."""
        if self.penalty == 0:
            return
        elif constraints.ndim > 0:
            lagrangian.add_(
                0.5
                * self.penalty
                * torch.dot(constraints, constraints)
            )
        else:
            lagrangian.add_(
                0.5
                * self.penalty
                * torch.square(constraints)
            )


    def forward(self, loss: Tensor, constraints: Tensor) -> Tensor:
        """
        Calculates and returns the Augmented Lagrangian.

        Computes the augmented Lagrangian::

            L = loss + sum(duals_i @ constraints_i for all groups) + 0.5 * penalty * ||constraints||^2

        where `loss` is the objective value, `duals_i` are the dual variables, `constraints_i` are constraint values,
        `penalty` is the penalty parameter, and the sum is over all constraint groups.

        :param loss: Loss (objective function) value
        :type loss: Tensor
        :param constraints: Tensor of constraint values
        :type constraints: Tensor
        :return: Lagrangian
        :rtype: Tensor
        """
        lagrangian = torch.zeros_like(loss)
        lagrangian.add_(loss)

        offset = 0
        for group in self.param_groups:
            duals, group_constraints = _process_constraint_group(group, offset, constraints, update_duals=False)
            lagrangian.add_(duals @ group_constraints)
            offset += len(duals)

        self._add_penalty_term(lagrangian, constraints)
        return lagrangian


    def update(self, constraints: Tensor) -> None:
        """
        Updates the dual variables using constrained gradient ascent with optional momentum.

        For each constraint group, performs the dual variable update.

        First, update the momentum buffer (if momentum > 0)::

            if momentum > 0:
                buffer_i = momentum * buffer_i + (1 - dampening) * constraints_i
            else:
                buffer_i = constraints_i

        Then, update the dual variables with clamping::

            duals_i = clamp(duals_i + lr * buffer_i, lower_bound, upper_bound)

        where `buffer_i` is the momentum buffer, `constraints_i` are constraint values, `duals_i` are dual variables,
        and `clamp(x, lb, ub)` projects to the dual range.

        :param constraints: Tensor of constraint values
        :type constraints: Tensor
        """
        offset = 0
        for group in self.param_groups:
            _process_constraint_group(group, offset, constraints, update_duals=True)
            offset += len(group["params"][0])

    step = update

    # evaluate the Lagrangian and update the dual variables
    def forward_update(self, loss: Tensor, constraints: Tensor) -> Tensor:
        """
        Combines `forward` and `update`; slightly faster than calling both separately.

        Computes the augmented Lagrangian and updates dual variables in one pass::

            L = loss + sum(duals_i @ constraints_i for all groups) + 0.5 * penalty * ||constraints||^2

        Then updates dual variables::

            duals_i = clamp(duals_i + lr * buffer_i, lower_bound, upper_bound)

        where the momentum buffer is updated as in :meth:`update`.

        :param loss: Loss (objective function) value
        :type loss: Tensor
        :param constraints: Tensor of constraint values
        :type constraints: Tensor
        :return: Lagrangian
        :rtype: Tensor
        """
        lagrangian = torch.zeros_like(loss)
        lagrangian.add_(loss)

        offset = 0
        for group in self.param_groups:
            duals, group_constraints = _process_constraint_group(group, offset, constraints, update_duals=True)
            lagrangian.add_(duals @ group_constraints)
            offset += len(duals)

        self._add_penalty_term(lagrangian, constraints)
        return lagrangian

    def state_dict(self) -> dict[str, Any]:
        """"""
        state_dict = super().state_dict()
        state_dict["state"]["penalty"] = self.penalty
        # save params themselves in state_dict instead of param ID in default PyTorch
        for id_pg, pg in enumerate(state_dict["param_groups"]):
            pg["params"] = [
                self.param_groups[id_pg]["params"][param_id]
                for param_id in pg["params"]
            ]
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """"""
        self.penalty = state_dict["state"]["penalty"]
        # self.dual_range = state_dict["state"]["dual_range"]
        params = state_dict["param_groups"]
        self.param_groups = []
        for param in params:
            self.param_groups.append(param)


def _process_constraint_group(
    group: dict[str, Any],
    offset: int,
    constraints: Tensor,
    update_duals: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Process a single constraint group: extract duals/constraints and optionally update duals.

    :param group: The constraint group dictionary
    :param offset: Start index of this group's slice within the full constraints tensor
    :param constraints: Full constraints tensor
    :param update_duals: Whether to update dual variables
    :return: Tuple of (duals, group_constraints)
    """
    duals = group["params"][0]
    n = len(duals)
    group_constraints = constraints[offset : offset + n] if constraints.ndim > 0 else constraints.unsqueeze(0)

    nu = group["nu"]
    ki = group.get("ki", 0.0)
    kp = group.get("kp", 0.0)
    momentum_buffer = group["momentum_buffer"]
    dual_lb = group.get("lower_bound")
    dual_ub = group.get("upper_bound")

    with torch.no_grad():
        if update_duals:
            if not group.get("_momentum_initialized", False):
                # t=0: θ₁ = θ₀ + κᵢe₀ + κₚξ₀  (paper Lemma 2, eq. 15a)
                duals.add_(group_constraints, alpha=ki).add_(momentum_buffer, alpha=kp)
                group["_momentum_initialized"] = True
            else:
                # t≥1: general recursion (paper Lemma 2, eq. 15c)
                _update_duals(duals, momentum_buffer, group_constraints, nu, ki, kp)
            clamp_(duals, min=dual_lb, max=dual_ub)
            _update_c_buffers(group_constraints, nu, momentum_buffer)

    return duals, group_constraints


def _init_constraint_group(
    m: int = None,
    nu: float = None,
    ki: float = None,
    kp: float = None,
    init_duals: float | Tensor = None,
    dual_range: Tuple[float, float] = None,
    is_ineq: bool = None,
    device = None,
):
    ## checks ##
    if init_duals is None and m is None:
        raise ValueError("At least one of m, init_duals must be set")

    if not isinstance(is_ineq, bool):
        raise ValueError(f"Expected a Boolean value for is_ineq, got {is_ineq}")

    m = m if m is not None else len(init_duals)

    if init_duals is None:  # initialize duals if not set or set to scalar
        init_duals = torch.zeros(m, requires_grad=False, device=device)
    elif isinstance(init_duals, float):
        init_duals = torch.zeros(m, requires_grad=False, device=device) + init_duals

    duals = Parameter(init_duals, requires_grad=False)

    if dual_range is None and not is_ineq:
        dual_range = (None, None)
    elif dual_range is None and is_ineq:
        dual_range = (0, None)

    settings_dict = {
        "nu": nu,
        "ki": ki,
        "kp": kp,
        "momentum_buffer": torch.zeros_like(
            init_duals, requires_grad=False, device=device
        ),
        "lower_bound": max(dual_range[0], 0) if is_ineq else dual_range[0],
        "upper_bound": dual_range[1],
        "is_ineq": is_ineq,
        "_momentum_initialized": False
    }
    settings_dict = {k: v for k, v in settings_dict.items() if v is not None}

    param_group = ([duals], settings_dict)
    return param_group


def _update_c_buffers(
    constraints: Tensor,
    nu: float,
    buffer: Tensor,
) -> None:
    """Update the constraint buffer with momentum."""
    buffer.mul_(nu).add_(constraints, alpha=1 - nu)


def _update_duals(
    duals: Tensor,
    buffer: Tensor,
    constraints: Tensor,
    nu: float,
    ki: float,
    kp: float
) -> None:
    """Update duals using the buffered constraint gradients."""
    # duals.add_(buffer, alpha=lr).add_()
    duals.add_( constraints, alpha=ki + kp * (1-nu) ).add_( buffer, alpha = -kp * (1-nu) )


