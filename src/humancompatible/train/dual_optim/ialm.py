import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from typing import Any, Tuple
from torch import clamp_, Tensor

# cite: Stochastic inexact augmented Lagrangian method for nonconvex expectation constrained optimization
# https://link.springer.com/content/pdf/10.1007/s10589-023-00521-z.pdf


class iALM(Optimizer):
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
        dampening: float = 0.0,
        is_ineq: bool = False,
        # ctol: float = 1e-4,
        device=None,
    ) -> None:
        """
        A wrapper over a PyTorch`Optimizer` that works on the dual maximization tasks according to the Augmented Lagrangian rule. Creates and updates dual variables.

        :param m: Number of constraints (determines the number of dual variables to create)
        :type m: int
        :param beta: Dual variable update rate.
        :type beta: float
        :param sigma: Multiplier for increasing`beta`.
        :type sigma: float
        :param gamma: Penalty update parameter.
        :type gamma: float
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
        :param ctol: Constraint tolerance; value that allows tiny violations of constraints to account for noise.
        :type ctol: float
        """

        if momentum > 0 and dampening == 0:
            dampening = momentum

        # self.dual_range = dual_range

        # self.beta = beta
        self.penalty = penalty
        # self.gamma = gamma
        # self.sigma = sigma
        # self.ctol = ctol

        duals, defaults = _init_constraint_group(
            m, beta, sigma, gamma, momentum, dampening, init_duals, dual_range, is_ineq, device
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
        m: int = None,
        beta: float = 1.0,
        sigma: float = 1.0,
        gamma: float = 1.0,
        momentum: float = None,
        dampening: float = None,
        init_duals: Tensor = None,
        dual_range: tuple[float, float] = None,
        is_ineq: bool = False,
        device = None
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
        """
        duals, settings_dict = _init_constraint_group(
            m, beta, sigma, gamma, momentum, dampening, init_duals, dual_range, is_ineq, device
        )
        param_group_dict = {"params": duals, **settings_dict}
        self.add_param_group(param_group_dict)

    def forward(self, loss: Tensor, constraints: Tensor) -> Tensor:
        """
        Calculates and returns the Augmented Lagrangian.

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
            duals, beta, group_constraints = _process_constraint_group_ialm(group, offset, constraints, update_duals=False)
            lagrangian.add_(duals @ group_constraints)
            offset += len(duals)

        # Use beta from first group for penalty term
        beta = self.param_groups[0]["beta"]
        lagrangian.add_(0.5 * beta * torch.dot(constraints, constraints))

        return lagrangian

    def update(self, constraints: Tensor) -> None:
        """
        Updates the dual variables

        :param constraints: Tensor of constraint values
        :type constraints: Tensor
        """
        offset = 0
        for group in self.param_groups:
            _process_constraint_group_ialm(group, offset, constraints, update_duals=True)
            offset += len(group["params"][0])

        # Update beta by sigma for each group
        for group in self.param_groups:
            group["beta"].mul_(group["sigma"])

    step = update

    # evaluate the Lagrangian and update the dual variables
    def forward_update(self, loss: Tensor, constraints: Tensor) -> Tensor:
        """
        Combines `forward` and `update`; slightly faster.

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
            duals, beta, group_constraints = _process_constraint_group_ialm(group, offset, constraints, update_duals=True)
            lagrangian.add_(duals @ group_constraints)
            offset += len(duals)

        # Use beta from first group for penalty term
        beta = self.param_groups[0]["beta"]
        lagrangian.add_(
            0.5 * beta * torch.dot(constraints, constraints)
        )

        # Update beta by sigma for each group
        for group in self.param_groups:
            group["beta"].mul_(group["sigma"])

        return lagrangian

    def state_dict(self) -> dict[str, Any]:

        state_dict = super().state_dict()
        state_dict["state"]["penalty"] = self.penalty
        # state_dict["state"]["dual_range"] = self.dual_range
        # save params themselves in state_dict instead of param ID in default PyTorch
        for id_pg, pg in enumerate(state_dict["param_groups"]):
            pg["params"] = [
                self.param_groups[id_pg]["params"][param_id]
                for param_id in pg["params"]
            ]
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.penalty = state_dict["state"]["penalty"]
        # self.dual_range = state_dict["state"]["dual_range"]
        params = state_dict["param_groups"]
        self.param_groups = []
        for param in params:
            self.param_groups.append(param)


def _process_constraint_group_ialm(
    group: dict[str, Any],
    offset: int,
    constraints: Tensor,
    update_duals: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Process a single constraint group: extract parameters and optionally update duals.

    :param group: The constraint group dictionary
    :param offset: Start index of this group's slice within the full constraints tensor
    :param constraints: Full constraints tensor
    :param update_duals: Whether to update dual variables
    :return: Tuple of (duals, beta, group_constraints)
    """
    duals = group["params"][0]
    n = len(duals)
    beta = group.get("beta")
    gamma = group.get("gamma")
    momentum = group.get("momentum", 0.0)
    dampening = group.get("dampening", 0.0)
    momentum_buffer = group.get("momentum_buffer")
    dual_lb = group.get("lower_bound")
    dual_ub = group.get("upper_bound")

    group_constraints = constraints[offset : offset + n] if constraints.ndim > 0 else constraints.unsqueeze(0)

    with torch.no_grad():
        if update_duals:
            if momentum > 0:
                _update_c_buffers(group_constraints, momentum, dampening, momentum_buffer)
            _update_duals(duals, beta, gamma, momentum_buffer if momentum > 0 else group_constraints)
            clamp_(duals, min=dual_lb, max=dual_ub)

    return duals, beta, group_constraints


def _init_constraint_group(
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
    ## checks ##
    if init_duals is None and m is None:
        raise ValueError("At least one of`m`,`init_duals` must be set")

    if momentum is not None and (momentum < 0 or momentum > 1):
        raise ValueError(f"`momentum`must be within [0,1]; got {momentum}")

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
        "beta": Parameter(torch.tensor(beta), requires_grad=False),
        "sigma": Parameter(torch.tensor(sigma), requires_grad=False),
        "gamma": Parameter(torch.tensor(gamma), requires_grad=False),
        "momentum": momentum,
        "dampening": dampening,
        "momentum_buffer": torch.zeros_like(
            init_duals, requires_grad=False, device=device
        ),
        "lower_bound": max(dual_range[0], 0) if is_ineq else dual_range[0],
        "upper_bound": dual_range[1],
        "is_ineq": is_ineq
    }
    settings_dict = {k: v for k, v in settings_dict.items() if v is not None}

    param_group = ([duals], settings_dict)
    return param_group

def _update_c_buffers(
    constraints: Tensor,
    momentum: float,
    dampening: float,
    buffer: Tensor,
):
    if momentum == 0:
        buffer = constraints
    else:
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

        \mathcal{L}_{t+1} & \leftarrow f_t(\theta_{t}) + \pmb{\lambda}_{t+1}^T \mathbf{c}_t(\theta_{t}) + \frac{\rho}{2} \| \mathbf{c}_t(\theta_{t}) \|^2_2

    :param m: Number of constraints (determines the number of dual variables to create)
    :type m: int
    :param beta: Dual variable update rate.
    :type beta: float
    :param sigma: Multiplier for increasing`beta`.
    :type sigma: float
    :param gamma: Penalty update parameter.
    :type gamma: float
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
    :param ctol: Constraint tolerance; allows tiny violations of constraints to account for noise.
    :type ctol: float
    """
)