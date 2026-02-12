import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from typing import Any, Iterable, Tuple
from torch import clamp_, Tensor, no_grad
from torch.optim.optimizer import _use_grad_for_differentiable


class ALM(Optimizer):
    def __init__(
        self,
        m: int = None,
        lr: float = 0.01,
        init_duals: float | Tensor = None,
        penalty: float = 1.0,
        *,
        dual_range: Tuple[float, float] = (0.0, 100.0),
        momentum: float = 0,
        dampening: float = 0
    ) -> None:
        """
        A wrapper over a PyTorch`Optimizer` that works on the dual maximization tasks according to the Augmented Lagrangian rule. Creates and updates dual variables.

        :param m: Number of constraints (determines the number of dual variables to create)
        :type m: int
        :param lr: Dual variable update rate 
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
        """

        ## checks ##
        if m is None and not isinstance(init_duals, Tensor):
            raise ValueError("At least one of`m`,`init_duals` must be set")
        m = m if m is not None else len(init_duals)
        if isinstance(lr, Iterable) and len(lr) != m:
            raise ValueError("`lr`must be the same length as`init_duals`or`m`")
        if penalty < 0:
            raise ValueError(f"`penalty`must be non-negative, got {penalty}")
        if momentum < 0 or momentum > 1:
            raise ValueError(f"`momentum`must be within [0,1]; got {momentum}")

        self.penalty = penalty
        self.dual_range = dual_range
        self.momentum = momentum

        if init_duals is None:
            init_duals = torch.zeros(m, requires_grad=False)

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "momentum_buffer": torch.zeros_like(init_duals, requires_grad = False)
        }

        self.defaults = defaults
        duals = Parameter(init_duals, requires_grad=False)
        super().__init__([duals], defaults)


    @property
    def duals(self) -> Tensor:
        """
        :return: Dual variables, concatenated into a single tensor.
        :rtype: Tensor
        """
        return torch.cat([group["params"][0] for group in self.param_groups])

    def add_constraint_group(
        self, m: int = None, lr: float = None, momentum: float = None, init_duals: Tensor = None
    ) -> None:
        """
        Allows to add a group of dual variables with separate initial values and learning rates.

        :param m: Size of group (number of dual variables to add)
        :type m: int
        :param lr: Dual variable update rate
        :type lr: float
        :param init_duals: Initial values for the new dual variables
        :type init_duals: Tensor
        """
        if init_duals is None and m is None:
            raise ValueError("At least one of`size`,`init_duals` must be set")
        if isinstance(lr, Iterable) and len(lr) != m:
            raise ValueError("`lr`should be the same length as`init_duals`or`m`")
        if init_duals is None:
            init_duals = Parameter(torch.zeros(m))
        param_group_dict = {"params": [init_duals]}
        if lr is not None:
            param_group_dict["lr"] = lr
            param_group_dict["momentum"] = momentum
            param_group_dict["momentum_buffer"] = torch.zeros_like(init_duals, requires_grad = False)

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
        for i, group in enumerate(self.param_groups):
            duals = group["params"][0]
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            lagrangian.add_(duals @ group_constraints)

        if self.penalty > 0:
            lagrangian.add_(
                0.5 * self.penalty * torch.square(torch.dot(constraints, constraints))
            )

        return lagrangian


    def update(self, constraints: Tensor) -> None:
        """"""
        """
        Updates the dual variables
        
        :param constraints: Tensor of constraint values
        :type constraints: Tensor
        """
        for i, group in enumerate(self.param_groups):
            duals, lr, momentum, dampening, momentum_buffer = group["params"][0], group["lr"], group["momentum"], group["dampening"], group["momentum_buffer"]
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            with torch.no_grad():
                _update_duals(duals, group_constraints, lr, momentum, dampening, momentum_buffer)
                clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])


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
        for i, group in enumerate(self.param_groups):
            duals, lr, momentum, dampening, momentum_buffer = group["params"][0], group["lr"], group["momentum"], group["dampening"], group["momentum_buffer"]
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            with torch.no_grad():
                _update_duals(duals, group_constraints, lr, momentum, dampening, momentum_buffer)
                clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])

            lagrangian.add_(duals @ group_constraints)

        if self.penalty > 0:
            lagrangian.add_(
                0.5 * self.penalty * torch.dot(constraints, constraints)
            )

        return lagrangian

    def state_dict(self) -> dict[str, Any]:

        packed_state = {"penalty": self.penalty, "dual_range": self.dual_range}
        state_dict = {"state": packed_state, "param_groups": self.param_groups}

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.penalty = state_dict["state"]["penalty"]
        self.dual_range = state_dict["state"]["dual_range"]
        params = state_dict["param_groups"]
        self.param_groups = []
        for param in params:
            self.param_groups.append(param)


def _update_duals(duals: Tensor, constraints: Tensor, lr: float, momentum: float, dampening: float, buffer: Tensor) -> None:
    if momentum == 0:
        buffer = constraints
    else:
        buffer.mul_(momentum).add_(constraints, alpha = 1 - dampening)
    duals.add_(buffer, alpha = lr)
