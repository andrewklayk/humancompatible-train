import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from typing import Any, Iterable, Tuple, Callable
from torch import clamp_, Tensor, no_grad
from torch.optim.optimizer import _use_grad_for_differentiable
from .barrier import (
    quad_log,
    quad_log_der,
    quad_recipr,
    quad_recipr_der
)




class PBM(Optimizer):
    def __init__(
        self,
        m: int = None,
        lr: float = 0.01,  # | Iterable[float] = 0.01 , TODO: individual LRs for each constraint
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        mu: float = 0.9,
        penalty_update: str = 'dimin',
        pbf: str | Callable[..., Tensor] = 'quadratic_logarithmic',
        *,
        dual_range: Tuple[float, float] = (1e-6, 100.),
        penalty_range: Tuple[float, float] = (1e-6, 100.)
    ) -> None:
        """
        A wrapper over a PyTorch`Optimizer` that works on the dual maximization tasks according to the Penalty-Barrier Method rule. Creates and updates dual variables.

        :param m: Number of constraints (determines the number of dual variables to create)
        :type m: int
        :param lr: Dual variable update rate 
        :type lr: float
        :param init_duals: Initial values for the dual variables. Defaults to 1e-6 for all.
        :type init_duals: float | Tensor
        :param init_penalties: Initial values for the penalty variables. Defaults to 1e-6 for all.
        :type init_penalties: float | Tensor
        :param mu: Multiplier for penalty update. 
        :type mu: float
        :param pbf: Penalty-Barrier Function to use. Can be one of `"quadratic_logarithmic", "quadratic_reciprocal"`
        :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
        :type dual_range: Tuple[float, float]
        """

        ## checks ##
        if m is None and not isinstance(init_duals, Tensor):
            raise ValueError("At least one of`m`,`init_duals` must be set")
        m = m if m is not None else len(init_duals)
        if isinstance(lr, Iterable) and len(lr) != m:
            raise ValueError("`lr`should be the same length as`init_duals`or`m`")
        
        if penalty_update == 'dimin':
            penalty_update_f = _update_penalties_dimin
        elif penalty_update == 'dimin_dual':
            penalty_update_f = _update_penalties_dimin_dual

        self.dual_range = dual_range
        self.penalty_range = penalty_range
        defaults = {"lr": lr, "mu": mu, "penalty_update": penalty_update_f, "pbf": pbf}
        self.defaults = defaults

        if init_duals is None: # initialize duals if not set
            init_duals = torch.zeros(m, requires_grad=False) + dual_range[0]
        
        if init_penalties is None: # initialize penalties if not set
            init_penalties = torch.zeros(m, requires_grad=False) + penalty_range[1]

        duals = Parameter(init_duals, requires_grad=False)
        penalties = Parameter(init_penalties, requires_grad=False)
        super().__init__([duals, penalties], defaults)


    @property
    def duals(self) -> Tensor:
        """
        :return: Dual variables, concatenated into a single tensor.
        :rtype: Tensor
        """
        return torch.cat([group["params"][0] for group in self.param_groups])
    
    @property
    def penalties(self) -> Tensor:
        """
        :return: Penalties, concatenated into a single tensor.
        :rtype: Tensor
        """
        return torch.cat([group["params"][1] for group in self.param_groups])

    def add_constraint_group(
        self, m: int = None, lr: float = None, init_duals: Tensor = None
    ) -> None:
        raise NotImplementedError()

    # def add_constraint_group(
    #     self, m: int = None, lr: float = None, init_duals: Tensor = None
    # ) -> None:
    #     """
    #     Allows to add a group of dual variables with separate initial values and learning rates.

    #     :param m: Size of group (number of dual variables to add)
    #     :type m: int
    #     :param lr: Dual variable update rate
    #     :type lr: float
    #     :param init_duals: Initial values for the new dual variables
    #     :type init_duals: Tensor
    #     """
    #     if init_duals is None and m is None:
    #         raise ValueError("At least one of`size`,`init_duals` must be set")
    #     if isinstance(lr, Iterable) and len(lr) != m:
    #         raise ValueError("`lr`should be the same length as`init_duals`or`m`")
    #     if init_duals is None:
    #         init_duals = Parameter(torch.zeros(m))
    #     param_group_dict = {"params": [init_duals]}
    #     if lr is not None:
    #         param_group_dict["lr"] = lr
    #     self.add_param_group(param_group_dict)

    def update(self, constraints: Tensor) -> Tensor:
        for i, group in enumerate(self.param_groups):
            duals, penalties, mu, penalty_update, pbf = group["params"][0], group["params"][1], group["mu"], group["penalty_update"], group["pbf"]
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            cdivp = group_constraints.div(penalties)
            with torch.no_grad():
                _update_duals(duals, cdivp, penalty_barrier_funcs[pbf]['d'], mu)
                clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                _update_penalties(penalties, mu, penalty_update, duals)
                clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

    def forward(self, loss: Tensor, constraints: Tensor) -> Tensor:
        lagrangian = torch.zeros_like(loss)
        lagrangian.add_(loss)
        for i, group in enumerate(self.param_groups):
            duals, penalties, pbf = group["params"][0], group["params"][1], group["pbf"]
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            # calculate lagrangian
            cdivp = group_constraints.div(penalties)
            pbf_val = penalty_barrier_funcs[pbf]['f'](cdivp)
            lagrangian.add_(duals.mul(penalties) @ pbf_val)

        return lagrangian

    # evaluate the Lagrangian and update the dual variables
    # TODO: choice of update methods, optimize e.g. c/p
    def forward_update(self, loss: Tensor, constraints: Tensor) -> Tensor:
        lagrangian = torch.zeros_like(loss)
        lagrangian.add_(loss)
        for i, group in enumerate(self.param_groups):
            duals, penalties, mu, penalty_update, pbf = group["params"][0], group["params"][1], group["mu"], group["penalty_update"], group["pbf"]
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            # calculate lagrangian
            cdivp = group_constraints.div(penalties)
            # update duals and penalties
            with torch.no_grad():
                _update_duals(duals, cdivp, penalty_barrier_funcs[pbf]['d'], mu)
                clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                _update_penalties(penalties, mu, penalty_update, duals)
                clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

            cdivp = group_constraints.div(penalties)
            pbf_val = penalty_barrier_funcs[pbf]['f'](cdivp)
            lagrangian.add_(duals.mul(penalties) @ pbf_val)

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



penalty_barrier_funcs = {
    'quadratic_logarithmic': {'f': quad_log, 'd': quad_log_der},
    'quadratic_reciprocal': {'f': quad_recipr, 'd': quad_recipr_der}
}

#TODO: add dual smoothing?
def _update_duals(duals: Tensor, cdivp: Tensor, pbf_der: Callable, mu: float) -> None:
    pbf_der_val = pbf_der(cdivp)
    mult = clamp_(pbf_der_val, mu, 1/mu)
    duals.mul_(mult)

#TODO: add penalty smoothing?
def _update_penalties(penalties: Tensor, mu: Tensor, rule: str, duals: Tensor):
    if rule == 'dimin':
        penalties.mul_(mu)
    elif rule == 'dimin_dual':
        penalties.mul_(mu).mul_(duals)

def _update_penalties_dimin(penalties: Tensor, mu: Tensor, _):
    penalties.mul_(mu)

def _update_penalties_dimin_dual(penalties: Tensor, mu: Tensor, duals: Tensor):
    penalties.mul_(mu).mul_(duals)