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
        mu: float = 0.3,
        lr: float = None,
        penalty_update: str = 'dimin',
        pbf: str = 'quadratic_logarithmic',
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        *,
        dual_range: Tuple[float, float] = (1e-6, 100.),
        dual_momentum: float = 0.,
        dual_dampening: float = 0.,
        penalty_range: Tuple[float, float] = (1e-2, 100.),
    ) -> None:
        """
        A wrapper over a PyTorch`Optimizer` that works on the dual maximization tasks according to the Penalty-Barrier Method rule. Creates and updates dual variables.

        :param m: Number of constraints (determines the number of dual variables to create)
        :type m: int
        :param mu: Multiplier for penalty update. 
        :type mu: float
        :param penalty_update: Penalty update strategy; must be one of `dimin`,`dimin_dual`,`const`. Defaults to`dimin`.
        :type penalty_update: str
        :param pbf: Penalty-Barrier Function to use. Must be one of `quadratic_logarithmic`,`quadratic_reciprocal`
        :type pbf: str
        :param init_duals: Initial values for the dual variables. Defaults to dual lower bound for all.
        :type init_duals: float | Tensor
        :param init_penalties: Initial values for the penalty variables. Defaults to the penalty upper bound for all.
        :type init_penalties: float | Tensor
        :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
        :type dual_range: Tuple[float, float]
        :param momentum: Momentum/Smoothing factor for dual variables. Equivalent to SGD momentum. Set to `0` to disable.
        :type momentum: float
        :param dampening: Dampening for momentum. Equivalent to SGD dampening. Set to `0` to disable.
        :type dampening: float
        """

        # ## checks ##
        # if m is None and not isinstance(init_duals, Tensor):
        #     raise ValueError("At least one of`m`,`init_duals` must be set")
        # m = m if m is not None else len(init_duals)
        
        # if penalty_update == 'dimin':
        #     penalty_update_f = _update_penalties_dimin
        # elif penalty_update == 'dimin_dual':
        #     penalty_update_f = _update_penalties_dimin_dual
        # elif penalty_update == 'const':
        #     penalty_update_f = _update_penalties_const

        self.dual_range = dual_range
        self.penalty_range = penalty_range

        # if init_duals is None or isinstance(init_duals, (int, float)): # initialize duals if not set or set to scalar
        #     init_duals = torch.zeros(m, requires_grad=False) + (init_duals if isinstance(init_duals, (int, float)) else dual_range[0])
        
        # if init_penalties is None or isinstance(init_penalties, (int, float)): # initialize penalties if not set or set to scalar
        #     init_penalties = torch.zeros(m, requires_grad=False) + (init_penalties if isinstance(init_penalties, (int, float)) else penalty_range[1])

        # if lr is None:
        #     lr = mu

        # defaults = {
        #     "mu": mu,
        #     "lr": lr,
        #     "penalty_update": penalty_update_f,
        #     "pbf": pbf,
        #     "dual_momentum": dual_momentum,
        #     "dual_dampening": dual_dampening,
        #     "dual_momentum_buffer": torch.zeros_like(init_duals, requires_grad = False),
        # }

        # self.defaults = defaults

        # duals = Parameter(init_duals, requires_grad=False)
        # penalties = Parameter(init_penalties, requires_grad=False)

        params, defaults = self._init_constraint_group(m, mu, lr, penalty_update, pbf, init_duals, init_penalties, dual_momentum, dual_dampening, dual_range, penalty_range)

        super().__init__(params, defaults)

    @staticmethod
    def _init_constraint_group(
        m: int,
        mu: float = None,
        lr: float = None,
        penalty_update: str = None,
        pbf: str = None,
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        dual_momentum: float = None,
        dual_dampening: float = None,
        dual_range: Tuple[float, float] = None,
        penalty_range: Tuple[float, float] = None
    ):
        if init_duals is None and m is None:
            raise ValueError("At least one of`size`,`init_duals` must be set")
        
        if init_duals is None or isinstance(init_duals, (int, float)): # initialize duals if not set or set to scalar
            init_duals = torch.zeros(m, requires_grad=False) + (init_duals if isinstance(init_duals, (int, float)) else dual_range[0])
        if init_penalties is None or isinstance(init_penalties, (int, float)): # initialize penalties if not set or set to scalar
            init_penalties = torch.zeros(m, requires_grad=False) + (init_penalties if isinstance(init_penalties, (int, float)) else penalty_range[1])

        duals = Parameter(init_duals, requires_grad=False)
        penalties = Parameter(init_penalties, requires_grad=False)
        
        if penalty_update == 'dimin':
            penalty_update_f = _update_penalties_dimin
        elif penalty_update == 'dimin_dual':
            penalty_update_f = _update_penalties_dimin_dual
        elif penalty_update == 'const':
            penalty_update_f = _update_penalties_const
        else:
            penalty_update_f = None

        settings_dict = {
            "mu": mu,
            "lr": lr,
            "penalty_update": penalty_update_f,
            "pbf": pbf,
            "dual_momentum": dual_momentum,
            "dual_dampening": dual_dampening,
            "dual_momentum_buffer": torch.zeros_like(init_duals, requires_grad = False),
        }
        settings_dict = {k:v for k,v in settings_dict.items() if v is not None}

        param_group = ([duals, penalties], settings_dict)

        return param_group

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
        self,
        m: int,
        mu: float = None,
        lr: float = None,
        penalty_update: str = None,
        pbf: str = None,
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        *,
        dual_momentum: float = None,
        dual_dampening: float = None,
    ) -> None:
        """
        Allows to add a group of dual variables with separate initial values and learning rates.

        :param m: Size of group (number of dual variables to add)
        :type m: int
        :param mu: Multiplier for penalty update. 
        :type mu: float
        :param penalty_update: Penalty update strategy; must be one of `dimin`,`dimin_dual`,`const`. Defaults to`dimin`.
        :type penalty_update: str
        :param pbf: Penalty-Barrier Function to use. Can be one of `"quadratic_logarithmic", "quadratic_reciprocal"`
        :type pbf: str
        :param init_duals: Initial values for the dual variables. Defaults to the lower bound for all.
        :type init_duals: float | Tensor
        :param init_penalties: Initial values for the penalty variables. Defaults to min(10, lower_bound) for all.
        :type init_penalties: float | Tensor
        """
        
        
        params, settings_dict = self._init_constraint_group(m, mu, lr, penalty_update, pbf, init_duals, init_penalties, dual_momentum, dual_dampening, self.dual_range, self.penalty_range)
        param_group_dict = {"params": params, **settings_dict}
        self.add_param_group(param_group_dict)

    def update(self, constraints: Tensor) -> None:
        """
        Updates the dual variables and penalties based on the current constraint violations.

        :param constraints: Tensor of constraint violations.
        :type constraints: torch.Tensor
        :return: None
        :rtype: None
        """

        for i, group in enumerate(self.param_groups):
            duals, penalties, mu, lr, penalty_update, pbf, momentum, dampening, buffer = group["params"][0], group["params"][1], group["mu"], group["lr"], group["penalty_update"], group["pbf"], group['dual_momentum'], group['dual_dampening'], group['dual_momentum_buffer']
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            cdivp = group_constraints.div(penalties)
            with torch.no_grad():
                _update_duals(duals, cdivp, penalty_barrier_funcs[pbf]['d'], mu, momentum, dampening, buffer)
                clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                # penalty_update(penalties, lr, duals)
                # clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

    def forward(self, loss: Tensor, constraints: Tensor) -> Tensor:
        """
        Computes the Penalty-Barrier Lagrangian value for the given loss and constraints.

        :param loss: Loss (objective function) value.
        :type loss: torch.Tensor
        :param constraints: Tensor of constraint violations.
        :type constraints: torch.Tensor
        :return: Penalty-Barrier Lagrangian value.
        :rtype: torch.Tensor
        """

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
    # TODO: optimize e.g. c/p
    def forward_update(self, loss: Tensor, constraints: Tensor) -> Tensor:
        """
        Evaluates the Penalty-Barrier Lagrangian and updates the dual variables and penalties.

        Combines the computation of the Lagrangian and the update of dual variables and penalties
        in a single step, which can be more efficient than calling `forward` and `update` separately.

        :param loss: Loss (objective function) value.
        :type loss: torch.Tensor
        :param constraints: Tensor of constraint violations.
        :type constraints: torch.Tensor
        :return: Penalty-Barrier Lagrangian value.
        :rtype: torch.Tensor
        """

        lagrangian = torch.zeros_like(loss)
        lagrangian.add_(loss)
        for i, group in enumerate(self.param_groups):
            duals, penalties, mu, lr, _update_penalties, pbf, momentum, dampening, buffer = group["params"][0], group["params"][1], group["mu"], group["lr"], group["penalty_update"], group["pbf"], group['dual_momentum'], group['dual_dampening'], group['dual_momentum_buffer']
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            # calculate lagrangian
            cdivp = group_constraints.div(penalties)
            # update duals and penalties
            with torch.no_grad():
                _update_duals(duals, cdivp, penalty_barrier_funcs[pbf]['d'], mu, momentum, dampening, buffer)
                clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                # _update_penalties(penalties, lr, duals)
                # clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

            cdivp = group_constraints.div(penalties)
            pbf_val = penalty_barrier_funcs[pbf]['f'](cdivp)
            lagrangian.add_(duals.mul(penalties) @ pbf_val)

        return lagrangian
    

    def update_penalties(self):
        """
        Performs the penalty update according to`penalty_update`.
        """
        for group in self.param_groups:
            duals, penalties, lr, _update_penalties = group["params"][0], group["params"][1], group["lr"], group["penalty_update"]
            _update_penalties(penalties, lr, duals)
            clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

    # TODO: redo state dict to save the params (dual variables) themselves and not their IDs
    def state_dict(self) -> dict[str, Any]:
        
        state_dict = super().state_dict()
        state_dict["state"]["penalty_range"] = self.penalty_range
        state_dict["state"]["dual_range"] = self.dual_range
        # save params themselves in state_dict instead of param ID in default PyTorch
        for id_pg, pg in enumerate(state_dict['param_groups']):
            pg['params'] = [self.param_groups[id_pg]['params'][param_id] for param_id in pg['params'] ]
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.penalty_range = state_dict["state"]["penalty_range"]
        self.dual_range = state_dict["state"]["dual_range"]
        params = state_dict["param_groups"]
        self.param_groups = []
        for param in params:
            self.param_groups.append(param)



penalty_barrier_funcs = {
    'quadratic_logarithmic': {'f': quad_log, 'd': quad_log_der},
    'quadratic_reciprocal': {'f': quad_recipr, 'd': quad_recipr_der}
}

def _update_duals(duals: Tensor, cdivp: Tensor, pbf_der: Callable, mu: float, momentum: float, dampening: float, buffer: Tensor) -> None:
    pbf_der_val = pbf_der(cdivp)
    if momentum == 0 or not buffer.any():
        buffer = pbf_der_val
    else:
        buffer.mul_(momentum).add_(pbf_der_val, alpha = 1 - dampening)

    buffer.clamp_(mu, 1/mu)
    duals.mul_(buffer)
    

def _update_penalties_const(penalties: Tensor, mu: Tensor, duals: Tensor):
    pass

def _update_penalties_dimin(penalties: Tensor, mu: Tensor, duals: Tensor):
    penalties.mul_(mu)

def _update_penalties_dimin_dual(penalties: Tensor, mu: Tensor, duals: Tensor):
    penalties.mul_(mu).mul_(duals)