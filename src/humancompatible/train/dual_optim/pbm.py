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
        penalty_mult: float = 0.1,
        gamma: float = 0.9,
        delta: float = 0.9,
        penalty_update: str = 'dimin_adapt',
        *,
        pbf: str = 'quadratic_logarithmic',
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        dual_range: Tuple[float, float] = (0.0001, 100.),
        penalty_range: Tuple[float, float] = (0.1, 1.),
        device = None,
        primal_update_process_length=1, # length of the primal update process - if =1, is the original algorithm
    ) -> None:
        """
        A wrapper over a PyTorch`Optimizer` that works on the dual maximization tasks according to the Penalty-Barrier Method rule. Creates and updates dual variables.

        :param m: Number of constraints (determines the number of dual variables to create)
        :type m: int
        :param penalty_mult: Multiplier for penalty update (K1 or K2). For K2 (adaptive penalty update), values close to 1 correspond to a high "momentum".
        :type penalty_mult: float
        :param gamma: Multiplier for dual parameter update. Values close to 1 correspond to a high "momentum".
        :type gamma: float
        :param delta: Violation/satisfaction parameter for penalty update; values > 1 make the penalties decrease faster on violated constraints and vice versa.
        :type delta: float 
        :param penalty_update: Penalty update strategy; must be one of `dimin`,`dimin_dual`,`dimin_adapt`,`const`. Defaults to`dimin_adapt`.
        :type penalty_update: str
        :param pbf: Penalty-Barrier Function to use. Must be one of `quadratic_logarithmic`,`quadratic_reciprocal`
        :type pbf: str
        :param init_duals: Initial values for the dual variables. Defaults to dual lower bound for all.
        :type init_duals: float | Tensor
        :param init_penalties: Initial values for the penalty variables. Defaults to the penalty upper bound for all.
        :type init_penalties: float | Tensor
        :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
        :type dual_range: Tuple[float, float]
        """

        self.dual_range = dual_range
        self.penalty_range = penalty_range

        params, defaults = self._init_constraint_group(m, penalty_mult, penalty_update, delta, pbf, init_duals, init_penalties, gamma, dual_range, penalty_range, primal_update_process_length, device)
        self.iter = 0
        super().__init__(params, defaults)

    @staticmethod
    def _init_constraint_group(
        m: int,
        p_mult: float = None,
        penalty_update: str = None,
        delta: float = None,
        pbf: str = None,
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        dual_momentum: float = None,
        dual_range: Tuple[float, float] = None,
        penalty_range: Tuple[float, float] = None,
        primal_update_process_length: int = 1,
        device = None
    ):
        if init_duals is None and m is None:
            raise ValueError("At least one of`size`,`init_duals` must be set")
        
        if init_duals is None or isinstance(init_duals, (int, float)): # initialize duals if not set or set to scalar
            init_duals = torch.zeros(m, requires_grad=False, device=device) + (init_duals if isinstance(init_duals, (int, float)) else dual_range[0])
        if init_penalties is None or isinstance(init_penalties, (int, float)): # initialize penalties if not set or set to scalar
            init_penalties = torch.zeros(m, requires_grad=False, device=device) + (init_penalties if isinstance(init_penalties, (int, float)) else penalty_range[1])

        duals = Parameter(init_duals, requires_grad=False)
        penalties = Parameter(init_penalties, requires_grad=False)
        
        primal_update_process_length = primal_update_process_length

        if penalty_update == 'dimin':
            penalty_update_f = _update_penalties_dimin
        elif penalty_update == 'dimin_dual':
            penalty_update_f = _update_penalties_dimin_dual
        elif penalty_update == 'dimin_adapt':
            penalty_update_f = _update_penalties_adapt
        elif penalty_update == 'const':
            penalty_update_f = _update_penalties_const
        elif penalty_update is None:
            penalty_update_f = None
        else:
            raise ValueError(f'Unknown penalty update function: {penalty_update}!')

        settings_dict = {
            "p_mult": p_mult,
            "penalty_update": penalty_update_f,
            "delta": delta,
            "pbf": pbf,
            "dual_momentum": dual_momentum,
            "primal_update_process_length": primal_update_process_length,
            "dual_momentum_buffer": torch.zeros_like(init_duals, requires_grad = False, device=device),
        }
        settings_dict = {k:v for k,v in settings_dict.items() if v is not None}

        param_group = ([duals, penalties], settings_dict)

        return param_group

    @property
    def duals(self) -> Tensor:
        """
        Returns all dual variables concatenated from all constraint groups.

        :return: Dual variables, concatenated into a single tensor.
        :rtype: Tensor
        """
        return torch.cat([group["params"][0] for group in self.param_groups])
    
    @property
    def penalties(self) -> Tensor:
        """
        Returns all penalty variables concatenated from all constraint groups.

        :return: Penalties, concatenated into a single tensor.
        :rtype: Tensor
        """
        return torch.cat([group["params"][1] for group in self.param_groups])

    def add_constraint_group(
        self,
        m: int,
        penalty_mult: float = None,
        penalty_update: str = None,
        delta: float = None,
        pbf: str = None,
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        *,
        momentum: float = None,
        primal_update_process_length: int = 1
    ) -> None:
        """
        Adds an additional group of dual variables with separate hyperparameters and barrier functions.

        :param m: Number of constraints in this group (determines the number of dual variables to add)
        :type m: int
        :param penalty_mult: Multiplier for penalty update (K1 or K2). If None, inherits from parent. For adaptive penalty update, values close to 1 correspond to high "momentum".
        :type penalty_mult: float
        :param penalty_update: Penalty update strategy; must be one of `dimin`, `dimin_dual`, `dimin_adapt`, `const`. If None, defaults to `dimin`.
        :type penalty_update: str
        :param delta: Violation/satisfaction parameter for penalty update. If None, inherits from parent.
        :type delta: float
        :param pbf: Penalty-Barrier Function to use. Must be one of `quadratic_logarithmic`, `quadratic_reciprocal`.
        :type pbf: str
        :param init_duals: Initial values for the dual variables in this group. Defaults to dual lower bound for all.
        :type init_duals: float | Tensor
        :param init_penalties: Initial values for the penalty variables in this group. Defaults to penalty upper bound for all.
        :type init_penalties: float | Tensor
        :param momentum: Multiplier for dual parameter update in this group. Values close to 1 correspond to high "momentum". If None, inherits from parent.
        :type momentum: float
        :param primal_update_process_length: Length of the primal update process for this group. If 1 (default), uses original algorithm variant.
        :type primal_update_process_length: int
        """
        
        
        params, settings_dict = self._init_constraint_group(m, penalty_mult, penalty_update, delta, pbf, init_duals, init_penalties, momentum, self.dual_range, self.penalty_range, primal_update_process_length)
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
            duals, penalties, p_mult, _update_penalties, delta, pbf, momentum, primal_update_process_length  = group["params"][0], group["params"][1], group["p_mult"], group["penalty_update"], group["delta"], group["pbf"], group['dual_momentum'], group["primal_update_process_length"]
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            cdivp = group_constraints.div(penalties)
            with torch.no_grad():
                _update_duals(duals, cdivp, penalty_barrier_funcs[pbf]['d'], momentum)
                clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                _update_penalties(penalties, p_mult, duals, penalty_barrier_funcs[pbf]['d'](group_constraints), delta)
                clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

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
        in a single step.

        :param loss: Loss (objective function) value.
        :type loss: torch.Tensor
        :param constraints: Tensor of constraint violations.
        :type constraints: torch.Tensor
        :return: Penalty-Barrier Lagrangian value.
        :rtype: torch.Tensor
        """

        lagrangian = torch.zeros_like(loss)
        lagrangian.add_(loss)
        _last_c_group_index = 0
        for i, group in enumerate(self.param_groups):
            duals, penalties, p_mult, _update_penalties, delta, pbf, momentum, primal_update_process_length  = group["params"][0], group["params"][1], group["p_mult"], group["penalty_update"], group["delta"], group["pbf"], group['dual_momentum'], group["primal_update_process_length"]
            group_constraints = constraints[_last_c_group_index : _last_c_group_index + len(duals)]
            _last_c_group_index = _last_c_group_index + len(duals)
            # calculate lagrangian
            if self.iter + 1 == primal_update_process_length: # this enables a second variant of the algorithm
                # update duals and penalties
                cdivp = group_constraints.div(penalties)
                with torch.no_grad():
                    _update_duals(duals, cdivp, penalty_barrier_funcs[pbf]['d'], momentum)
                    clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                    _update_penalties(penalties, p_mult, duals, penalty_barrier_funcs[pbf]['d'](group_constraints), delta)
                    clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

            cdivp = group_constraints.div(penalties)
            pbf_val = penalty_barrier_funcs[pbf]['f'](cdivp)
            lagrangian.add_(duals.mul(penalties) @ pbf_val)

        # update the iter
        self.iter = (self.iter + 1) % primal_update_process_length

        return lagrangian
    
    def update_penalties(self, constraints: Tensor) -> None:
        """
        Updates penalties according to the specified penalty update strategy for each constraint group.

        :param constraints: Tensor of constraint violations.
        :type constraints: torch.Tensor
        :return: None
        :rtype: None
        """
        for i, group in enumerate(self.param_groups):
            duals, penalties, p_mult, _update_penalties, pbf = group["params"][0], group["params"][1], group["p_mult"], group["penalty_update"], group['pbf']
            group_constraints = constraints[i * len(duals) : (i + 1) * len(duals)]
            _update_penalties(penalties, p_mult, duals, penalty_barrier_funcs[pbf]['d'](group_constraints))
            clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])


    def state_dict(self) -> dict[str, Any]:
        """
        Returns the state of the optimizer as a dictionary, including dual and penalty ranges and all constraint groups.

        :return: Dictionary containing optimizer state with param groups and configuration.
        :rtype: dict[str, Any]
        """
        
        state_dict = super().state_dict()
        state_dict["state"]["penalty_range"] = self.penalty_range
        state_dict["state"]["dual_range"] = self.dual_range
        # save params themselves in state_dict instead of param ID in default PyTorch
        for id_pg, pg in enumerate(state_dict['param_groups']):
            pg['params'] = self.param_groups[id_pg]['params']
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads the optimizer state from a dictionary, including ranges and all constraint groups.

        :param state_dict: Dictionary containing optimizer state (as returned by state_dict).
        :type state_dict: dict[str, Any]
        :return: None
        :rtype: None
        """
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

def _update_duals(duals: Tensor, cdivp: Tensor, pbf_der: Callable, gamma: float) -> None:
    pbf_der_val = pbf_der(cdivp)
    upd = pbf_der_val.mul(duals)
    duals.mul_(gamma).add_(upd, alpha=1-gamma)

def _update_penalties_const(penalties: Tensor, p_mult: Tensor = None, duals: Tensor = None, phi_der: Tensor = None, delta: float = None):
    pass

def _update_penalties_dimin(penalties: Tensor, p_mult: Tensor, duals: Tensor = None, phi_der: Tensor = None, delta: float = None):
    penalties.mul_(p_mult)

def _update_penalties_adapt(penalties: Tensor, p_mult: Tensor, duals: Tensor, phi_der: Tensor, delta: float):
    d_phd = torch.where(phi_der < 1., phi_der, torch.clamp(delta * phi_der, min=1.0))
    b = (1-p_mult)*penalties/(d_phd + 1e-8)
    penalties.mul_(p_mult).add_(b)

def _update_penalties_dimin_dual(penalties: Tensor, p_mult: Tensor, duals: Tensor, phi_der: Tensor = None, delta: float = None):
    penalties.mul_(p_mult).mul_(duals)


penalty_update_funcs = {
    'const': _update_penalties_const,
    'dimin': _update_penalties_dimin,
    'adapt': _update_penalties_adapt,
    'dimin_dual': _update_penalties_dimin_dual
}