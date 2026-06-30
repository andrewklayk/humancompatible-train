import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from typing import Any, Tuple, Callable
from torch import clamp_, Tensor
from .barrier import quad_log, quad_log_der, quad_recipr, quad_recipr_der


class PBM(Optimizer):
    def __init__(
        self,
        m: int = None,
        penalty_mult: float = 0.1,
        gamma: float = 0.1,
        delta: float = 1.0,
        penalty_update: str = "dimin_adapt",
        *,
        pbf: str = "quadratic_logarithmic",
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        dual_range: Tuple[float, float] = (0.0001, 100.0),
        penalty_range: Tuple[float, float] = (0.1, 1.0),
        device=None,
        primal_update_process_length=1,  # length of the primal update process - if =1, is the original algorithm,
        gamma_annealing=True,
        penalty_annealing=True,
        epoch_length = None # set this if gamma_annealing=True
    ) -> None:

        self.dual_range = dual_range
        self.penalty_range = penalty_range
        self.primal_update_process_length = primal_update_process_length
        self.gamma_annealing = gamma_annealing
        self.penalty_annealing = penalty_annealing
        self.gamma0 = gamma
        self.inner_iter = 0 # modulo inner loop iters
        self.epoch_iter = 0 # epoch iters (for gamma update only)
        self.epoch_length = epoch_length
        self.epoch_counter = 0

        if (gamma_annealing or penalty_annealing) and epoch_length is None:
            raise ValueError("For gamma / penalty annealing, 'epoch_length' must be set to len(train_loader)!")

        # gamma schedule -> 1
        if self.gamma_annealing: 
            def gamma_schedule(step_num, gamma0, k0=None):
                # (1 - gamma_k) decays like 1/k  ->  gamma_k -> 1, never equals 1
                # at k=0 returns gamma0; k0 sets how fast it climbs
                if k0 is None:
                    k0 = 1.0 / (1.0 - gamma0)        # makes gamma_0 == gamma0 exactly
                return 1.0 - 1.0 / (step_num**0.5 + k0)
            self.gamma_schedule = gamma_schedule

        else: # constant schedule - no change in gamma
            self.gamma_schedule = lambda step_num, gamma0: gamma0 # constant 

        # K schedule for annealing penalty changes
        if self.penalty_annealing:
            def K_schedule(step_num, K0):
                k0 = 1.0 / (1.0 - K0)
                return 1.0 - 1.0 / (step_num**0.5 + k0)
                
            self.K_schedule = K_schedule
        else: # constant schedule - no change in gamma
            self.K_schedule = lambda step_num, K: K # constant 

        params, defaults = self._init_constraint_group(
            m,
            penalty_mult,
            penalty_update,
            delta,
            pbf,
            init_duals,
            init_penalties,
            gamma,
            dual_range,
            penalty_range,
            primal_update_process_length,
            device,
        )
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
        device=None,
    ):
        if init_duals is None and m is None:
            raise ValueError("At least one of`size`,`init_duals` must be set")

        if init_duals is None or isinstance(
            init_duals, (int, float)
        ):  # initialize duals if not set or set to scalar
            init_duals = torch.zeros(m, requires_grad=False, device=device) + (
                init_duals if isinstance(init_duals, (int, float)) else dual_range[0]
            )
        if init_penalties is None or isinstance(
            init_penalties, (int, float)
        ):  # initialize penalties if not set or set to scalar
            init_penalties = torch.zeros(m, requires_grad=False, device=device) + (
                init_penalties
                if isinstance(init_penalties, (int, float))
                else penalty_range[1]
            )

        duals = Parameter(init_duals, requires_grad=False)
        penalties = Parameter(init_penalties, requires_grad=False)

        primal_update_process_length = primal_update_process_length

        if penalty_update == "dimin":
            penalty_update_f = _update_penalties_dimin
        elif penalty_update == "dimin_dual":
            penalty_update_f = _update_penalties_dimin_dual
        elif penalty_update == "dimin_adapt":
            penalty_update_f = _update_penalties_adapt
        elif penalty_update == "const":
            penalty_update_f = _update_penalties_const
        elif penalty_update == "aimd":
            penalty_update_f = _update_penalties_aimd
        elif penalty_update is None:
            penalty_update_f = None
        else:
            raise ValueError(f"Unknown penalty update function: {penalty_update}!")

        settings_dict = {
            "p_mult": p_mult,
            "penalty_update": penalty_update_f,
            "delta": delta,
            "pbf": pbf,
            "dual_momentum": dual_momentum,
            "primal_update_process_length": primal_update_process_length,
        }
        settings_dict = {k: v for k, v in settings_dict.items() if v is not None}

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
        primal_update_process_length: int = 1,
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

        params, settings_dict = self._init_constraint_group(
            m,
            penalty_mult,
            penalty_update,
            delta,
            pbf,
            init_duals,
            init_penalties,
            momentum,
            self.dual_range,
            self.penalty_range,
            primal_update_process_length,
        )
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

        _last_c_group_index = 0
        for group in self.param_groups:
            (
                duals,
                penalties,
                p_mult,
                _update_penalties,
                delta,
                pbf,
                momentum,
                primal_update_process_length,
            ) = (
                group["params"][0],
                group["params"][1],
                group["p_mult"],
                group["penalty_update"],
                group["delta"],
                group["pbf"],
                group["dual_momentum"],
                group["primal_update_process_length"],
            )
            group_constraints = constraints[_last_c_group_index : _last_c_group_index + len(duals)]
            _last_c_group_index += len(duals)

            # update the duals only if the inner loop ended
            if (
                self.inner_iter + 1 == primal_update_process_length
            ): 
                
                cdivp = group_constraints.div(penalties)

                # update gamma and K
                gamma = self.gamma_schedule(self.epoch_counter, self.gamma0)
                p_mult = self.K_schedule(self.epoch_counter, p_mult)
                    
                with torch.no_grad():
                    _update_duals(duals, cdivp, penalty_barrier_funcs[pbf]["d"], gamma)
                    clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                    _update_penalties(
                        penalties,
                        p_mult,
                        duals,
                        penalty_barrier_funcs[pbf]["d"](group_constraints),
                        delta,
                        cdivp,
                    )
                    clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

        # update the iter
        self.inner_iter = (self.inner_iter + 1) % self.primal_update_process_length
        
        # keep track of the epoch counter only in the case of gamma annealing
        if self.gamma_annealing:
            self.epoch_iter += 1

        # update all iters if gamma annealing
        if self.gamma_annealing and self.epoch_iter == self.epoch_length:
            self.epoch_counter += 1 # increment the epoch
            self.epoch_iter = 0 # reset the counter


    step = update

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
        start = 0
        for i, group in enumerate(self.param_groups):
            duals, penalties, pbf = group["params"][0], group["params"][1], group["pbf"]
            group_constraints = constraints[start : start + len(duals)]
            start += len(duals)
            # calculate lagrangian
            cdivp = group_constraints.div(penalties)
            pbf_val = penalty_barrier_funcs[pbf]["f"](cdivp)
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
            (
                duals,
                penalties,
                p_mult,
                _update_penalties,
                delta,
                pbf,
                momentum,
                primal_update_process_length,
            ) = (
                group["params"][0],
                group["params"][1],
                group["p_mult"],
                group["penalty_update"],
                group["delta"],
                group["pbf"],
                group["dual_momentum"],
                group["primal_update_process_length"],
            )
            group_constraints = constraints[
                _last_c_group_index : _last_c_group_index + len(duals)
            ]
            _last_c_group_index = _last_c_group_index + len(duals)
            # calculate lagrangian
            if (
                self.inner_iter + 1 == primal_update_process_length
            ):  # this enables a second variant of the algorithm
                # update duals and penalties
                cdivp = group_constraints.div(penalties)

                # update gamma and K is annealing
                gamma = self.gamma_schedule(self.epoch_counter, self.gamma0)
                p_mult = self.K_schedule(self.epoch_counter, p_mult)
                with torch.no_grad():
                    _update_duals(
                        duals, cdivp, penalty_barrier_funcs[pbf]["d"], gamma
                    )
                    clamp_(duals, min=self.dual_range[0], max=self.dual_range[1])
                    _update_penalties(
                        penalties,
                        p_mult,
                        duals,
                        penalty_barrier_funcs[pbf]["d"](group_constraints),
                        delta,
                        cdivp,
                    )
                    clamp_(
                        penalties, min=self.penalty_range[0], max=self.penalty_range[1]
                    )

            cdivp = group_constraints.div(penalties)
            pbf_val = penalty_barrier_funcs[pbf]["f"](cdivp)

            # change duals to 0 for them < 1e-4, but do not overwrite the actual duals to keep the momentum working
            active = duals >= 1e-5
            if active.any():
                lagrangian.add_(duals[active].mul(penalties[active]) @ pbf_val[active])

        # update the iter
        self.inner_iter = (self.inner_iter + 1) % self.primal_update_process_length
        
        if self.gamma_annealing:
            self.epoch_iter += 1

        # update all iters if gamma annealing
        if self.gamma_annealing and self.epoch_iter == self.epoch_length:
            self.epoch_counter += 1 # increment the epoch
            self.epoch_iter = 0 # reset the counter

        return lagrangian

    def update_penalties(self, constraints: Tensor) -> None:
        """
        Updates penalties according to the specified penalty update strategy for each constraint group.

        :param constraints: Tensor of constraint violations.
        :type constraints: torch.Tensor
        :return: None
        :rtype: None
        """
        _last_c_group_index = 0
        for group in self.param_groups:
            duals, penalties, p_mult, _update_penalties, delta, pbf = (
                group["params"][0],
                group["params"][1],
                group["p_mult"],
                group["penalty_update"],
                group["delta"],
                group["pbf"],
            )
            group_constraints = constraints[_last_c_group_index : _last_c_group_index + len(duals)]
            _last_c_group_index += len(duals)
            cdivp = group_constraints.div(penalties)
            _update_penalties(
                penalties,
                p_mult,
                duals,
                penalty_barrier_funcs[pbf]["d"](group_constraints),
                delta,
                cdivp,
            )
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
        for id_pg, pg in enumerate(state_dict["param_groups"]):
            pg["params"] = self.param_groups[id_pg]["params"]
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
    "quadratic_logarithmic": {"f": quad_log, "d": quad_log_der},
    "quadratic_reciprocal": {"f": quad_recipr, "d": quad_recipr_der},
}


def _update_duals(
    duals: Tensor, cdivp: Tensor, pbf_der: Callable, gamma: float
) -> None:
    pbf_der_val = pbf_der(cdivp)
    upd = pbf_der_val.mul(duals)
    duals.mul_(gamma).add_(upd, alpha=1 - gamma)


def _update_penalties_const(
    penalties: Tensor,
    p_mult: Tensor = None,
    duals: Tensor = None,
    phi_der: Tensor = None,
    delta: float = None,
    _cdivp: Tensor = None,
):
    pass


def _update_penalties_dimin(
    penalties: Tensor,
    p_mult: Tensor,
    duals: Tensor = None,
    phi_der: Tensor = None,
    delta: float = None,
    _cdivp: Tensor = None,
):
    penalties.mul_(p_mult)


def _update_penalties_adapt(
    penalties: Tensor,
    p_mult: Tensor,
    duals: Tensor,
    phi_der: Tensor,
    delta: float,
    _cdivp: Tensor = None,
):
    d_phd = torch.where(phi_der < 1.0, phi_der, delta * phi_der)
    b = (1 - p_mult) * penalties / (d_phd + 1e-8)
    penalties.mul_(p_mult).add_(b)


def _update_penalties_aimd(
    penalties: Tensor,
    p_mult: Tensor,
    duals: Tensor,
    phi_der: Tensor,
    delta: float,
    cdivp: Tensor,
):
    p_add_rate = 0.1
    p_upd_add = torch.where(cdivp <= 0.0, p_add_rate, 0.0)
    p_upd_mult = torch.where(cdivp > 0.0, p_mult, 1.0)
    penalties.add_(p_upd_add).mul_(p_upd_mult)


def _update_penalties_dimin_dual(
    penalties: Tensor,
    p_mult: Tensor,
    duals: Tensor,
    phi_der: Tensor = None,
    delta: float = None,
    cdivp: Tensor = None,
):
    penalties.mul_(p_mult).mul_(duals)


PBM.__doc__ = (

    r"""
    A Dual Optimizer that works on the dual maximization tasks according to the Penalty-Barrier Method rule. Creates and updates dual variables. Reference: https://doi.org/10.48550/arXiv.2605.18618
    
    .. note::
        
        Natively, this method only supports inequality constraints (see reference). However, it is easy to transform one into the other:

        .. math::
            g(x) = |h(x)| \leq 0

        We suggest using a small tolerance parameter on the right-hand side instead of 0.

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
)