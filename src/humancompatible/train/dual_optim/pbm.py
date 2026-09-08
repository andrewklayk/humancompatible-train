import warnings

import torch
import torch.distributed as dist
from torch.nn import Parameter
from typing import Any, Optional, Tuple
from torch import clamp_, Tensor

from .barrier import quad_log, quad_log_der, quad_recipr, quad_recipr_der
from .base import DualOptimizer


class PBM(DualOptimizer):
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
        dual_range: Tuple[float, float] = (1e-9, 100.0),
        penalty_range: Tuple[float, float] = (0.1, 1.0),
        device=None,
        primal_update_process_length=1,  # length of the primal update process - if =1, is the original algorithm,
        gamma_annealing=True,
        penalty_annealing=True,
        epoch_length = None, # set this if gamma_annealing=True,
        rho = None, # only should be set if penalty_update == 'alm'; is equal to the penalty multiplier of the ALM; by default rho = 2.0
        process_group: Optional[dist.ProcessGroup] = None,
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
                if K0 == 1:
                    return 1.0 # constant

                k0 = 1.0 / (1.0 - K0)
                return 1.0 - 1.0 / (step_num**0.5 + k0)

            self.K_schedule = K_schedule
        else: # constant schedule - no change in gamma
            self.K_schedule = lambda step_num, K: K # constant

        params, settings = self._make_group(
            m,
            penalty_mult,
            penalty_update,
            delta,
            pbf,
            init_duals,
            init_penalties,
            dual_range,
            penalty_range,
            primal_update_process_length,
            rho=rho,
            device=device,
        )
        super().__init__(
            [{"params": params, **settings}],
            self._scalar_defaults(settings),
            process_group=process_group,
        )

    @classmethod
    def _make_group(
        cls,
        m: int,
        p_mult: float = None,
        penalty_update: str = None,
        delta: float = None,
        pbf: str = None,
        init_duals: float | Tensor = None,
        init_penalties: float | Tensor = None,
        dual_range: Tuple[float, float] = None,
        penalty_range: Tuple[float, float] = None,
        primal_update_process_length: int = 1,
        rho = None,
        device=None,
    ):
        # Checked before defaulting init_duals below, which would otherwise mask
        # the "neither m nor init_duals given" error.
        if init_duals is None and m is None:
            raise ValueError("At least one of m, init_duals must be set")

        # Duals default to the lower end of their range (they must stay strictly
        # positive), penalties to the upper end of theirs.
        if init_duals is None:
            init_duals = dual_range[0]
        if init_penalties is None or isinstance(init_penalties, (int, float)):
            init_penalties = torch.zeros(m, requires_grad=False, device=device) + (
                init_penalties
                if isinstance(init_penalties, (int, float))
                else penalty_range[1]
            )

        duals, settings = cls._base_group(
            m, init_duals, dual_range, is_ineq=False, device=device
        )
        penalties = Parameter(init_penalties, requires_grad=False)

        if penalty_update is not None and penalty_update not in penalty_updates:
            raise ValueError(f"Unknown penalty update function: {penalty_update}!")

        if penalty_update == "alm":
            if rho is None:
                warnings.warn(
                    "rho parameter is not set for the ALM penalty update. "
                    "By default, rho = 2.0. Set a custom value in the init to "
                    "hide this message.",
                    stacklevel=3,
                )
                rho = 2.0
            if penalty_range[1] <= 10.0:
                warnings.warn(
                    "penalty range for ALM penalty update should be large. Note "
                    "that the penalty is in each iteration equal to lambda * rho, "
                    "which can give large numbers in norm. We suggest setting the "
                    "upper range of the penalties to some bigger number.",
                    stacklevel=3,
                )

        settings.update(
            {
                "p_mult": p_mult,
                "penalty_update": penalty_update,
                "delta": delta,
                "pbf": pbf,
                "primal_update_process_length": primal_update_process_length,
                "rho": rho,
            }
        )
        return [duals, penalties], cls._drop_none(settings)

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
        rho: float = None,
        name: str = None,
        bound: float = None,
    ) -> None:
        """
        Adds an additional group of dual variables with separate hyperparameters and barrier functions.

        :param m: Number of constraints in this group (determines the number of dual variables to add)
        :type m: int
        :param penalty_mult: Multiplier for penalty update (K1 or K2). If None, inherits from parent. For adaptive penalty update, values close to 1 correspond to high "momentum".
        :type penalty_mult: float
        :param penalty_update: Penalty update strategy; must be one of `dimin`, `dimin_dual`, `dimin_adapt`, `const`, `aimd`, `alm`. If None, inherits from parent.
        :type penalty_update: str
        :param delta: Violation/satisfaction parameter for penalty update. If None, inherits from parent.
        :type delta: float
        :param pbf: Penalty-Barrier Function to use. Must be one of `quadratic_logarithmic`, `quadratic_reciprocal`.
        :type pbf: str
        :param init_duals: Initial values for the dual variables in this group. Defaults to dual lower bound for all.
        :type init_duals: float | Tensor
        :param init_penalties: Initial values for the penalty variables in this group. Defaults to penalty upper bound for all.
        :type init_penalties: float | Tensor
        :param momentum: Deprecated and ignored. The dual smoothing factor `gamma` is a property of the optimizer, not of a group.
        :type momentum: float
        :param primal_update_process_length: Length of the primal update process for this group. If 1 (default), uses original algorithm variant.
        :type primal_update_process_length: int
        :param rho: ALM penalty multiplier; only used when `penalty_update='alm'`.
        :type rho: float
        :param name: Name for this group, used when passing constraints as a mapping. Defaults to `group<k>`.
        :type name: str
        :param bound: Right-hand side of this group's constraints, if any. Only used by :meth:`violation`.
        :type bound: float

        .. note::
            The dual and penalty safeguarding ranges are properties of the
            optimizer and are shared by every group.
        """
        if momentum is not None:
            warnings.warn(
                "PBM.add_constraint_group(momentum=...) is ignored and will be "
                "removed; the dual smoothing factor gamma is set on the optimizer.",
                DeprecationWarning,
                stacklevel=2,
            )

        params, settings = self._make_group(
            m,
            penalty_mult,
            penalty_update,
            delta,
            pbf,
            init_duals,
            init_penalties,
            self.dual_range,
            self.penalty_range,
            primal_update_process_length,
            rho=rho,
        )
        if bound is not None:
            settings["bound"] = bound
        if name is not None:
            settings["name"] = name
        self.add_param_group({"params": params, **settings})

    # ------------------------------------------------------------------ #
    # hooks
    # ------------------------------------------------------------------ #

    def _snapshot(self, group: dict[str, Any]) -> Any:
        """The multipliers and penalties this primal step is entitled to use.

        Copied *before* the dual update: sharing a constraint estimate between the
        surrogate and the multiplier update correlates the two, which biases the
        primal gradient.
        """
        return (
            group["params"][0].detach().clone(),
            group["params"][1].detach().clone(),
        )

    def _should_update(self, group: dict[str, Any]) -> bool:
        # Enables the variant with several primal steps per dual step.
        return self.inner_iter + 1 == group["primal_update_process_length"]

    def _dual_update(self, group: dict[str, Any], c: Tensor) -> None:
        duals, penalties = group["params"][0], group["params"][1]
        gamma = self.gamma_schedule(self.epoch_counter, self.gamma0)
        _update_duals(
            duals,
            c.div(penalties),
            penalty_barrier_funcs[group["pbf"]]["d"],
            gamma,
        )

    def _post_update(self, group: dict[str, Any], c: Tensor) -> None:
        duals, penalties = group["params"][0], group["params"][1]
        pbf = group["pbf"]
        # Computed against the pre-update penalties, which _update_penalties then
        # overwrites in place.
        cdivp = c.div(penalties)
        p_mult = self.K_schedule(self.epoch_counter, group["p_mult"])
        penalty_updates[group["penalty_update"]](
            penalties,
            p_mult,
            duals,
            penalty_barrier_funcs[pbf]["d"](c),
            group["delta"],
            cdivp,
            rho=group.get("rho", 2.0),
        )
        clamp_(penalties, min=self.penalty_range[0], max=self.penalty_range[1])

    def _add_constraint_contributions(
        self, lagrangian: Tensor, group: dict[str, Any], snapshot: Any, c: Tensor
    ) -> None:
        lam, pen = snapshot
        pbf_val = penalty_barrier_funcs[group["pbf"]]["f"](c / pen)
        lagrangian.add_((lam * pen) @ pbf_val)

    def _end_of_step(self) -> None:
        self.inner_iter = (self.inner_iter + 1) % self.primal_update_process_length

        # keep track of the epoch counter only in the case of gamma annealing
        if self.gamma_annealing:
            self.epoch_iter += 1

        if self.gamma_annealing and self.epoch_iter == self.epoch_length:
            self.epoch_counter += 1 # increment the epoch
            self.epoch_iter = 0 # reset the counter

    def _extra_state(self) -> dict[str, Any]:
        return {
            "dual_range": self.dual_range,
            "penalty_range": self.penalty_range,
            # Without these, resuming a checkpoint would restart the gamma / K
            # annealing schedules from epoch 0.
            "inner_iter": self.inner_iter,
            "epoch_iter": self.epoch_iter,
            "epoch_counter": self.epoch_counter,
        }

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        self.dual_range = state["dual_range"]
        self.penalty_range = state["penalty_range"]
        # .get() so that checkpoints written before the counters were persisted
        # still load.
        self.inner_iter = state.get("inner_iter", self.inner_iter)
        self.epoch_iter = state.get("epoch_iter", self.epoch_iter)
        self.epoch_counter = state.get("epoch_counter", self.epoch_counter)


penalty_barrier_funcs = {
    "quadratic_logarithmic": {"f": quad_log, "d": quad_log_der},
    "quadratic_reciprocal": {"f": quad_recipr, "d": quad_recipr_der},
}


def _update_duals(
    duals: Tensor, cdivp: Tensor, pbf_der, gamma: float
) -> None:

    pbf_der_val = pbf_der(cdivp)
    upd = pbf_der_val.mul(duals)
    duals.mul_(gamma).add_(upd, alpha=1 - gamma)


# Every penalty update takes the same arguments so that the strategy can be
# stored in the param group as a *string* and resolved here; keeping a resolved
# callable in the group made state dicts pickle a module-level function.


def _update_penalties_const(
    penalties: Tensor,
    p_mult: Tensor = None,
    duals: Tensor = None,
    phi_der: Tensor = None,
    delta: float = None,
    cdivp: Tensor = None,
    rho: float = None,
    ):

    pass


def _update_penalties_alm(
    penalties: Tensor,
    p_mult: Tensor = None,
    duals: Tensor = None,
    phi_der: Tensor = None,
    delta: float = None,
    cdivp: Tensor = None,
    rho: float = 2.0,
    ):

    penalties.copy_(duals * rho) # the penalty update that transforms SPBM into ALM


def _update_penalties_dimin(
    penalties: Tensor,
    p_mult: Tensor,
    duals: Tensor = None,
    phi_der: Tensor = None,
    delta: float = None,
    cdivp: Tensor = None,
    rho: float = None,
):
    penalties.mul_(p_mult)


def _update_penalties_adapt(
    penalties: Tensor,
    p_mult: Tensor,
    duals: Tensor,
    phi_der: Tensor,
    delta: float,
    cdivp: Tensor = None,
    rho: float = None,
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
    rho: float = None,
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
    rho: float = None,
):
    penalties.mul_(p_mult).mul_(duals)


penalty_updates = {
    "const": _update_penalties_const,
    "dimin": _update_penalties_dimin,
    "dimin_dual": _update_penalties_dimin_dual,
    "dimin_adapt": _update_penalties_adapt,
    "aimd": _update_penalties_aimd,
    "alm": _update_penalties_alm,
}


PBM.__doc__ = (

    r"""
    A Dual Optimizer that works on the dual maximization tasks according to the Penalty-Barrier Method rule. Creates and updates dual variables.

    .. math::
        \mathcal{L}(\theta; \pmb{\lambda}, \mathbf{p}) = f(\theta) + \sum_i \lambda_i p_i \varphi \left( \frac{c_i(\theta)}{p_i} \right)

        \pmb{\lambda}_{t+1} \leftarrow \gamma \pmb{\lambda}_t + (1 - \gamma) \pmb{\lambda}_t \varphi' \left( \frac{\mathbf{c}_t}{\mathbf{p}_t} \right)

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
    :param penalty_update: Penalty update strategy; must be one of `dimin`,`dimin_dual`,`dimin_adapt`,`const`,`aimd`,`alm`. Defaults to`dimin_adapt`.
    :type penalty_update: str
    :param pbf: Penalty-Barrier Function to use. Must be one of `quadratic_logarithmic`,`quadratic_reciprocal`
    :type pbf: str
    :param init_duals: Initial values for the dual variables. Defaults to dual lower bound for all.
    :type init_duals: float | Tensor
    :param init_penalties: Initial values for the penalty variables. Defaults to the penalty upper bound for all.
    :type init_penalties: float | Tensor
    :param dual_range: Safeguarding range for dual variables; they will be`clamp`-ed to this range.
    :type dual_range: Tuple[float, float]
    :param penalty_range: Safeguarding range for penalty variables; they will be`clamp`-ed to this range.
    :type penalty_range: Tuple[float, float]
    :param primal_update_process_length: Number of primal steps per dual/penalty update. `1` (default) is the original algorithm.
    :type primal_update_process_length: int
    :param gamma_annealing: Whether to anneal `gamma` towards 1 over epochs. Requires `epoch_length`.
    :type gamma_annealing: bool
    :param penalty_annealing: Whether to anneal the penalty multiplier towards 1 over epochs. Requires `epoch_length`.
    :type penalty_annealing: bool
    :param epoch_length: `len(train_loader)`; required when either annealing option is enabled.
    :type epoch_length: int
    :param rho: ALM penalty multiplier; only used when `penalty_update='alm'`, where it reduces this method to the Augmented Lagrangian.
    :type rho: float
    :param process_group: Distributed process group for DDP. When set, constraint values are averaged across all workers via ``dist.all_reduce`` before each dual update, keeping dual variables consistent across replicas. Defaults to ``None`` (no synchronization).
    :type process_group: dist.ProcessGroup, optional

    .. note::
        Constraint values may be passed to :meth:`forward` / :meth:`update` /
        :meth:`forward_update` as a flat tensor, as one tensor per constraint
        group, or as a mapping from group name to tensor. See
        :meth:`~humancompatible.train.dual_optim.base.DualOptimizer._gather_constraints`.
    """
)
