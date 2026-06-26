import math
from functools import reduce

import torch
from torch.optim import Optimizer

from .direction import CuttingPlane, DirectionResult, GradientCombination, GradientDirection
from .inverse_hessian import DenseInverseHessian, LimitedMemoryInverseHessian
from .line_search import backtracking, weak_wolfe
from .point_set import Point, PointSet


class NonOpt(Optimizer):
    def __init__(
        self,
        params,
        direction: str = "cutting_plane",
        line_search: str = "weak_wolfe",
        inverse_hessian: str = "limited_memory",
        history_size: int = 20,
        *,
        stationarity_radius_initialization_factor: float = 1e-01,
        stationarity_radius_initialization_minimum: float = 1e-02,
        stationarity_radius_update_factor: float = 1e-01,
        stationarity_tolerance: float = 1e-04,
        stationarity_tolerance_factor: float = 1e+00,
        objective_similarity_tolerance: float = 1e-05,
        objective_similarity_limit: int = 10,
        iterate_norm_tolerance: float = 1e+20,
        point_set_options: dict = None,
        direction_options: dict = None,
        line_search_options: dict = None,
        inverse_hessian_options: dict = None,
    ) -> None:
        defaults = dict(
            direction=direction,
            line_search=line_search,
            inverse_hessian=inverse_hessian,
            history_size=history_size,
            stationarity_radius_initialization_factor=stationarity_radius_initialization_factor,
            stationarity_radius_initialization_minimum=stationarity_radius_initialization_minimum,
            stationarity_radius_update_factor=stationarity_radius_update_factor,
            stationarity_tolerance=stationarity_tolerance,
            stationarity_tolerance_factor=stationarity_tolerance_factor,
            objective_similarity_tolerance=objective_similarity_tolerance,
            objective_similarity_limit=objective_similarity_limit,
            iterate_norm_tolerance=iterate_norm_tolerance,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "NonOpt doesn't support per-parameter options (parameter groups)"
            )
        self._params = self.param_groups[0]["params"]

        if direction == "cutting_plane":
            self._direction = CuttingPlane(**(direction_options or {}))
        elif direction == "gradient_combination":
            self._direction = GradientCombination(**(direction_options or {}))
        elif direction == "gradient":
            self._direction = GradientDirection(**(direction_options or {}))
        else:
            raise ValueError(f"Unknown direction computation strategy: {direction}!")

        if line_search == "weak_wolfe":
            self._line_search = weak_wolfe
        elif line_search == "backtracking":
            self._line_search = backtracking
        else:
            raise ValueError(f"Unknown line search strategy: {line_search}!")
        self._line_search_options = line_search_options or {}

        if inverse_hessian == "limited_memory":
            self._inverse_hessian = LimitedMemoryInverseHessian(
                history_size=history_size, **(inverse_hessian_options or {})
            )
        elif inverse_hessian == "dense":
            self._inverse_hessian = DenseInverseHessian(
                **(inverse_hessian_options or {})
            )
        else:
            raise ValueError(
                f"Unknown inverse Hessian approximation: {inverse_hessian}!"
            )

        self._point_set = PointSet(**(point_set_options or {}))
        self._numel_cache = None

    # -- flat parameter handling (as in torch.optim.LBFGS) ----------------------

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, dim=0).detach().clone()

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, dim=0).detach().clone()

    def _set_flat_params(self, x):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.copy_(x[offset : offset + numel].view_as(p))
            offset += numel

    # -- termination strategy (port of NonOptTerminationBasic) ------------------

    def _radii_update_check(self, state, result: DirectionResult) -> bool:
        group = self.param_groups[0]
        threshold = (
            state["stationarity_radius"]
            * group["stationarity_tolerance_factor"]
            * state["stationarity_reference_current"]
        )
        return (
            result.direction_norm_inf <= threshold
            and result.combination_norm_inf <= threshold
        )

    def _check_termination(self, state, result: DirectionResult, f: float) -> None:
        group = self.param_groups[0]
        tolerance = group["stationarity_tolerance"]
        reference = state["stationarity_reference_current"]

        if (
            state["stationarity_radius"] <= tolerance
            and result.combination_norm_inf
            <= tolerance * group["stationarity_tolerance_factor"] * reference
        ):
            state["status"] = "stationary"

        if f - state["objective_reference"] >= -group[
            "objective_similarity_tolerance"
        ] * max(1.0, abs(state["objective_reference"])):
            state["objective_similarity_counter"] += 1
        else:
            state["objective_similarity_counter"] = max(
                0, state["objective_similarity_counter"] - 1
            )
        state["objective_reference"] = f

        similarity_exceeded = (
            state["objective_similarity_counter"] > group["objective_similarity_limit"]
        )
        if state["stationarity_radius"] <= tolerance and similarity_exceeded:
            state["status"] = "objective_similarity"

        if state["stationarity_radius"] > tolerance and (
            result.radii_update_triggered
            or self._radii_update_check(state, result)
            or similarity_exceeded
        ):
            state["objective_similarity_counter"] = 0
            state["update_radii"] = True

    # -- main step ---------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single NonOpt (outer) iteration: direction computation,
        termination/radii check, line search, inverse Hessian update and point
        set update.

        :param closure: A closure that reevaluates the model, calls
            ``backward()``, and returns the loss.
        :type closure: Callable
        :return: Loss at the iterate the step started from.
        :rtype: torch.Tensor
        """
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        state = self.state[self._params[0]]

        def evaluate(x):
            self._set_flat_params(x)
            with torch.enable_grad():
                loss = closure()
            return float(loss), self._gather_flat_grad()

        # lazy initialization on the first step
        if "x" not in state:
            state["x"] = self._gather_flat_params()
            loss = closure()
            state["f"] = float(loss)
            state["g"] = self._gather_flat_grad()
            gradient_norm_inf = float(state["g"].abs().max())
            state["stationarity_radius"] = max(
                group["stationarity_radius_initialization_minimum"],
                group["stationarity_radius_initialization_factor"] * gradient_norm_inf,
            )
            state["stationarity_reference"] = gradient_norm_inf
            state["objective_reference"] = state["f"]
            state["objective_similarity_counter"] = 0
            state["stepsize"] = 1.0
            state["iterate_norm_initial"] = float(state["x"].norm())
            state["n_iterations"] = 0
            state["status"] = "running"
        else:
            loss = None

        if state["status"] != "running":
            self._set_flat_params(state["x"])
            return loss

        x, f, g = state["x"], state["f"], state["g"]
        state["update_radii"] = False
        state["stationarity_reference_current"] = max(
            1.0, state["stationarity_reference"], float(g.abs().max())
        )

        # divergence check
        if float(x.norm()) >= group["iterate_norm_tolerance"] * max(
            1.0, state["iterate_norm_initial"]
        ):
            state["status"] = "diverged"
            self._set_flat_params(x)
            return loss

        # direction computation
        result = self._direction.compute(
            evaluate,
            x,
            f,
            g,
            self._point_set,
            self._inverse_hessian,
            state["stationarity_radius"],
            lambda res: self._radii_update_check(state, res),
        )

        # termination and radii update checks
        self._check_termination(state, result, f)
        if state["status"] != "running":
            self._set_flat_params(x)
            return loss
        if state["update_radii"]:
            state["stationarity_radius"] = max(
                group["stationarity_tolerance"],
                group["stationarity_radius_update_factor"]
                * state["stationarity_radius"],
            )
            state["stepsize"] = 1.0

        # line search
        search = self._line_search(
            evaluate,
            x,
            f,
            g,
            result.direction,
            state["stepsize"],
            result.decrease_reference,
            **self._line_search_options,
        )

        # inverse Hessian update (with self-correction; may be skipped)
        self._inverse_hessian.update(search.x - x, search.g - g)

        # accept the iterate and update the point set
        if search.stepsize > 0.0:
            self._point_set.add(Point(x, f, g))
        state["x"], state["f"], state["g"] = (
            search.x.detach().clone(),
            search.f,
            search.g.detach().clone(),
        )
        state["stepsize"] = search.stepsize if search.stepsize > 0.0 else state["stepsize"]
        state["n_iterations"] += 1
        self._point_set.update(state["x"], state["stationarity_radius"])

        self._set_flat_params(state["x"])
        return loss

    # -- inspection helpers -------------------------------------------------------

    @property
    def status(self) -> str:
        """
        Solver status: ``"running"`` (or not yet started), ``"stationary"``
        (stationarity established within the requested tolerance),
        ``"objective_similarity"`` (insufficient objective improvement at the
        final stationarity radius), or ``"diverged"``.

        :return: Status string.
        :rtype: str
        """
        state = self.state[self._params[0]]
        return state.get("status", "running")

    @property
    def converged(self) -> bool:
        """
        Whether the solver has terminated successfully (by stationarity or by
        objective similarity).

        :return: True if the solver has converged.
        :rtype: bool
        """
        return self.status in ("stationary", "objective_similarity")

    @property
    def stationarity_radius(self) -> float:
        """
        Current stationarity radius (``math.inf`` before the first step).

        :return: Stationarity radius.
        :rtype: float
        """
        state = self.state[self._params[0]]
        return state.get("stationarity_radius", math.inf)


NonOpt.__doc__ = r"""
    A PyTorch port of NonOpt (https://frankecurtis.github.io/NonOpt/), an
    open-source solver for unconstrained minimization of locally Lipschitz —
    possibly nonconvex and nonsmooth — objective functions, by Frank E. Curtis
    and collaborators.  Reference: Curtis & Zebiane,
    https://doi.org/10.48550/arXiv.2503.22826

    The method combines quasi-Newton (self-correcting BFGS) Hessian
    approximations with proximal-bundle (cutting-plane) or gradient-sampling
    direction computations and an (inexact) weak Wolfe line search.  At each
    iteration a convex quadratic subproblem over the convex hull of recently
    observed (sub)gradients is solved to obtain the search direction; a
    stationarity radius is shrunk adaptively until approximate stationarity is
    established.

    The optimizer follows the interface of :class:`torch.optim.LBFGS`: it
    requires a closure that re-evaluates the loss and its gradient, and it may
    call the closure several times per step (for bundle and line search
    evaluations).  Like LBFGS, it works with a single parameter group and is
    intended for *deterministic* (full-batch) objectives:

    .. code-block:: python

        optimizer = NonOpt(model.parameters())

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(model(input), target)
            loss.backward()
            return loss

        for _ in range(max_iterations):
            optimizer.step(closure)
            if optimizer.converged:
                break

    .. note::

        The objective only needs to be differentiable almost everywhere
        (``loss.backward()`` must produce *a* subgradient, which autograd
        does for the usual nonsmooth primitives such as ``abs``, ``max`` or
        ``relu``).  Deviations from the C++ reference: no objective scaling is
        applied, the quadratic subproblem is solved by an accelerated
        projected-gradient method without a trust-region constraint (inactive
        at its default value in the C++ implementation), and subgradient
        aggregation is not implemented.

    :param params: Iterable of parameters to optimize.
    :type params: iterable
    :param direction: Direction computation strategy; one of ``cutting_plane``
        (proximal bundle; default, as in NonOpt), ``gradient_combination``
        (gradient sampling), ``gradient`` (plain quasi-Newton).
    :type direction: str
    :param line_search: Line search strategy; one of ``weak_wolfe`` (default),
        ``backtracking``.
    :type line_search: str
    :param inverse_hessian: Inverse Hessian approximation; one of
        ``limited_memory`` (L-BFGS-style two-loop recursion; default),
        ``dense`` (explicit matrix, supports BFGS and DFP updates; only for
        small problems).
    :type inverse_hessian: str
    :param history_size: Number of curvature pairs kept by the limited-memory
        approximation.
    :type history_size: int
    :param stationarity_radius_initialization_factor: Factor for initializing
        the stationarity radius: the initial radius is the maximum of this value
        times the inf-norm of the initial gradient and
        `stationarity_radius_initialization_minimum`.
    :type stationarity_radius_initialization_factor: float
    :param stationarity_radius_initialization_minimum: Minimum initial value of
        the stationarity radius.
    :type stationarity_radius_initialization_minimum: float
    :param stationarity_radius_update_factor: Factor by which the stationarity
        radius is multiplied when the radii-update conditions are met.
    :type stationarity_radius_update_factor: float
    :param stationarity_tolerance: Tolerance for declaring stationarity; the
        algorithm reports convergence once the stationarity radius reaches this
        value and the minimum-norm gradient combination is small.
    :type stationarity_tolerance: float
    :param stationarity_tolerance_factor: Factor applied to the stationarity
        tolerance/radius in the termination and radii-update tests.
    :type stationarity_tolerance_factor: float
    :param objective_similarity_tolerance: If consecutive objective values agree
        to within this relative tolerance, a counter is increased; reaching
        `objective_similarity_limit` triggers a radius decrease or termination.
    :type objective_similarity_tolerance: float
    :param objective_similarity_limit: Limit for the objective similarity
        counter.
    :type objective_similarity_limit: int
    :param iterate_norm_tolerance: Divergence is declared when the iterate norm
        exceeds this value times ``max(1, ||x_0||)``.
    :type iterate_norm_tolerance: float
    :param point_set_options: Keyword arguments forwarded to
        :class:`~humancompatible.train.dual_optim.nonopt.point_set.PointSet`.
    :type point_set_options: dict
    :param direction_options: Keyword arguments forwarded to the direction
        computation strategy (see
        :mod:`~humancompatible.train.dual_optim.nonopt.direction`).
    :type direction_options: dict
    :param line_search_options: Keyword arguments forwarded to the line search
        (see :mod:`~humancompatible.train.dual_optim.nonopt.line_search`).
    :type line_search_options: dict
    :param inverse_hessian_options: Keyword arguments forwarded to the inverse
        Hessian approximation (see
        :mod:`~humancompatible.train.dual_optim.nonopt.inverse_hessian`).
    :type inverse_hessian_options: dict
    """
