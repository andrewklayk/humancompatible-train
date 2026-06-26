"""
Direction computation strategies, ports of ``NonOptDirectionComputationGradient``,
``NonOptDirectionComputationCuttingPlane`` (proximal-bundle, the NonOpt default)
and ``NonOptDirectionComputationGradientCombination`` (gradient-sampling) from
https://github.com/frankecurtis/NonOpt.

Each strategy assembles a bundle of (sub)gradients ``G`` and cutting-plane
values ``c``, solves the dual subproblem

.. math::
    \\min_{\\omega \\in \\Delta} \\tfrac{1}{2} \\omega^T (G^T W G)\\, \\omega
    - (c - f_k \\mathbf{1})^T \\omega,

and returns the search direction :math:`d = -W G \\omega` together with the
quantities the termination test and line search need.  Unlike the C++
implementation, no trust-region constraint is imposed on the subproblem (the
C++ default trust-region radius of ``1e+10 * ||g||`` makes it inactive in
practice anyway).
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor

from .inverse_hessian import InverseHessian
from .point_set import Point, PointSet
from .qp import solve_simplex_qp


@dataclass
class DirectionResult:
    """Search direction and the associated subproblem quantities."""

    direction: Tensor
    omega: Tensor
    combination: Tensor  #: convex combination of bundle gradients, ``G @ omega``
    dual_quadratic_value: float  #: ``(G w)' W (G w)``
    direction_norm_inf: float
    direction_norm2_squared: float
    combination_norm_inf: float
    combination_norm2_squared: float
    radii_update_triggered: bool = False

    @property
    def decrease_reference(self) -> float:
        """Reference value for sufficient-decrease tests:
        ``min(dual quadratic, max(||G w||^2, ||d||^2))``."""
        return min(
            self.dual_quadratic_value,
            max(self.combination_norm2_squared, self.direction_norm2_squared),
        )


def _solve_bundle_subproblem(
    gradients: list,
    cut_values: list,
    f_current: float,
    inverse_hessian: InverseHessian,
) -> DirectionResult:
    """Solves the dual subproblem for the given bundle and recovers the primal
    direction."""
    G = torch.stack(gradients, dim=1)
    WG = inverse_hessian.apply_matrix(G)
    Q = (G.t() @ WG).double()
    b = torch.tensor(
        [value - f_current for value in cut_values], dtype=torch.float64, device=G.device
    )
    omega = solve_simplex_qp(Q, b).to(G.dtype)
    direction = -(WG @ omega)
    combination = G @ omega
    return DirectionResult(
        direction=direction,
        omega=omega,
        combination=combination,
        dual_quadratic_value=float(omega.double() @ (Q @ omega.double())),
        direction_norm_inf=float(direction.abs().max()),
        direction_norm2_squared=float(direction.dot(direction)),
        combination_norm_inf=float(combination.abs().max()),
        combination_norm2_squared=float(combination.dot(combination)),
    )


class DirectionComputation:
    """
    Base class for direction computation strategies.

    :param step_acceptance_tolerance: Tolerance for the sufficient-decrease test
        that terminates the inner loop early.
    :type step_acceptance_tolerance: float
    :param downshift_constant: Cutting-plane downshift constant; the linear term
        of an added cut is the minimum of its linearization value and the current
        objective minus this constant times the squared distance to the iterate.
    :type downshift_constant: float
    :param inner_iteration_limit: Limit on inner (re-solve) iterations.
    :type inner_iteration_limit: int
    :param try_gradient_step: Whether to first try a cheap steepest-descent-like
        step before assembling the full bundle.
    :type try_gradient_step: bool
    :param gradient_stepsize: Stepsize used by the tentative gradient step.
    :type gradient_stepsize: float
    :param try_shortened_step: Whether to also evaluate a shortened step in each
        inner iteration (enriches the bundle near the iterate).
    :type try_shortened_step: bool
    :param shortened_stepsize: Stepsize factor for the shortened step.
    :type shortened_stepsize: float
    :param add_far_points: Whether to add trial points lying outside the
        stationarity radius to the bundle.
    :type add_far_points: bool
    """

    def __init__(
        self,
        step_acceptance_tolerance: float = 1e-08,
        downshift_constant: float = 1e-01,
        inner_iteration_limit: int = 2,
        try_gradient_step: bool = True,
        gradient_stepsize: float = 1e-03,
        try_shortened_step: bool = True,
        shortened_stepsize: float = 1e-03,
        add_far_points: bool = False,
    ) -> None:
        self.step_acceptance_tolerance = step_acceptance_tolerance
        self.downshift_constant = downshift_constant
        self.inner_iteration_limit = inner_iteration_limit
        self.try_gradient_step = try_gradient_step
        self.gradient_stepsize = gradient_stepsize
        self.try_shortened_step = try_shortened_step
        self.shortened_stepsize = shortened_stepsize
        self.add_far_points = add_far_points

    def compute(
        self,
        evaluate: Callable[[Tensor], Tuple[float, Tensor]],
        x: Tensor,
        f: float,
        g: Tensor,
        point_set: PointSet,
        inverse_hessian: InverseHessian,
        stationarity_radius: float,
        radii_update_check: Callable[[DirectionResult], bool],
    ) -> DirectionResult:
        """
        Computes a search direction at the current iterate.

        :param evaluate: Callable mapping a flat iterate to ``(objective, gradient)``.
        :type evaluate: Callable
        :param x: Current iterate (flat).
        :type x: torch.Tensor
        :param f: Objective value at ``x``.
        :type f: float
        :param g: (Sub)gradient at ``x``.
        :type g: torch.Tensor
        :param point_set: Point set; may be augmented with trial points.
        :type point_set: PointSet
        :param inverse_hessian: Inverse Hessian approximation.
        :type inverse_hessian: InverseHessian
        :param stationarity_radius: Current stationarity radius.
        :type stationarity_radius: float
        :param radii_update_check: Predicate implementing the termination
            strategy's radii-update test for a candidate direction.
        :type radii_update_check: Callable
        :return: Search direction and subproblem quantities.
        :rtype: DirectionResult
        """
        raise NotImplementedError

    # -- shared building blocks -------------------------------------------------

    def _cut_value(self, x: Tensor, f: float, point: Point, linearize: bool) -> float:
        """Cutting-plane linear term for a bundle point: the minimum of the
        linearization value (if used by the strategy) and the downshifted value."""
        difference = x - point.x
        downshift = f - self.downshift_constant * float(difference.dot(difference))
        if not linearize:
            return downshift
        linearization = point.f + float(point.g.dot(difference))
        return min(linearization, downshift)

    def _try_gradient_step(self, evaluate, x, f, g, inverse_hessian, radii_update_check):
        """Tentative steepest-descent-like step; returns an accepted result or None."""
        result = _solve_bundle_subproblem([g], [f], f, inverse_hessian)
        x_trial = x + self.gradient_stepsize * result.direction
        f_trial, _ = evaluate(x_trial)
        radii_flag = radii_update_check(result)
        accepted = (
            f_trial - f
            < -self.step_acceptance_tolerance * self.gradient_stepsize * result.decrease_reference
        )
        if accepted or radii_flag:
            result.radii_update_triggered = radii_flag
            return result
        return None

    def _inner_loop(
        self,
        evaluate,
        x,
        f,
        point_set,
        inverse_hessian,
        stationarity_radius,
        radii_update_check,
        gradients,
        cut_values,
        linearize,
        sample=None,
    ) -> DirectionResult:
        """Inner loop: alternates between evaluating the trial point implied by
        the current bundle and re-solving the subproblem with an enriched bundle."""
        result = _solve_bundle_subproblem(gradients, cut_values, f, inverse_hessian)
        inner_iteration = 1

        while True:
            x_trial = x + result.direction
            f_trial, g_trial = evaluate(x_trial)
            radii_flag = radii_update_check(result)
            if (
                f_trial - f
                < -self.step_acceptance_tolerance * result.decrease_reference
            ) or radii_flag:
                result.radii_update_triggered = radii_flag
                return result

            if inner_iteration > self.inner_iteration_limit:
                return result

            # add the (full-step) trial point to the point set and bundle
            if self.add_far_points or result.direction_norm_inf <= stationarity_radius:
                trial_point = Point(x_trial, f_trial, g_trial)
                point_set.add(trial_point)
                gradients.append(g_trial)
                cut_values.append(self._cut_value(x, f, trial_point, linearize))

            # add a shortened-step point near the iterate
            if self.try_shortened_step and result.direction_norm_inf > 0.0:
                shortened = (
                    self.shortened_stepsize
                    * min(stationarity_radius, result.direction_norm_inf)
                    / result.direction_norm_inf
                )
                x_short = x + shortened * result.direction
                f_short, g_short = evaluate(x_short)
                radii_flag = radii_update_check(result)
                if (
                    f_short - f
                    < -self.step_acceptance_tolerance * shortened * result.decrease_reference
                ) or radii_flag:
                    result.radii_update_triggered = radii_flag
                    return result
                short_point = Point(x_short, f_short, g_short)
                point_set.add(short_point)
                gradients.append(g_short)
                cut_values.append(self._cut_value(x, f, short_point, linearize))

            if sample is not None:
                sample(gradients, cut_values)

            result = _solve_bundle_subproblem(gradients, cut_values, f, inverse_hessian)
            inner_iteration += 1


class GradientDirection(DirectionComputation):
    """
    Plain quasi-Newton direction :math:`d = -W g` from the gradient at the
    current iterate only (port of ``NonOptDirectionComputationGradient``).
    """

    def compute(
        self,
        evaluate,
        x,
        f,
        g,
        point_set,
        inverse_hessian,
        stationarity_radius,
        radii_update_check,
    ) -> DirectionResult:
        result = _solve_bundle_subproblem([g], [f], f, inverse_hessian)
        result.radii_update_triggered = radii_update_check(result)
        return result


class CuttingPlane(DirectionComputation):
    """
    Proximal-bundle (cutting-plane) direction computation, the NonOpt default
    (port of ``NonOptDirectionComputationCuttingPlane``).  The bundle is seeded
    with the gradients of nearby points from the point set, with linear terms
    set from downshifted cutting planes, and enriched with trial points until a
    sufficient model decrease is realized.
    """

    def compute(
        self,
        evaluate,
        x,
        f,
        g,
        point_set,
        inverse_hessian,
        stationarity_radius,
        radii_update_check,
    ) -> DirectionResult:
        if self.try_gradient_step:
            result = self._try_gradient_step(
                evaluate, x, f, g, inverse_hessian, radii_update_check
            )
            if result is not None:
                return result

        gradients = [g]
        cut_values = [f]
        for point in point_set:
            if float((x - point.x).abs().max()) <= stationarity_radius:
                gradients.append(point.g)
                cut_values.append(self._cut_value(x, f, point, linearize=True))

        return self._inner_loop(
            evaluate,
            x,
            f,
            point_set,
            inverse_hessian,
            stationarity_radius,
            radii_update_check,
            gradients,
            cut_values,
            linearize=True,
        )


class GradientCombination(DirectionComputation):
    """
    Gradient-sampling-style direction computation (port of
    ``NonOptDirectionComputationGradientCombination``).  In addition to nearby
    points from the point set, gradients are evaluated at randomly sampled
    points within the stationarity radius of the current iterate; linear terms
    use the downshifted value only.

    :param random_sample_factor: Number of points to sample per subproblem
        solve.  If at least 1, it is the absolute number of points; otherwise
        the number sampled is this factor times the number of variables
        (rounded down, at least one point is always sampled).
    :type random_sample_factor: float
    :param generator: Optional ``torch.Generator`` for reproducible sampling.
    :type generator: torch.Generator
    """

    def __init__(
        self,
        random_sample_factor: float = 10,
        generator: torch.Generator = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.random_sample_factor = random_sample_factor
        self.generator = generator

    def _sample_count(self, n: int) -> int:
        if self.random_sample_factor >= 1.0:
            return int(self.random_sample_factor)
        return max(1, int(self.random_sample_factor * n))

    def compute(
        self,
        evaluate,
        x,
        f,
        g,
        point_set,
        inverse_hessian,
        stationarity_radius,
        radii_update_check,
    ) -> DirectionResult:
        if self.try_gradient_step:
            result = self._try_gradient_step(
                evaluate, x, f, g, inverse_hessian, radii_update_check
            )
            if result is not None:
                return result

        gradients = [g]
        cut_values = [f]

        def sample(gradients, cut_values):
            for _ in range(self._sample_count(x.numel())):
                perturbation = (
                    torch.rand(
                        x.shape, dtype=x.dtype, device=x.device, generator=self.generator
                    )
                    * 2.0
                    - 1.0
                )
                x_sample = x + stationarity_radius * perturbation
                f_sample, g_sample = evaluate(x_sample)
                sampled_point = Point(x_sample, f_sample, g_sample)
                point_set.add(sampled_point)
                gradients.append(g_sample)
                cut_values.append(self._cut_value(x, f, sampled_point, linearize=False))

        for point in point_set:
            if float((x - point.x).abs().max()) <= stationarity_radius:
                gradients.append(point.g)
                cut_values.append(self._cut_value(x, f, point, linearize=False))
        sample(gradients, cut_values)

        return self._inner_loop(
            evaluate,
            x,
            f,
            point_set,
            inverse_hessian,
            stationarity_radius,
            radii_update_check,
            gradients,
            cut_values,
            linearize=False,
            sample=sample,
        )
