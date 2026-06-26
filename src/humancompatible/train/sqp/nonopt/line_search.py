"""
Line search strategies, ports of ``NonOptLineSearchWeakWolfe`` and
``NonOptLineSearchBacktracking`` from https://github.com/frankecurtis/NonOpt.

Both searches measure sufficient decrease against a model-based reference value
supplied by the direction computation,
``min(d' H d, max(||G w||^2, ||d||^2))``, rather than the directional
derivative, as nonsmoothness makes the latter unreliable.
"""

from typing import Callable, NamedTuple, Tuple

from torch import Tensor


class LineSearchResult(NamedTuple):
    """Outcome of a line search: accepted stepsize and trial point data."""

    stepsize: float
    x: Tensor
    f: float
    g: Tensor


def weak_wolfe(
    evaluate: Callable[[Tensor], Tuple[float, Tensor]],
    x: Tensor,
    f: float,
    g: Tensor,
    d: Tensor,
    stepsize_previous: float,
    decrease_reference: float,
    *,
    stepsize_initial: float = 1.0,
    stepsize_minimum: float = 1e-20,
    stepsize_maximum: float = 1e+02,
    sufficient_decrease_threshold: float = 1e-10,
    sufficient_decrease_fudge_factor: float = 1e-10,
    curvature_threshold: float = 9e-01,
    curvature_fudge_factor: float = 1e-10,
    stepsize_decrease_factor: float = 5e-01,
    stepsize_increase_factor: float = 1e+01,
    stepsize_bound_tolerance: float = 1e-20,
) -> LineSearchResult:
    """
    Weak Wolfe line search along direction ``d``.  Brackets a stepsize satisfying
    a sufficient decrease condition (relative to ``decrease_reference``) and a
    weak curvature condition.  If the bracketing interval collapses, any simple
    decrease is accepted; otherwise a null step (stepsize 0) is returned.

    :param evaluate: Callable mapping a flat iterate to ``(objective, gradient)``.
    :type evaluate: Callable
    :param x: Current iterate (flat).
    :type x: torch.Tensor
    :param f: Objective value at ``x``.
    :type f: float
    :param g: Gradient at ``x``.
    :type g: torch.Tensor
    :param d: Search direction.
    :type d: torch.Tensor
    :param stepsize_previous: Stepsize accepted in the previous iteration; the
        initial trial stepsize is ``stepsize_increase_factor`` times this value,
        capped by ``stepsize_initial``.
    :type stepsize_previous: float
    :param decrease_reference: Model decrease reference from the QP subproblem.
    :type decrease_reference: float
    :return: Accepted stepsize and trial point data.
    :rtype: LineSearchResult
    """
    directional_derivative = float(g.dot(d))
    lower = stepsize_minimum
    upper = stepsize_maximum
    stepsize = max(
        stepsize_minimum,
        min(stepsize_increase_factor * stepsize_previous, min(stepsize_initial, stepsize_maximum)),
    )

    while True:
        x_trial = x + stepsize * d
        f_trial, g_trial = evaluate(x_trial)

        sufficient_decrease = (
            f_trial - f
            <= -sufficient_decrease_threshold * stepsize * decrease_reference
            + sufficient_decrease_fudge_factor
        )
        if sufficient_decrease:
            curvature_condition = (
                float(g_trial.dot(d))
                >= curvature_threshold * directional_derivative - curvature_fudge_factor
            )
            if curvature_condition:
                return LineSearchResult(stepsize, x_trial, f_trial, g_trial)

        # interval collapsed: accept simple decrease or take a null step
        if (
            stepsize <= lower + stepsize_bound_tolerance
            or stepsize >= upper - stepsize_bound_tolerance
        ):
            if f_trial < f:
                return LineSearchResult(stepsize, x_trial, f_trial, g_trial)
            return LineSearchResult(0.0, x, f, g)

        if sufficient_decrease:
            lower = stepsize
        else:
            upper = stepsize
        stepsize = (1.0 - stepsize_decrease_factor) * lower + stepsize_decrease_factor * upper


def backtracking(
    evaluate: Callable[[Tensor], Tuple[float, Tensor]],
    x: Tensor,
    f: float,
    g: Tensor,
    d: Tensor,
    stepsize_previous: float,
    decrease_reference: float,
    *,
    stepsize_initial: float = 1.0,
    stepsize_minimum: float = 1e-20,
    sufficient_decrease_threshold: float = 1e-10,
    sufficient_decrease_fudge_factor: float = 1e-10,
    stepsize_decrease_factor: float = 5e-01,
    stepsize_increase_factor: float = 1e+01,
) -> LineSearchResult:
    """
    Backtracking (Armijo) line search along direction ``d``.  Same interface and
    acceptance reference as :func:`weak_wolfe`, but without a curvature
    condition.

    :return: Accepted stepsize and trial point data.
    :rtype: LineSearchResult
    """
    stepsize = max(
        stepsize_minimum,
        min(stepsize_increase_factor * stepsize_previous, stepsize_initial),
    )

    while True:
        x_trial = x + stepsize * d
        f_trial, g_trial = evaluate(x_trial)

        if (
            f_trial - f
            <= -sufficient_decrease_threshold * stepsize * decrease_reference
            + sufficient_decrease_fudge_factor
        ):
            return LineSearchResult(stepsize, x_trial, f_trial, g_trial)

        if stepsize <= stepsize_minimum:
            if f_trial < f:
                return LineSearchResult(stepsize, x_trial, f_trial, g_trial)
            return LineSearchResult(0.0, x, f, g)

        stepsize *= stepsize_decrease_factor
