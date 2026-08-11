"""Test problems shared by the paper experiments."""

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
from torch import Tensor


@dataclass
class Problem:
    """A constrained problem with, where available, a reference solution.

    The constraint convention is the package's: ``constraints(params)`` returns a
    flat tensor that should be ``<= 0``.

    :param make_params: Returns freshly initialised parameters at the problem's
        standard starting point. Called once per run so methods cannot share
        state.
    :param y_star: Reference multipliers, or ``None`` when none exist (a
        nonconvex problem has no a-priori multipliers, so only the KKT residual
        at the returned iterate is meaningful).
    :param grad_lipschitz: Lipschitz constant of ``grad f`` (for the quadratic
        problems here, ``max |eig(Q)|``). Together with ``jac_norm_sq`` it lets a
        driver derive a primal step size instead of hard-coding one.
    :param jac_norm_sq: ``||J||_2^2``, the spectral norm squared of the
        constraint Jacobian, which is what a quadratic penalty of coefficient
        ``rho`` adds to the surrogate's curvature.
    """

    name: str
    m: int
    make_params: Callable[[], list[Tensor]]
    objective: Callable[[list[Tensor]], Tensor]
    constraints: Callable[[list[Tensor]], Tensor]
    y_star: Optional[np.ndarray] = None
    x_star: Optional[np.ndarray] = None
    f_star: Optional[float] = None
    is_convex: bool = True
    notes: str = ""
    grad_lipschitz: Optional[float] = None
    jac_norm_sq: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def has_reference_multipliers(self) -> bool:
        return self.y_star is not None

    def scaled(self, alpha: float) -> "Problem":
        """The same problem with its constraints scaled by ``alpha > 0``.

        Scaling ``c`` by ``alpha`` leaves the feasible set and ``x*`` untouched and
        divides the multipliers by ``alpha``, since stationarity reads
        ``grad f + (alpha J)' (y*/alpha) = grad f + J' y*``. A faithful dual rule
        must therefore land on ``y*/alpha`` — anything else means an absolute
        constraint scale leaked into the update.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive; got {alpha}")
        base = self.constraints
        return replace(
            self,
            name=f"{self.name}_x{alpha:g}",
            constraints=lambda params: alpha * base(params),
            y_star=None if self.y_star is None else self.y_star / alpha,
            # A quadratic penalty on the scaled constraints has alpha^2 the
            # curvature, so the derived primal step has to see it.
            jac_norm_sq=None if self.jac_norm_sq is None
                        else self.jac_norm_sq * alpha**2,
            notes=f"{self.notes}; constraints scaled by {alpha:g}",
        )

    def primal_step(self, penalty_coefficient: float = 0.0) -> float:
        """A step size that is ``1/L`` for the surrogate's smooth part.

        ``L = L_f + rho ||J||^2`` bounds the curvature of
        ``f + y'c + rho/2 ||[c]_+||^2`` for a quadratic ``f`` and linear ``c``,
        uniformly in ``y`` (the linear term adds no curvature). Deriving the step
        this way rather than tuning it keeps the E0a comparison about dual
        dynamics, and makes visible that a method carrying a large quadratic term
        must pay for it with a smaller primal step.
        """
        if self.grad_lipschitz is None or self.jac_norm_sq is None:
            raise ValueError(f"{self.name} declares no Lipschitz data")
        return 1.0 / (self.grad_lipschitz + penalty_coefficient * self.jac_norm_sq)
