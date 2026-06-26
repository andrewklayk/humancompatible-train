"""
Inverse Hessian approximations used by the NonOpt port, with the self-correcting
quasi-Newton updates of Curtis and Que.

Ports of ``NonOptApproximateHessianUpdateBFGS``/``DFP`` and
``NonOptSymmetricMatrixDense``/``LimitedMemory`` from
https://github.com/frankecurtis/NonOpt.  The classes here maintain the *inverse*
Hessian approximation :math:`W \\approx H^{-1}` directly, since the direction
computation only needs products :math:`W v`.
"""

import math
from collections import deque

import torch
from torch import Tensor


def self_correcting_scalar(
    u: float,
    v: float,
    w: float,
    correction_threshold_1: float,
    correction_threshold_2: float,
) -> float:
    """
    Computes the self-correcting BFGS damping scalar :math:`\\phi` such that the
    corrected gradient displacement :math:`\\tilde y = (1-\\phi) y + \\phi s`
    satisfies
    :math:`\\langle s, \\tilde y\\rangle / \\langle s, s\\rangle \\geq \\eta_1` and
    :math:`\\langle \\tilde y, \\tilde y\\rangle / \\langle s, \\tilde y\\rangle \\leq \\eta_2`.

    Direct port of ``ApproximateHessianUpdateBFGS::evaluateSelfCorrectingScalar``.

    :param u: Squared norm of the iterate displacement, :math:`\\|s\\|_2^2`.
    :type u: float
    :param v: Inner product :math:`\\langle s, y\\rangle`.
    :type v: float
    :param w: Squared norm of the gradient displacement, :math:`\\|y\\|_2^2`.
    :type w: float
    :param correction_threshold_1: Lower-bound threshold :math:`\\eta_1`.
    :type correction_threshold_1: float
    :param correction_threshold_2: Upper-bound threshold :math:`\\eta_2`.
    :type correction_threshold_2: float
    :return: Correction scalar in :math:`[0, 1]`.
    :rtype: float
    """
    eta1, eta2 = correction_threshold_1, correction_threshold_2

    # scalar for the lower bound on <s,y>/<s,s>
    scalar1 = 0.0
    if u <= 0.0:
        scalar1 = 1.0
    else:
        if v / u < eta1:
            if eta1 * u - v > 0.0 and u - v > 0.0:
                scalar1 = (eta1 * u - v) / (u - v)
            else:
                scalar1 = 0.0
        if (
            scalar1 > 0.0
            and scalar1**2 * u + scalar1 * (1.0 - scalar1) * v + (1.0 - scalar1) ** 2 * w
            > eta1 * (scalar1 * u + (1.0 - scalar1) * v)
        ):
            scalar1 = 1.0

    # scalar for the upper bound on <y,y>/<s,y>
    scalar2 = 0.0
    if v <= 0.0:
        scalar2 = 1.0
    elif w / v > eta2:
        temporary1 = u - 2.0 * v + w
        temporary2 = 2.0 * (v - w) - eta2 * (u - v)
        temporary3 = w - eta2 * v
        discriminant = temporary2**2 - 4.0 * temporary1 * temporary3
        if temporary3 > 0.0 and discriminant >= 0.0 and -temporary2 + math.sqrt(discriminant) > 0.0:
            scalar2 = 2.0 * temporary3 / (-temporary2 + math.sqrt(discriminant))
        else:
            scalar2 = 1.0

    return max(scalar1, scalar2)


class InverseHessian:
    """
    Base class for inverse Hessian approximations.  Implements the displacement
    correction and update-skipping logic shared by all updates.

    :param correction_threshold_1: Self-correction lower-bound threshold; if the
        update is corrected, the gradient displacement ``y`` is modified so that
        ``<s,y>/<s,s>`` is at least this value.
    :type correction_threshold_1: float
    :param correction_threshold_2: Self-correction upper-bound threshold; if the
        update is corrected, ``y`` is modified so that ``<y,y>/<s,y>`` is at most
        this value.
    :type correction_threshold_2: float
    :param norm_tolerance: Update is skipped if either displacement norm falls
        below this tolerance.
    :type norm_tolerance: float
    :param product_tolerance: Update is skipped unless
        ``<s,y> >= product_tolerance * ||s|| * ||y||``.
    :type product_tolerance: float
    :param initial_scaling: Whether to scale the initial matrix by
        ``<s,y>/<y,y>`` at the first successful update.
    :type initial_scaling: bool
    """

    def __init__(
        self,
        correction_threshold_1: float = 1e-08,
        correction_threshold_2: float = 1e+08,
        norm_tolerance: float = 1e-08,
        product_tolerance: float = 1e-20,
        initial_scaling: bool = False,
    ) -> None:
        self.correction_threshold_1 = correction_threshold_1
        self.correction_threshold_2 = correction_threshold_2
        self.norm_tolerance = norm_tolerance
        self.product_tolerance = product_tolerance
        self.initial_scaling = initial_scaling
        self.initial_update_performed = False

    def _corrected_displacements(self, s: Tensor, y: Tensor):
        """Returns the corrected gradient displacement, or None if the update
        should be skipped."""
        if float(s.norm()) <= self.norm_tolerance:
            return None
        if float(y.norm()) <= self.norm_tolerance:
            return None
        scalar = self_correcting_scalar(
            float(s.dot(s)),
            float(s.dot(y)),
            float(y.dot(y)),
            self.correction_threshold_1,
            self.correction_threshold_2,
        )
        if scalar > 0.0:
            y = (1.0 - scalar) * y + scalar * s
        if float(s.dot(y)) < self.product_tolerance * float(s.norm()) * float(y.norm()):
            return None
        return y

    def apply(self, v: Tensor) -> Tensor:
        """
        Computes the product :math:`W v`.

        :param v: Vector of shape ``(n,)``.
        :type v: torch.Tensor
        :return: Product :math:`W v`.
        :rtype: torch.Tensor
        """
        return self.apply_matrix(v.unsqueeze(1)).squeeze(1)

    def apply_matrix(self, V: Tensor) -> Tensor:
        """
        Computes the product :math:`W V` column-wise.

        :param V: Matrix of shape ``(n, k)``.
        :type V: torch.Tensor
        :return: Product :math:`W V` of shape ``(n, k)``.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def update(self, s: Tensor, y: Tensor) -> bool:
        """
        Updates the approximation from the iterate displacement ``s`` and the
        gradient displacement ``y`` (after self-correction).

        :param s: Iterate displacement of shape ``(n,)``.
        :type s: torch.Tensor
        :param y: Gradient displacement of shape ``(n,)``.
        :type y: torch.Tensor
        :return: Whether the update was performed (False if skipped).
        :rtype: bool
        """
        raise NotImplementedError


class LimitedMemoryInverseHessian(InverseHessian):
    """
    Limited-memory BFGS inverse Hessian approximation with self-correcting
    updates.  Products :math:`W V` are computed with the (vectorized) two-loop
    recursion; nothing of size :math:`n \\times n` is ever stored.

    :param history_size: Number of curvature pairs kept.
    :type history_size: int
    """

    def __init__(self, history_size: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history_size = history_size
        self.pairs = deque(maxlen=history_size)
        self.gamma = 1.0

    def reset(self) -> None:
        self.pairs.clear()
        self.gamma = 1.0
        self.initial_update_performed = False

    def update(self, s: Tensor, y: Tensor) -> bool:
        y = self._corrected_displacements(s, y)
        if y is None:
            return False
        if self.initial_scaling and not self.initial_update_performed:
            self.gamma = float(s.dot(y)) / float(y.dot(y))
            self.initial_update_performed = True
        rho = 1.0 / float(s.dot(y))
        self.pairs.append((s.clone(), y.clone(), rho))
        return True

    def apply_matrix(self, V: Tensor) -> Tensor:
        Q = V.clone()
        alphas = []
        for s, y, rho in reversed(self.pairs):
            alpha = rho * (s @ Q)  # shape (k,)
            Q -= torch.outer(y, alpha)
            alphas.append(alpha)
        R = self.gamma * Q
        for (s, y, rho), alpha in zip(self.pairs, reversed(alphas)):
            beta = rho * (y @ R)
            R += torch.outer(s, alpha - beta)
        return R


class DenseInverseHessian(InverseHessian):
    """
    Dense inverse Hessian approximation supporting self-correcting BFGS and DFP
    updates.  Stores the full ``(n, n)`` matrix; only suitable for problems with
    a moderate number of variables.

    :param formula: Quasi-Newton update formula, ``"bfgs"`` or ``"dfp"``.
    :type formula: str
    """

    def __init__(self, formula: str = "bfgs", **kwargs) -> None:
        super().__init__(**kwargs)
        if formula not in ("bfgs", "dfp"):
            raise ValueError(f"Unknown quasi-Newton update formula: {formula}!")
        self.formula = formula
        self.W = None

    def reset(self) -> None:
        self.W = None
        self.initial_update_performed = False

    def _materialize(self, like: Tensor) -> None:
        if self.W is None:
            n = like.numel()
            self.W = torch.eye(n, dtype=like.dtype, device=like.device)

    def update(self, s: Tensor, y: Tensor) -> bool:
        y = self._corrected_displacements(s, y)
        if y is None:
            return False
        self._materialize(s)
        if self.initial_scaling and not self.initial_update_performed:
            self.W = (float(s.dot(y)) / float(y.dot(y))) * torch.eye(
                s.numel(), dtype=s.dtype, device=s.device
            )
            self.initial_update_performed = True
        if self.formula == "bfgs":
            rho = 1.0 / float(s.dot(y))
            Wy = self.W @ y
            self.W -= rho * (torch.outer(s, Wy) + torch.outer(Wy, s))
            self.W += rho * (1.0 + rho * float(y.dot(Wy))) * torch.outer(s, s)
        else:  # dfp
            Wy = self.W @ y
            self.W -= torch.outer(Wy, Wy) / float(y.dot(Wy))
            self.W += torch.outer(s, s) / float(s.dot(y))
        return True

    def apply_matrix(self, V: Tensor) -> Tensor:
        if self.W is None:
            return V.clone()
        return self.W @ V
