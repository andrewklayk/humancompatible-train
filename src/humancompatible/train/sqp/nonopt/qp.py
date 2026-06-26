"""
Solver for the quadratic subproblem arising in the NonOpt direction computations.

The subproblem is the dual of the (proximal) cutting-plane / gradient-combination
subproblem:

.. math::
    \\min_{\\omega \\in \\Delta^k} \\;
    \\tfrac{1}{2} \\omega^T (G^T W G) \\omega - b^T \\omega,

where :math:`\\Delta^k` is the unit simplex, the columns of :math:`G` are the
(sub)gradients in the bundle, :math:`W` is an inverse Hessian approximation and
:math:`b` collects the linear (cutting-plane) terms.  The primal search direction
is recovered as :math:`d = -W G \\omega`.

The reference C++ implementation (https://github.com/frankecurtis/NonOpt) solves
this with a specialized dual active-set method; since the subproblem dimension
equals the bundle size (small), an accelerated projected-gradient method with an
exact simplex projection is used here instead.
"""

import torch
from torch import Tensor


def project_onto_simplex(v: Tensor) -> Tensor:
    """
    Computes the Euclidean projection of a vector onto the unit simplex
    :math:`\\{\\omega : \\omega \\geq 0, \\sum_i \\omega_i = 1\\}`.

    :param v: Vector to project.
    :type v: torch.Tensor
    :return: Projection of `v` onto the unit simplex.
    :rtype: torch.Tensor
    """
    k = v.numel()
    u, _ = torch.sort(v, descending=True)
    cumulative = torch.cumsum(u, dim=0) - 1.0
    indices = torch.arange(1, k + 1, dtype=v.dtype, device=v.device)
    positive = u - cumulative / indices > 0
    if not positive.any():  # only possible with non-finite input
        return torch.full_like(v, 1.0 / k)
    rho = int(torch.nonzero(positive)[-1].item())
    theta = cumulative[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


def solve_simplex_qp(
    Q: Tensor,
    b: Tensor,
    tol: float = 1e-10,
    max_iterations: int = None,
) -> Tensor:
    """
    Solves :math:`\\min_{\\omega \\in \\Delta^k} \\tfrac{1}{2}\\omega^T Q \\omega - b^T \\omega`
    over the unit simplex with an accelerated projected-gradient (FISTA) method.

    :param Q: Symmetric positive semi-definite matrix of shape ``(k, k)``.
    :type Q: torch.Tensor
    :param b: Linear term of shape ``(k,)``.
    :type b: torch.Tensor
    :param tol: Tolerance on the projected-gradient KKT residual.
    :type tol: float
    :param max_iterations: Iteration limit; defaults to ``max(200, 20 * k)``.
    :type max_iterations: int
    :return: Approximate solution ``omega`` of shape ``(k,)``.
    :rtype: torch.Tensor
    """
    k = b.numel()
    if k == 1:
        return torch.ones_like(b)

    Q = 0.5 * (Q + Q.t())

    # Lipschitz constant of the gradient; k is small, so an exact eigenvalue is cheap
    try:
        lipschitz = float(torch.linalg.eigvalsh(Q)[-1])
    except Exception:
        lipschitz = float(Q.abs().sum(dim=1).max())

    if not lipschitz > tol:
        # negligible quadratic term: the linear program is solved at a vertex
        omega = torch.zeros_like(b)
        omega[int(torch.argmax(b))] = 1.0
        return omega

    if max_iterations is None:
        max_iterations = max(200, 20 * k)

    omega = torch.full_like(b, 1.0 / k)
    accelerated = omega.clone()
    momentum = 1.0
    for _ in range(max_iterations):
        gradient = Q @ omega - b
        kkt_residual = (omega - project_onto_simplex(omega - gradient)).abs().max()
        if kkt_residual <= tol:
            break
        omega_new = project_onto_simplex(
            accelerated - (Q @ accelerated - b) / lipschitz
        )
        momentum_new = 0.5 * (1.0 + (1.0 + 4.0 * momentum**2) ** 0.5)
        accelerated = omega_new + ((momentum - 1.0) / momentum_new) * (
            omega_new - omega
        )
        # restart acceleration if it points uphill
        if torch.dot(omega_new - omega, gradient) > 0:
            accelerated = omega_new.clone()
            momentum_new = 1.0
        omega, momentum = omega_new, momentum_new

    return omega
