"""
Stochastic-approximation (SAA) oracle for the Ghost-penalty SQP method.

Given a minibatch, returns the sample-average-approximation of the quantities
that define subproblem (22) in Facchinei & Kungurtsev (2025):

    grad_f = (1/J) Σ_j ∇f(x, ξ_j)        objective gradient
    cval   = (1/J) Σ_j c_i(x, ζ_j)       constraint values        (i = 1..m)
    cgrad  = (1/J) Σ_j ∇c_i(x, ζ_j)      constraint Jacobian rows

Because a *mean* loss over a minibatch back-propagates to the *averaged*
gradient, a single ``torch.autograd.grad`` call per quantity already yields the
SAA average -- no manual per-sample averaging is required.
"""

from typing import Callable, List, Sequence

import numpy as np
import torch


def n_params(net):
    """Total number of (trainable) scalar parameters n of ``net``."""
    return sum(p.numel() for p in net.parameters())


def _flat_grad(scalar, params):
    """Flatten dL/dparams into a single 1-D numpy vector of length n.

    ``allow_unused=True`` lets a constraint depend on only part of the network
    (the unused parameters contribute a zero gradient block).
    """
    grads = torch.autograd.grad(scalar, params, allow_unused=True, retain_graph=False)
    flat = [
        (p.new_zeros(p.numel()) if g is None else g.reshape(-1))
        for p, g in zip(params, grads)
    ]
    return torch.cat(flat).detach().cpu().numpy()


def evaluate_oracle_from_tensors(
        model: torch.nn.Module,
        objective: torch.Tensor,
        constraints: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the oracle: objective gradient and constraint Jacobian"""

    params = list(model.parameters())

    # objective
    grad_f = _flat_grad(objective, params)
    n = grad_f.shape[0]
    
    # constraints (if passed as a single tensor)
    # if isinstance(constraints, torch.Tensor):
    m = constraints.numel()
    grads = torch.autograd.grad(
        constraints.reshape(-1),
        params,
        grad_outputs=torch.eye(m, dtype=constraints.dtype, device=constraints.device),
        is_grads_batched=True,
        allow_unused=True,
    )
    cgrad = torch.cat([
        (p.new_zeros(m, p.numel()) if g is None else g.reshape(m, -1))
        for p, g in zip(params, grads)
    ], dim=1)
    return grad_f, constraints.detach().cpu().numpy().astype(float), cgrad.detach().cpu().numpy().astype(float)


def evaluate_oracle(
        net: torch.nn.Module,
        objective_fn: Callable[..., torch.Tensor],
        constraint_fns: Sequence[Callable[..., torch.Tensor]] | Callable[..., torch.Tensor],
        batch
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the SAA objective gradient, constraint values and Jacobian.

    Parameters
    ----------
    net : torch.nn.Module
        The model whose parameters x are being optimized.
    objective_fn : callable(net, batch) -> scalar tensor
        Mean objective loss over ``batch`` (its gradient is the SAA average of
        ``∇f``).
    constraint_fns : sequence of callable(net, batch) -> scalar tensor, or
        a single callable(net, batch) -> (m,) tensor
        Each returns the mean of constraint ``c_i`` over ``batch``; the
        constraint is interpreted as ``c_i(x) <= 0``. The vector form shares
        one forward pass and gets the Jacobian from one batched backward.
    batch : object
        Whatever ``objective_fn`` / ``constraint_fns`` consume (e.g. an
        ``(X, y)`` tuple). The same ``batch`` is fed to every function.

    Returns
    -------
    grad_f : (n,) ndarray
    cval   : (m,) ndarray
    cgrad  : (m, n) ndarray
    """

    if callable(constraint_fns):
        cvals = constraint_fns(net, batch).reshape(-1)
    else:
        cvals = torch.stack([c_fn(net, batch) for c_fn in constraint_fns])

    fval = objective_fn(net, batch)

    grad_f, cval, cgrad = evaluate_oracle_from_tensors(net, fval, cvals)

    # grad_f = _flat_grad(objective_fn(net, batch), params)
    # n = grad_f.shape[0]

    # if constraint_fns passed as a single R^n -> R^m function instead of a list of scalar functions,
    # we can use is_grads_batched=True to do a single backward pass instead of m
    # if callable(constraint_fns):
    #     c = constraint_fns(net, batch).reshape(-1) # reshape to flatten
    #     m = c.numel()
    #     grads = torch.autograd.grad(
    #         c, params, grad_outputs=torch.eye(m, dtype=c.dtype, device=c.device),
    #         is_grads_batched=True, allow_unused=True,
    #     )
    #     cgrad = torch.cat([
    #         (p.new_zeros(m, p.numel()) if g is None else g.reshape(m, -1))
    #         for p, g in zip(params, grads)
    #     ], dim=1)
    #     return grad_f, c.detach().cpu().numpy().astype(float), cgrad.detach().cpu().numpy().astype(float)

    # m = len(constraint_fns)

    # cval = np.empty(m, dtype=float)
    # cgrad = np.empty((m, n), dtype=float)
    # for i, c_fn in enumerate(constraint_fns):
    #     c = c_fn(net, batch)
    #     cval[i] = float(c.detach().cpu())
    #     cgrad[i] = _flat_grad(c, params)

    return grad_f, cval, cgrad
