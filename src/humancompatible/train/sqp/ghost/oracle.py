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


def evaluate_oracle(net, objective_fn, constraint_fns, batch):
    """Evaluate the SAA objective gradient, constraint values and Jacobian.

    Parameters
    ----------
    net : torch.nn.Module
        The model whose parameters x are being optimized.
    objective_fn : callable(net, batch) -> scalar tensor
        Mean objective loss over ``batch`` (its gradient is the SAA average of
        ``∇f``).
    constraint_fns : sequence of callable(net, batch) -> scalar tensor
        Each returns the mean of constraint ``c_i`` over ``batch``; the
        constraint is interpreted as ``c_i(x) <= 0``.
    batch : object
        Whatever ``objective_fn`` / ``constraint_fns`` consume (e.g. an
        ``(X, y)`` tuple). The same ``batch`` is fed to every function.

    Returns
    -------
    grad_f : (n,) ndarray
    cval   : (m,) ndarray
    cgrad  : (m, n) ndarray
    """
    params = list(net.parameters())

    grad_f = _flat_grad(objective_fn(net, batch), params)
    n = grad_f.shape[0]
    m = len(constraint_fns)

    cval = np.empty(m, dtype=float)
    cgrad = np.empty((m, n), dtype=float)
    for i, c_fn in enumerate(constraint_fns):
        c = c_fn(net, batch)
        cval[i] = float(c.detach().cpu())
        cgrad[i] = _flat_grad(c, params)

    return grad_f, cval, cgrad
