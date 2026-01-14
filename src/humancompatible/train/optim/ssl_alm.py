from typing import Iterable, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable


class SSLALM(Optimizer):
    def __init__(
        self,
        params,
        m: int,
        # tau in paper
        lr: Union[float, Tensor] = 5e-2,
        # eta in paper
        dual_lr: Union[
            float, Tensor
        ] = 5e-2,  # keep as tensor for different learning rates for different constraints in the future? idk
        dual_bound: Union[float, Tensor] = 100,
        # penalty term multiplier
        rho: float = 1.0,
        # smoothing term multiplier
        mu: float = 2.0,
        # smoothing term update multiplier
        beta: float = 0.5,
        *,
        init_dual_vars: Optional[Tensor] = None,
        # whether some of the dual variables should not be updated
        fix_dual_vars: Optional[Tensor] = None,
        differentiable: bool = False,
        implicit_backward_in_step: bool = True,
        # custom_project_fn: Optional[Callable] = project_fn
    ):
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if isinstance(dual_lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor dual_lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if dual_lr < 0.0:
            raise ValueError(f"Invalid dual learning rate: {dual_lr}")
        if init_dual_vars is not None and len(init_dual_vars) != m:
            raise ValueError(
                f"init_dual_vars should be of length m: expected {m}, got {len(init_dual_vars)}"
            )
        if fix_dual_vars is not None:
            raise NotImplementedError()
        if init_dual_vars is None and fix_dual_vars is not None:
            raise ValueError(
                f"if fix_dual_vars is not None, init_dual_vars should not be None."
            )

        if differentiable:
            raise NotImplementedError("TorchSSLALM does not support differentiable")

        defaults = dict(
            lr=lr,
            dual_lr=dual_lr,
            rho=rho,
            mu=mu,
            beta=beta,
            differentiable=differentiable,
        )

        super().__init__(params, defaults)

        self.m = m
        self.dual_lr = dual_lr
        self.dual_bound = dual_bound
        self.rho = rho
        self.c_vals: list[Union[float, Tensor]] = []
        self._c_val_average = [None]
        self._implicit_backward = implicit_backward_in_step
        self._constraints = []
        # essentially, move everything here to self.state[param_group]
        # self.state[param_group]['smoothing_avg'] <= z for that param_group;
        # ...['grad'] <= grad w.r.t. that param_group
        # ...['G'] <= G w.r.t. that param_group // idk if necessary
        # ...['c_grad'][c_i] <= grad of ith constraint w.r.t. that group<w
        if init_dual_vars is not None:
            self._dual_vars = init_dual_vars
        else:
            self._dual_vars = torch.zeros(m, requires_grad=False)

    def _init_group(
        self,
        group,
        params,
        grads,
        l_term_grads,  # gradient of the lagrangian term, updated from parameter gradients during dual_step
        aug_term_grads,  # gradient of the regularization term, updated from parameter gradients during dual_step
        smoothing,
    ):
        has_sparse_grad = False

        for p in group["params"]:
            state = self.state[p]

            params.append(p)

            if len(state) == 0:
                state["smoothing"] = p.detach().clone()
                if not self._implicit_backward:
                    state["l_term_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["aug_term_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            smoothing.append(state.get("smoothing"))

            grads.append(p.grad)

            if not self._implicit_backward:
                l_term_grads.append(state["l_term_grad"])
                aug_term_grads.append(state["aug_term_grad"])
                
        return has_sparse_grad

    def __setstate__(self, state):
        super().__setstate__(state)

    def dual_step(self, i: int, c_val: Tensor):
        r"""Perform an update of the dual parameters.
        Also saves constraint gradient for weight update. To be called BEFORE :func:`step` in an iteration!

        Args:
            i (int): index of the constraint
            c_val (Tensor): an estimate of the value of the constraint at which the gradient was computed; used for dual parameter update
        """
        # update dual multipliers
        if c_val.numel() != 1:
            raise ValueError(
                f"`dual_step` expected a scalar `c_val`, got shape {c_val.shape}"
            )
        self._dual_vars[i].add_(c_val.detach(), alpha=self.dual_lr)

        # check dual bound
        if self._dual_vars[i] >= self.dual_bound:
            self._dual_vars[i].zero_()
        
        if not self._implicit_backward:
            # save constraint grad
            self._save_param_grads(i, c_val)
        else:
            # save constraint values
            if c_val.ndim == 0:
                self._constraints.append(c_val.unsqueeze(0))
            else:
                self._constraints.append(c_val)

    def step(self, loss: torch.Tensor, constraints: Tensor = None):
        r"""Perform an update of the primal parameters (network weights & slack variables). To be called AFTER :func:`dual_step` in an iteration!

        Args:
            constraints (Tensor): A Tensor of estimates of values of **ALL** constraints; used for primal parameter update.
        """

        if self._implicit_backward:
            if constraints is None:
                constraints = torch.cat(self._constraints)

            L = lagrangian(loss, constraints, self._dual_vars, self.rho)
            L.backward()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            l_term_grads: list[Tensor] = []
            aug_term_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            lr = group["lr"]
            mu = group["mu"]
            beta = group["beta"]
            _ = self._init_group(
                group, params, grads, l_term_grads, aug_term_grads, smoothing
            )

            self.modified_sgd(params, grads, l_term_grads, aug_term_grads, smoothing, mu, beta, lr)
        
        self._constraints = []
        
    @_use_grad_for_differentiable
    def modified_sgd(self, params, grads, l_term_grads, aug_term_grads, smoothing, mu, beta, lr):
        for i, param in enumerate(params):
                ### calculate Lagrange f-n gradient (G) ###
            if self._implicit_backward:
                G_i = grads[i].add_(param - smoothing[i], alpha=mu)
            else:
                G_i = torch.zeros_like(param, memory_format=torch.preserve_format)
                G_i.add_(grads[i]).add_(l_term_grads[i]).add_(
                        aug_term_grads[i], alpha=self.rho
                    ).add_(param - smoothing[i], alpha=mu)

                l_term_grads[i].zero_()
                aug_term_grads[i].zero_()

            smoothing[i].add_(smoothing[i], alpha=-beta).add_(
                    param, alpha=beta
                )

            param.add_(G_i, alpha=-lr)

        
def lagrangian(loss: torch.Tensor, constraints: torch.Tensor, dual_vars: torch.Tensor, rho: float):
    return loss + dual_vars @ constraints + 0.5*rho*torch.square(torch.linalg.norm(constraints, ord=2))

def _save_param_grads(self, i, c_val):
    for group in self.param_groups:
        params: list[Tensor] = []
        grads: list[Tensor] = []
        l_term_grads: list[Tensor] = []
        aug_term_grads: list[Tensor] = []
        smoothing: list[Tensor] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        max_exp_avg_sqs: list[Tensor] = []
        state_steps: list[Tensor] = []
        _ = self._init_group(
                group,
                params,
                grads,
                l_term_grads,
                aug_term_grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                smoothing,
            )
        for p_i, p in enumerate(params):
            if p.grad is None:
                continue
            l_term_grads[p_i].add_(p.grad, alpha=self._dual_vars[i].item())
            aug_term_grads[p_i].add_(p.grad, alpha=c_val.item())