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
        dual_bound : Union[
            float, Tensor
        ] = 100,
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
            # custom_project_fn=custom_project_fn
        )

        super().__init__(params, defaults)

        # self.param_groups.append()

        self.m = m
        self.dual_lr = dual_lr
        self.dual_bound = dual_bound
        self.rho = rho
        self.beta = beta
        self.mu = mu
        self.c_vals: list[Union[float, Tensor]] = []
        self._c_val_average = [None]
        # essentially, move everything here to self.state[param_group]
        # self.state[param_group]['smoothing_avg'] <= z for that param_group;
        # ...['grad'] <= grad w.r.t. that param_group
        # ...['G'] <= G w.r.t. that param_group // idk if necessary
        # ...['c_grad'][c_i] <= grad of ith constraint w.r.t. that group<w
        if init_dual_vars is not None:
            self._dual_vars = init_dual_vars
        else:
            self._dual_vars = torch.zeros(m, requires_grad=False)

    def _init_group(self, group, params, grads, c_grads, smoothing):
        # SHOULDN'T calculate values, only set them from the state of the respective param_group
        # calculations only happen in step() (or rather in the func version of step)
        has_sparse_grad = False

        for p in group["params"]:
            state = self.state[p]

            params.append(p)
            
            # load z (smoothing term)
            # Lazy state initialization
            if len(state) == 0:
                state["smoothing"] = p.detach().clone()
                state["c_grad"] = []

            smoothing.append(state.get("smoothing"))

            grads.append(p.grad)
            c_grads.append(state.get("c_grad"))

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

        # c_vals is cleaned in step()
        self.c_vals.append(c_val.detach())

        # update dual multipliers
        dual_update_tensor = torch.zeros_like(self._dual_vars)
        dual_update_tensor[i] = self.dual_lr * c_val
        self._dual_vars.add_(dual_update_tensor)
        for i in range(len(self._dual_vars)):
            if self._dual_vars[i] >= self.dual_bound: #or self._dual_vars[i] < 0:
                self._dual_vars[i].zero_()

        # save constraint grad
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            _ = self._init_group(group, params, grads, c_grads, smoothing)

            for p in group["params"]:
                state = self.state[p]
                # state['c_grad'] is cleaned in step()
                # so it is always empty on dual_step()
                state["c_grad"].append(p.grad)

    @_use_grad_for_differentiable
    def step(self, c_val: Union[Iterable | Tensor] = None):
        r"""Perform an update of the primal parameters (network weights & slack variables). To be called AFTER :func:`dual_step` in an iteration!

        Args:
            c_val (Tensor): an Iterable of estimates of values of **ALL** constraints; used for primal parameter update.
                Ideally, must be evaluated on an independent sample from the one used in :func:`dual_step`
        """
        
        if c_val is None:
            c_val = self.c_vals
        if isinstance(c_val, Iterable) and not isinstance(c_val, torch.Tensor):
            # if len(c_val) == 1 and isinstance(c_val[0], torch.Tensor):
            #     c_val = c_val[0]
            # else:
            c_val = torch.stack(c_val)
            if c_val.ndim > 1:
                c_val = c_val.squeeze(-1)
                
        if c_val.numel() != self.m:
            raise ValueError(f"Number of elements in c_val must be equal to m={self.m}, got {c_val.numel()}")
        G = []

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            lr = group["lr"]
            _ = self._init_group(group, params, grads, c_grads, smoothing)

            for i, param in enumerate(params):
                ### calculate Lagrange f-n gradient (G) ###

                # stack list of grads w.r.t. constraints to get
                # tensor of shape (*param.shape, m)
                l_term_grad = 0
                aug_term_grad = 0
                # if c_grads[i] is not None:
                for j, c_grad in enumerate(c_grads[i]):
                    if c_grad is None:
                        continue
                    l_term_grad += c_grad * self._dual_vars[j]
                    aug_term_grad += c_grad * c_val[j]

                G_i = (
                    grads[i]
                    + l_term_grad
                    + self.rho * aug_term_grad
                    + self.mu * (param - smoothing[i])
                )
                G.append(G_i)

                smoothing[i].add_(param - smoothing[i], alpha=self.beta)

                param.add_(G_i, alpha=-lr)

                ## PROJECT (keep in mind we do layer by layer)
                ## add slack variables to params in constructor?

                c_grads[i].clear()
        
        self.c_vals.clear()
        return G
