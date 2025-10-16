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

    def _init_group(
            self,
            group,
            params,
            grads,
            l_term_grads, # gradient of the lagrangian term, updated from parameter gradients during dual_step
            aug_term_grads, # gradient of the regularization term, updated from parameter gradients during dual_step
            smoothing
        ):
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
                state["l_term_grad"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["aug_term_grad"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            smoothing.append(state.get("smoothing"))

            grads.append(p.grad)

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

        if c_val.numel() != 1 or c_val.ndim > 0:
            raise ValueError(f"`dual_step` expected a scalar `c_val`, got an object of shape {c_val.shape}")
        self._dual_vars[i].add_(c_val.detach(), alpha=self.dual_lr)
        
        for j in range(len(self._dual_vars)):
            if self._dual_vars[j] >= self.dual_bound:
                self._dual_vars[j].zero_()

        # save constraint grad
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            l_term_grads: list[Tensor] = []
            aug_term_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            _ = self._init_group(group, params, grads, l_term_grads, aug_term_grads, smoothing)

            for p_i, p in enumerate(params):
                if p.grad is None:
                    continue
                l_term_grads[p_i].add_(p.grad, alpha=self._dual_vars[i].item())
                aug_term_grads[p_i].add_(p.grad, alpha=c_val.item())


    @_use_grad_for_differentiable
    def step(self, c_val: Union[Iterable | Tensor] = None):
        r"""Perform an update of the primal parameters (network weights & slack variables). To be called AFTER :func:`dual_step` in an iteration!

        Args:
            c_val (Tensor): an Iterable of estimates of values of **ALL** constraints; used for primal parameter update.
                Ideally, must be evaluated on an independent sample from the one used in :func:`dual_step`
        """
        
        # if c_val is None:
        #     c_val = self.c_vals
        # if isinstance(c_val, Iterable) and not isinstance(c_val, torch.Tensor):
        #     # if len(c_val) == 1 and isinstance(c_val[0], torch.Tensor):
        #     #     c_val = c_val[0]
        #     # else:
        #     c_val = torch.stack(c_val)
        #     if c_val.ndim > 1:
        #         c_val = c_val.squeeze(-1)
                
        # if c_val.numel() != self.m:
        #     raise ValueError(f"Number of elements in c_val must be equal to m={self.m}, got {c_val.numel()}")
        
        # G = []

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            l_term_grads: list[Tensor] = []
            aug_term_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            lr = group["lr"]
            _ = self._init_group(group, params, grads, l_term_grads, aug_term_grads, smoothing)

            for i, param in enumerate(params):
                ### calculate Lagrange f-n gradient (G) ###

                
                G_i = torch.zeros_like(param)
                G_i.add_(grads[i]).add_(l_term_grads[i]).add_(aug_term_grads[i], alpha=self.rho).add_(param - smoothing[i],alpha=self.mu)
                
                l_term_grads[i].zero_()
                aug_term_grads[i].zero_()

                smoothing[i].add_(smoothing[i], alpha=-self.beta).add_(param,alpha=self.beta)

                param.add_(G_i, alpha=-lr)

                ## PROJECT (keep in mind we do layer by layer)
                ## add slack variables to params in constructor?
