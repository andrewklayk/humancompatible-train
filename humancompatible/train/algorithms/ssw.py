from typing import Iterable, Optional, Union

import torch
from torch import Tensor

from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

# def project_fn(x, m):
#     for i in range(1, m + 1):
#         if x[-i] < 0:
#             x[-i] = 0
#     return x


def _dual_step_func(dual_var, lr, cval):
    return dual_var + lr * cval


# def step_fn(params, grads,)


class SSG(Optimizer):
    def __init__(
        self,
        params,
        m: int = 1,
        # constraint tolerance
        # ctols: Union[
        #     float, Tensor
        # ],
        # ctols_rule = "const",
        # learning rate
        lr: Union[float, Tensor] = 5e-2,
        # learning rate decrease rule
        # lr_rule = "const",
        # constraint learning rate
        dual_lr: Union[
            float, Tensor
        ] = 5e-2,  # keep as tensor for different learning rates for different constraints in the future? idk
        *,
        differentiable: bool = False,
    ):
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if isinstance(dual_lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor dual_lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if dual_lr < 0.0:
            raise ValueError(f"Invalid dual learning rate: {dual_lr}")
        if not (m == 1):
            raise ValueError(f"Switching Subgradient does not support multiple constraints."
                             "Consider taking the largest violation at each iteration.")
        if differentiable:
            raise NotImplementedError("SSw does not support differentiable")

        defaults = dict(
            lr=lr,
            dual_lr=dual_lr,
            differentiable=differentiable,
        )

        super().__init__(params, defaults)

        # self.param_groups.append()

        self.m = m
        self.lr = lr
        self.dual_lr = dual_lr
        # self.lr_rule = lr_rule
        # self.dual_lr_rule = dual_lr_rule
        self.c_vals: list[Union[float, Tensor]] = []
        # self.ctols = ctols
        # essentially, move everything here to self.state[param_group]
        # self.state[param_group]['smoothing_avg'] <= z for that param_group;
        # ...['grad'] <= grad w.r.t. that param_group
        # ...['G'] <= G w.r.t. that param_group // idk if necessary
        # ...['c_grad'][c_i] <= grad of ith constraint w.r.t. that group<w

    def _init_group(self, group, params, grads, c_grads):
        # SHOULDN'T calculate values, only set them from the state of the respective param_group
        # calculations only happen in step() (or rather in the func version of step)
        has_sparse_grad = False

        for p in group["params"]:
            state = self.state[p]

            params.append(p)

            # Lazy state initialization
            if len(state) == 0:
                state["c_grad"] = []

            grads.append(p.grad)
            c_grads.append(state.get("c_grad"))

        return has_sparse_grad

    def __setstate__(self, state):
        super().__setstate__(state)

    def dual_step(self, i: int, c_val: Tensor = None):
        r"""Save constraint gradient for weight update. To be called BEFORE :func:`step` in an iteration!

        Args:
            i (int): index of the constraint **(unused)**
            c_val (Tensor): an estimate of the value of the constraint at which the gradient was computed **(unused)**
        """
        
        if i > self.m:
            raise ValueError("SSw does not support multiple constraints.")

        # save constraint grad
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            _ = self._init_group(group, params, grads, c_grads)

            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    # state['c_grad'] is cleaned in step()
                    # so it is always empty on dual_step()
                    state["c_grad"].append(p.grad)

    @_use_grad_for_differentiable
    def step(self, c_val: Union[Iterable | Tensor]):
        r"""Perform an update of the primal parameters (network weights). To be called AFTER :func:`dual_step` in an iteration!

        Args:
            c_val (Tensor): an Iterable of estimates of values of **ALL** constraints; used for primal parameter update.
                Ideally, must be evaluated on an independent sample from the one used in :func:`dual_step`
        """

        # here assume c_val is a scalar

        update_con = c_val > 0
        
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            lr = group["lr"]
            _ = self._init_group(group, params, grads, c_grads)

            for i, param in enumerate(params):

                if update_con:
                    param.add_(c_grads[i][0], alpha=-self.dual_lr)
                else:
                    param.add_(param.grad, alpha=-lr)

                if c_grads[i] is not None:
                    c_grads[i].clear()