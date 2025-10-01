from typing import Iterable, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable


class SSLALM_Adam(Optimizer):
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
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
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
                "if fix_dual_vars is not None, init_dual_vars should not be None."
            )

        if differentiable:
            raise NotImplementedError("TorchSSLALM does not support differentiable")

        defaults = dict(
            lr=lr,
            dual_lr=dual_lr,
            rho=rho,
            mu=mu,
            beta=beta,
            amsgrad=amsgrad,
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu
        self.c_vals: list[Union[float, Tensor]] = []
        self._c_val_average = [None]
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
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
        c_grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        smoothing,
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
                state["c_grad"] = []
                state["obj_grad"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

                state["step"] = torch.tensor(0.0)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

                if group["amsgrad"]:
                    # raise NotImplementedError()
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in differentiable mode"
                )

            smoothing.append(state.get("smoothing"))

            state_steps.append(state["step"])

            # grads.append(p.grad)
            grads.append(state.get("obj_grad"))
            c_grads.append(state.get("c_grad"))
        return has_sparse_grad

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)

    # TODO: saving gradients of all parameters w.r.t. all constraints at once requires LOTS of memory
    def _save_grad_obj(self, sum_aggregate=False):
        r"""Function that saves the gradient to a dict. Needed for working with gradients of several functions with respect to the parameters.

        Args:
            sum_aggregate: (bool): if a value is already present, whether to add the current gradient (`True`) or overwrite (`False`). Used for working with multiple constraints.
        """

        # save constraint grad
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            _ = self._init_group(
                group,
                params,
                grads,
                c_grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                smoothing,
            )

            for p in group["params"]:
                state = self.state[p]
                # state['c_grad'] is cleaned in step()
                # so it is always empty on dual_step()
                if sum_aggregate:
                    state["obj_grad"].add_(p.grad)
                else:
                    state["obj_grad"] = p.grad

    # TODO: saving gradients of all parameters w.r.t. all constraints at once requires LOTS of memory
    def _save_grad_constr(self, i: int, sum_aggregate=False):
        r"""Function that saves the gradient to a dict. Needed for working with gradients of several functions with respect to the parameters.

        Args:
            i (int): index of the constraint
            sum_aggregate: (bool): if a value is already present, whether to add the current gradient (`True`) or overwrite (`False`). Used for working with multiple constraints.
        """

        # save constraint grad
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            _ = self._init_group(
                group,
                params,
                grads,
                c_grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                smoothing,
            )

            for p in group["params"]:
                state = self.state[p]
                # state['c_grad'] is cleaned in step()
                # so it is always empty on dual_step()
                if len(state["c_grad"]) <= i:
                    state["c_grad"].append(p.grad)
                elif sum_aggregate:
                    state["c_grad"][i].add_(p.grad)
                else:
                    state["c_grad"][i] = p.grad

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
            if self._dual_vars[i] >= self.dual_bound:  # or self._dual_vars[i] < 0:
                self._dual_vars[i].zero_()

        self._save_grad_constr(i, sum_aggregate=False)

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
            raise ValueError(
                f"Number of elements in c_val must be equal to m={self.m}, got {c_val.numel()}"
            )
        G = []

        self._save_grad_obj(sum_aggregate=False)

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            smoothing: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            lr = group["lr"]
            amsgrad = group["amsgrad"]

            _ = self._init_group(
                group,
                params,
                grads,
                c_grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                smoothing,
            )

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

                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]
                step_t += 1
                beta1 = self.beta1
                beta2 = self.beta2
                eps = self.eps

                exp_avg.lerp_(G_i, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(G_i, G_i, value=1 - beta2)

                smoothing[i].add_(param - smoothing[i], alpha=self.beta)

                # param.add_(G_i, alpha=-lr)

                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2**0.5

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(
                        max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i]
                    )

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(
                        eps
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)

                ## PROJECT (keep in mind we do layer by layer)
                ## add slack variables to params in constructor?

                c_grads[i].clear()

        self.c_vals.clear()
        return G
