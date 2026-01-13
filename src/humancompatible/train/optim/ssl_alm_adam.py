from typing import Iterable, Optional, Tuple, Union

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
        smoothing_beta: float = 0.5,
        betas: Tuple[float, float] = (0.9, 0.999),
        device="cpu",
        eps: float = 1e-8,
        amsgrad: bool = False,
        *,
        init_dual_vars: Optional[Tensor] = None,
        # whether some of the dual variables should not be updated
        fix_dual_vars: Optional[Tensor] = None,
        implicit_backward_in_step: bool = True,
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
            raise NotImplementedError("SSLALM does not support differentiable")

        defaults = dict(
            eps=eps,
            lr=lr,
            betas=betas,
            rho=rho,
            mu=mu,
            smoothing_beta=smoothing_beta,
            amsgrad=amsgrad,
            differentiable=differentiable,
        )

        super().__init__(params, defaults)

        self.m = m
        self.dual_lr = dual_lr
        self.dual_bound = dual_bound
        self.rho = rho
        self._constraint_params = dict(
            rho=rho,
            implicit_backward_in_step=implicit_backward_in_step
        )

        self._implicit_backward = implicit_backward_in_step
        if init_dual_vars is not None:
            self._dual_vars = init_dual_vars
        else:
            self._dual_vars = torch.zeros(m, requires_grad=False, device=device)

    def add_constraint(self):
        """
        Allows to dynamically add constraints. Increments`m`, appends a zero tensor to the end of`_dual_vars`.
        """
        self.m += 1
        self._dual_vars = torch.cat(
            (
                self._dual_vars,
                torch.zeros(1, requires_grad=False, device=self._dual_vars.device),
            )
        )

    def _init_group(
        self,
        group,
        params,
        grads,
        l_term_grads,  # gradient of the lagrangian term, updated from parameter gradients during dual_step
        aug_term_grads,  # gradient of the regularization term, updated from parameter gradients during dual_step
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        smoothing,
    ):
        has_sparse_grad = False

        for p in group["params"]:
            state = self.state[p]

            params.append(p)

            # Lazy state initialization
            if len(state) == 0:
                state["smoothing"] = p.detach().clone()

                if not self._implicit_backward:
                    state["l_term_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["aug_term_grad"] = torch.zeros_like(
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

            grads.append(p.grad)

            if not self._implicit_backward:
                l_term_grads.append(state["l_term_grad"])
                aug_term_grads.append(state["aug_term_grad"])

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
            return self._dual_vars[i]

    def step(self, loss: torch.Tensor = None, constraints: Tensor = None):
        r"""Perform an update of the primal parameters (network weights & slack variables). To be called AFTER :func:`dual_step` in an iteration!

        Args:
            constraints (Tensor): A Tensor of estimates of values of **ALL** constraints; used for primal parameter update.
        """

        if self._implicit_backward:
            L = lagrangian(loss, constraints, self._dual_vars, self.rho)
            L.backward()
        
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
            lr = group["lr"]
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            mu = group["mu"]
            smoothing_beta = group["smoothing_beta"]

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

            self.modified_adam(params, grads, l_term_grads, aug_term_grads, smoothing, mu, smoothing_beta, beta1, beta2, eps, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, amsgrad)
        
        if self._implicit_backward:
            self.zero_grad()


    @_use_grad_for_differentiable
    def modified_adam(self, params, grads, l_term_grads, aug_term_grads, smoothing, mu, smoothing_beta, beta1, beta2, eps, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, amsgrad):
        for i, param in enumerate(params):
            if self._implicit_backward:
                G_i = grads[i].add_(param - smoothing[i], alpha=mu)
            else:
                G_i = torch.zeros_like(param, memory_format=torch.preserve_format)
                G_i.add_(grads[i]).add_(l_term_grads[i]).add_(
                        aug_term_grads[i], alpha=self.rho
                    ).add_(param - smoothing[i], alpha=mu)

                l_term_grads[i].zero_()
                aug_term_grads[i].zero_()

            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]
            step_t += 1

            exp_avg.lerp_(G_i, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(G_i, G_i, value=1 - beta2)

            smoothing[i].add_(smoothing[i], alpha=-smoothing_beta).add_(
                    param, alpha=smoothing_beta
                )

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


def lagrangian(loss: torch.Tensor, constraints: torch.Tensor, dual_vars: torch.Tensor, rho: float):
    return loss + dual_vars @ constraints + 0.5*rho*torch.square(torch.linalg.norm(constraints, ord=2))