from typing import Iterable, Optional, Union
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
import humancompatible.train.optim.barrier as Barrier

class SSG_Barrier(Optimizer):
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
        barrier='quadratic_reciprocal', # barrier method used on the constraint
        obj_lr_infeas=5e-2, # primal lr when in infeasible region
        pi_0 = 10, # see PBM
        dual_mult_init = 0.9, # see PBM
        # learning rate decrease rule
        # lr_rule = "const",
        # constraint learning rate
        dual_lr: Union[
            float, Tensor
        ] = 5e-2,  # keep as tensor for different learning rates for different constraints in the future? idk
        *,
        differentiable: bool = False,
        beta: float = 0.5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps = 1e-8
        
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
            raise ValueError(
                f"Switching Subgradient does not support multiple constraints."
                "Consider taking the largest violation at each iteration."
            )
        if differentiable:
            raise NotImplementedError("SSw does not support differentiable")

        defaults = dict(
            lr=lr,
            dual_lr=dual_lr,
            differentiable=differentiable,
        )

        super().__init__(params, defaults)

        # set the barrier method
        if barrier == 'exponential':
            self.barrier = Barrier.exponential_penalty
        elif barrier == 'modified_log':
            self.barrier = Barrier.modified_log_barrier
        elif barrier == 'augmented_lagrangian':
            self.barrier = Barrier.augmented_lagrangian
        elif barrier == 'quadratic_logarithmic':
            self.barrier = Barrier.quadratic_logarithmic_penalty
        elif barrier == 'quadratic_reciprocal':
            self.barrier = Barrier.quadratic_reciprocal_penalty

        self.pi = pi_0
        self.pi_init = pi_0
        self.dual_mult_init = dual_mult_init
        self.iter = 0
        self.m = m
        self.lr = lr
        self.dual_lr = dual_lr
        self.obj_lr_infeas = obj_lr_infeas
        # self.lr_rule = lr_rule
        # self.dual_lr_rule = dual_lr_rule
        self.c_vals: list[Union[float, Tensor]] = []
        self.beta = beta    
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        # self.ctols = ctols
        # essentially, move everything here to self.state[param_group]
        # self.state[param_group]['smoothing_avg'] <= z for that param_group;
        # ...['grad'] <= grad w.r.t. that param_group
        # ...['G'] <= G w.r.t. that param_group // idk if necessary
        # ...['c_grad'][c_i] <= grad of ith constraint w.r.t. that group<w

    def _init_group(self, group, params, grads, c_grads, moments1, moments2):
        # SHOULDN'T calculate values, only set them from the state of the respective param_group
        # calculations only happen in step() (or rather in the func version of step)
        has_sparse_grad = False

        for p in group["params"]:
            state = self.state[p]

            params.append(p)

            # Lazy state initialization
            if len(state) == 0:
                state["c_grad"] = []
                state["step"] = 0

                # init the moments
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            # append the avgs
            grads.append(p.grad)
            c_grads.append(state.get("c_grad"))
            moments1.append(state.get("exp_avg"))
            moments2.append(state.get("exp_avg_sq"))

        return has_sparse_grad

    def __setstate__(self, state):
        super().__setstate__(state)

    def update_p(self):
        """
        Update the p paramater of the PBM. The strength of the violation vector. For bigger p, the smaller penalty, smaller p, larger penalty. 
        -> penalty grows over time.

        NOTE: this might not be a good idea. Is quite prone to outliers.
        """

        # self.pi = self.pi * self.pi_init * (self.dual_mult_init)**self.iter 
        self.pi = self.pi * self.dual_mult_init

        # safe guard
        if self.pi <= 0.01: 
            self.pi = 0.01
        if self.pi >= 20: 
            self.pi = 20

        # FUTURE: maybe once we are in the feasible, restart pi? 

    def dual_step(self, i: int, c_val: Tensor = None):
        r"""Save constraint gradient for weight update. To be called BEFORE :func:`step` in an iteration!

        Args:
            i (int): index of the constraint **(unused)**
            c_val (Tensor): an estimate of the value of the constraint at which the gradient was computed **(unused)**
        """

        # apply the barrier
        c_val = self.barrier(c_val)

        # compute the grad of the NN params 
        c_val.backward(retain_graph=True)

        if i > self.m:
            raise ValueError("SSw does not support multiple constraints.")

        # save constraint grad of the NN 
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            moments1: list[Tensor] = []
            moments2: list[Tensor] = []
            _ = self._init_group(group, params, grads, c_grads, moments1, moments2)

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

        # iterate 
        self.iter += 1

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            c_grads: list[Tensor] = []
            lr = group["lr"]
            moments1: list[Tensor] = []
            moments2: list[Tensor] = []
            _ = self._init_group(group, params, grads, c_grads, moments1, moments2)

            for i, param in enumerate(params):

                # choose whether to optimize constraints or the objective
                if update_con:
                    grad_cur = self.obj_lr_infeas * param.grad + c_grads[i][0] # optimize the constraint
                    # param.add_(grad_cur, alpha=-self.dual_lr) # also update the objective
                    lr = self.dual_lr
                else:
                    lr = self.lr
                    grad_cur = param.grad # in feasible region - just objective

                exp_avg = moments1[i]
                exp_avg_sq = moments2[i]
                t = self.iter

                # ----- Adam first/second moment updates -----
                exp_avg.mul_(self.beta1).add_(grad_cur, alpha=1 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad_cur, grad_cur, value=1 - self.beta2)

                # ----- Bias corrections -----
                bias_correction1 = 1 - self.beta1 ** t
                bias_correction2 = 1 - self.beta2 ** t

                # correct the moment bias
                corrected_moment1 = exp_avg / bias_correction1
                corrected_moment2 = exp_avg_sq.sqrt() / (bias_correction2**0.5)

                # ----- Parameter update -----
                param.addcdiv_(corrected_moment1, corrected_moment2 + self.eps, value=-lr)
                    

        # clear the grad
        for p in group["params"]:
            self.state[p]["c_grad"].clear()

