from typing import Iterable, Optional, Union
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
import humancompatible.train.optim.barrier as Barrier

"""
This module implements the PBM optimizer.
https://www.researchgate.net/publication/2775406_PenaltyBarrier_Multiplier_Methods_for_Convex_Programming_Problems

"""

class PBM(Optimizer):
    def __init__(
        self,
        params,
        m: int,
        # tau in paper
        lr: Union[float, Tensor] = 5e-2,
        # penalty term multiplier
        rho: float = 1.0,
        # smoothing term multiplier
        mu: float = 2.0,
        # smoothing term update multiplier
        beta: float = 0.5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        dual_beta: float = 0.9, # smoothing of the dual variables
        init_pi = 10.0,
        const_p = 1.0,
        epoch_len=None,
        penalty_update_m='CONST', # p parameter
        p_lb = 0.1,
        dual_bounds = (1e-4, 10),
        warm_start = 0, # number of epochs to warm start - no constraints included
        device="cpu",
        eps: float = 1e-8,
        amsgrad: bool = False,
        *,
        init_dual = 0.1,
        differentiable: bool = False,
        barrier='quadratic_logarithmic', # barrier method used on the constraint
    ):
        
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

        if differentiable:
            raise NotImplementedError("TorchSSLALM does not support differentiable")

        defaults = dict(
            lr=lr,
            rho=rho,
            mu=mu,
            beta=beta,
            amsgrad=amsgrad,
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

        if (penalty_update_m == "DIMINISH" or warm_start) and epoch_len is None:
            raise ValueError(f"Diminishing penalty update requires epoch_len to be defined")
        self.epoch_len = epoch_len

        # set the optimizer parameters
        self.m = m
        self.device = device
        self.rho = rho
        self.dual_bounds = dual_bounds
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.dual_beta = dual_beta
        self.init_pi = init_pi
        self.mu = mu
        self.c_vals: list[Union[float, Tensor]] = []
        self._c_val_average = [None]
        self.eps = eps
        self.warm_start = warm_start
        self.iter = 0
        self.init_dual = init_dual
        self._dual_vars = torch.ones(m, requires_grad=False, device=device) * init_dual
        self.p = torch.ones(m, requires_grad=False, device=device)
        self.constraints = torch.zeros(m, device=device) # array of current constraint values

        # set the defined penalty update method
        if penalty_update_m == 'ALM':
            self.update_p_method = self.update_p_ALM

        elif penalty_update_m == "CONST":
            self.update_p_method = self.update_p_const
            self.p_const = const_p
            self.p = torch.ones(m, requires_grad=False, device=device) * self.p_const

        elif penalty_update_m == "DIMINISH":
            self.update_p_method = self.update_p_diminishing
            self.p_const = const_p
            self.p = torch.ones(m, requires_grad=False, device=device) * self.p_const
            self.p_lb = p_lb

        else: 
            raise ValueError("NO SUCH p UPDATE METHOD EXISTS!")
                
    def add_constraint(self):
        """
        Allows to dynamically add constraints. Increments`m`, appends a zero tensor to the end of`_dual_vars`.
        """
        self.m += 1
        self._dual_vars = torch.cat(
            (
                self._dual_vars,
                torch.ones(1, requires_grad=False, device=self._dual_vars.device) * self.init_dual,
            )
        )

    def _init_group(
        self,
        group,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        smoothing
    ):
        has_sparse_grad = False

        for p in group["params"]:
            state = self.state[p]
            params.append(p)

            # Lazy state initialization
            if len(state) == 0:
                state["smoothing"] = p.detach().clone()
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

            state_steps.append(state["step"])
            grads.append(p.grad)
            smoothing.append(state.get("smoothing"))

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

    # --------------------------------------------------- p update functions -------------------------------------

    def update_p_ALM(self, i, t):
        """
        update the penalty coefficient - equiv. to ALM
        """

        self.p[i] = self.rho * self._dual_vars[i]


    def update_p_paper(self, i, t):
        """
        update the penalty coefficient - equiv. to ALM
        """

        self.p[i] =  t * self.rho * (self.init_pi)**self.iter
        pass

    def update_p_diminishing(self, i, t):

        # diminishing update once per defined number of steps
        if self.iter % self.epoch_len == 0:
            self.p[i] *= 0.9

        # threshold the p
        if self.p[i] < self.p_lb:
            self.p[i] = self.p_lb

    def update_p_const(self, i, t):
        """
        Constant p - do nothing
        """

        pass

    # -----------------------------------------------------------------------------------------------------------


    def dual_step(self, i: int, c_val: Tensor):
        r"""Perform an update of the dual parameters.
        Also saved the constraint value for later use in the weights update. To be called BEFORE :func:`step` in an iteration!

        Args:
            i (int): index of the constraint
            c_val (Tensor): an estimate of the value of the constraint at which the gradient was computed; used for dual parameter update
        """

        # check for incorrect input
        if c_val.numel() != 1:
            raise ValueError(
                f"`dual_step` expected a scalar `c_val`, got an object of shape {c_val.shape}"
            )
        
        # check for warm start - no condition for n epochs
        if self.warm_start > 0 and self.iter // self.epoch_len < self.warm_start:
            return # do nothing on duals before the warm start

        # --------------------------------

        # sub variable for computing grad wrt to the input to the penalty/ barrier
        t = torch.tensor(c_val.detach().item() , dtype=torch.float32, requires_grad=True, device=self._dual_vars.device)
        t = t / self.p[i]

        # compute the grad wrt. to the sub_var
        penalty_barrier_val = self.barrier(t)
        dloss_dt = torch.autograd.grad(penalty_barrier_val, t, retain_graph=True)[0]        

        # update dual variables 
        # self._dual_vars[i] = self._dual_vars[i] * dloss_dt # hard update
        self._dual_vars[i] = self.dual_beta * self._dual_vars[i]  +  (1 - self.dual_beta) * self._dual_vars[i] * dloss_dt # soft update
        
        # update penalty multiplier
        self.update_p_method(i, self._dual_vars[i])

        # safe-guarding 
        if self._dual_vars[i] <= self.dual_bounds[0]:
            self._dual_vars[i] = self.dual_bounds[0]
        if self._dual_vars[i] >= self.dual_bounds[1]:
            self._dual_vars[i] = self.dual_bounds[1]

        # compute the barrier/penalty of the output of NN  
        phi_constr = self.p[i].item() * self.barrier(c_val / self.p[i].item())

        # save the costraint value - later will be used in 
        self.constraints[i] = phi_constr

    @_use_grad_for_differentiable
    def step(self, loss_v: Tensor):
        r"""Perform an update of the primal parameters (network weights & slack variables). To be called AFTER :func:`dual_step` in an iteration!

        Args:
            loss_v (Tensor): Loss funciton value
        """

        with torch.enable_grad():
            
            if self.warm_start > 0 and self.iter // self.epoch_len < self.warm_start: # warmstart - just the objective
                F_loss = loss_v
            else: 
                # define the augmented F and backpropagate
                F_loss = loss_v + self._dual_vars @ self.constraints
        
            # backpropagate
            F_loss.backward()
        
        # load the smoothing and the momentums
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            smoothing: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            lr = group["lr"]
            amsgrad = group["amsgrad"]

            _ = self._init_group(
                group,
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                smoothing
            )

            # no need to track gradients at this point
            # update the NN params 
            for i, param in enumerate(params):
                
                # grads is the sum of gradient of the obj + linear comb. of the constraints + add the smoothing term
                G_i = torch.zeros_like(param)

                # check for warm start - no condition for n epochs
                if self.warm_start > 0 and self.iter // self.epoch_len < self.warm_start:
                    G_i.add_(grads[i])     # no ALM - just objective 
                else: 
                    G_i.add_(grads[i]).add_(param - smoothing[i], alpha=self.mu)

                # G_i.add_(grads[i]).add_(l_term_grads[i]).add_(param - smoothing[i], alpha=self.mu)
                # G_i.add_(grads[i]).add_(l_term_grads[i]) # objective + lagrangian part
                # G_i.add_(grads[i])     # no ALM - just objective 

                # update the smooting term
                # smoothing[i].add_(smoothing[i], alpha=-self.beta).add_(
                #     param, alpha=self.beta
                # )

                # # compute the adam moments
                # exp_avg = exp_avgs[i]
                # exp_avg_sq = exp_avg_sqs[i]
                # step_t = state_steps[i]
                # step_t += 1
                # beta1, beta2 = self.beta1, self.beta2
                # eps = self.eps

                # # moment1, moment2
                # exp_avg.lerp_(G_i, 1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(G_i, G_i, value=1 - beta2)

                # # bias correction of adam
                # bias_correction1 = 1 - beta1**step_t
                # bias_correction2 = 1 - beta2**step_t

                # # compute the bias corrected moments
                # step_size = lr / bias_correction1
                # bias_correction2_sqrt = bias_correction2**0.5

                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.maximum(
                #         max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i]
                #     )

                #     # Use the max. for normalizing running avg. of gradient
                #     denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(
                #         eps
                #     )
                # else:
                #     denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                # # adam step
                # param.addcdiv_(exp_avg, denom, value=-step_size)
                param.add_(G_i, alpha=-lr)

            # update p
            self.iter += 1

        # clean the gradients
        self.zero_grad()
        self.constraints = torch.zeros(self.m, device=self.device)


