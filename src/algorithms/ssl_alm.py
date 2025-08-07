import timeit
from copy import deepcopy
from typing import Callable

import numpy as np
import torch

from src.algorithms.Algorithm import Algorithm
from src.algorithms.utils import _set_weights, net_params_to_tensor


class SSLALM(Algorithm):
    def __init__(
        self, net, data, loss, constraints, custom_project_fn: Callable = None
    ):
        super().__init__(net, data, loss, constraints)
        self.project = custom_project_fn if custom_project_fn else self.project_fn

    @staticmethod
    def project_fn(x, m):
        for i in range(1, m + 1):
            if x[-i] < 0:
                x[-i] = 0
        return x

    def optimize(
        self,
        lambda_bound,
        eta,
        rho,
        tau,
        mu,
        vr_mult_obj,
        vr_mult_cval,
        vr_mult_cgrad,
        beta,
        batch_size,
        epochs,
        start_lambda=None,
        max_runtime=None,
        max_iter=None,
        seed=None,
        device="cpu",
        verbose=True,
        use_unbiased_penalty_grad=True,
        save_state_interval=1
    ):
        self.state_history = {}
        self.state_history["params"] = {"w": {}, "dual_ms": {}, "z": {}, "slack": {}}
        self.state_history["values"] = {"G": {}, "f": {}, "fg": {}, "c": {}, "cg": {}}
        self.state_history["time"] = {}
        
        m = len(self.constraints)
        slack_vars = torch.zeros(m, requires_grad=True)
        _lambda = (
            torch.zeros(m, requires_grad=True) if start_lambda is None else start_lambda
        )

        z = torch.concat(
            [net_params_to_tensor(self.net, flatten=True, copy=True), slack_vars]
        )

        c = self.constraints

        run_start = timeit.default_timer()

        if epochs is None:
            epochs = np.inf
        if max_iter is None:
            max_iter = np.inf
        if max_runtime is None:
            max_runtime = np.inf

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        loss_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, shuffle=True, generator=gen
        )
        loss_iter = iter(loss_loader)

        epoch = 0
        iteration = 0
        total_iters = 0

        f_grad_estimate = 0

        while True:
            elapsed = timeit.default_timer() - run_start
            iteration += 1
            total_iters += 1
            if epoch >= epochs or total_iters >= max_iter or elapsed > max_runtime:
                break

            self.state_history["time"][total_iters] = elapsed
            if total_iters % save_state_interval == 0:
                self.state_history["params"]["w"][total_iters] = deepcopy(
                    self.net.state_dict()
                )
                self.state_history["params"]["dual_ms"][total_iters] = (
                    _lambda.detach().cpu().numpy()
                )
                self.state_history["params"]["z"][total_iters] = (
                    z_par.detach().cpu().numpy()
                )
                self.state_history["params"]["slack"][total_iters] = (
                    slack_vars.detach().cpu().numpy()
                )

            try:
                (f_inputs, f_labels) = next(loss_iter)
            except StopIteration:
                epoch += 1
                iteration = 0
                loss_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size, shuffle=True, generator=gen
                )
                loss_iter = iter(loss_loader)
                (f_inputs, f_labels) = next(loss_iter)

            ########################
            ## UPDATE MULTIPLIERS ##
            ########################
            self.net.zero_grad()
            slack_vars.grad = None

            # sample for and calculate self.constraints (lines 2, 3)
            # update multipliers (line 3)
            with torch.no_grad():
                _lambda = _lambda + eta * c_val_estimate
            # dual safeguard (lines 4,5)
            if torch.norm(_lambda) >= lambda_bound:
                _lambda = torch.zeros_like(_lambda, requires_grad=True)

            x_t = torch.concat(
                [
                    net_params_to_tensor(self.net, flatten=True, copy=True),
                    slack_vars,
                ]
            )

            G = (
                f_grad_estimate
                + c_grad_estimate.T @ _lambda
                + rho * (c_grad_estimate.T @ c_val_estimate_2)
                + mu * (x_t - z)
            )
            x_t1 = self.project(x_t - tau * G, m)
            z += beta * (x_t - z)
            
            #### UPDATE NETWORK WEIGHTS ####
            with torch.no_grad():
                _set_weights(self.net, x_t1)
                for i in range(len(slack_vars)):
                    slack_vars[i] = x_t1[i - len(slack_vars)]
            # objective gradient
            loss_eval, f_grad_1 = self._objective_estimate(f_inputs, f_labels)
            self.net.zero_grad()

            c_sample = [ci.sample_loader() for ci in c]
            _c_val_1 = self._c_value_estimate(slack_vars, c, c_sample)
            c_val_1 = torch.concat(_c_val_1)
            c_grad_1 = self._constraint_grad_estimate(slack_vars, _c_val_1)

            # constraint value (2) (independent)
            if use_unbiased_penalty_grad:
                c_sample = [ci.sample_loader() for ci in c]
                c_val_2 = torch.concat(self._c_value_estimate(slack_vars, c, c_sample))
            else:
                c_val_2 = c_val_1

            f_grad_estimate = f_grad_1
            c_val_estimate = c_val_1
            c_val_estimate_2 = c_val_2
            c_grad_estimate = c_grad_1
            
            with torch.no_grad():
                f_grad_par = torch.narrow(
                    f_grad_estimate, 0, 0, f_grad_estimate.shape[-1] - m
                )
                c_grad_par = torch.narrow(
                    c_grad_estimate, 1, 0, c_grad_estimate.shape[-1] - m
                )
                G_par = torch.narrow(G, 0, 0, G.shape[-1] - m)
                z_par = torch.narrow(z, 0, 0, z.shape[-1] - m)

                self.state_history["values"]["G"][total_iters] = (
                    torch.norm(G_par).detach().cpu().numpy()
                )
                self.state_history["values"]["f"][total_iters] = (
                    loss_eval.detach().cpu().numpy()
                )
                self.state_history["values"]["fg"][total_iters] = (
                    torch.norm(f_grad_par).detach().cpu().numpy()
                )
                self.state_history["values"]["c"][total_iters] = (
                    c_val_2.detach().cpu().numpy()
                )
                self.state_history["values"]["cg"][total_iters] = (
                    torch.norm(c_grad_par, dim=1).detach().cpu().numpy()
                )

            if verbose:
                with np.printoptions(precision=6, suppress=True, floatmode="fixed"):
                    print(
                        f"""{epoch:2}|{iteration:5} | {tau} | {loss_eval.detach().cpu().numpy()}|{_lambda.detach().cpu().numpy()}|{c_val_estimate.detach().cpu().numpy()}|{slack_vars.detach().cpu().numpy()}""",
                        end="\r",
                    )

        ######################
        ### POSTPROCESSING ###
        ######################

        G_hat = torch.zeros_like(G)

        f_inputs, f_labels = self.dataset[:][0], self.dataset[:][1]
        cgrad_sample = [ci.sample_dataset(np.inf) for ci in c]
        c_sample = [ci.sample_dataset(np.inf) for ci in c]

        self.net.zero_grad()
        slack_vars.grad = None

        _, f_grad = self._objective_estimate(f_inputs, f_labels)
        self.net.zero_grad()
        # constraint grad estimate
        c_1 = [
            ci.eval(self.net, c_sample[i]).reshape(1) + slack_vars[i]
            for i, ci in enumerate(c)
        ]
        c_grad = self._constraint_grad_estimate(slack_vars, c_1)

        # independent constraint estimate
        with torch.no_grad():
            c_val_estimate = torch.concat(
                [
                    ci.eval(self.net, cgrad_sample[i]).reshape(1) + slack_vars[i]
                    for i, ci in enumerate(c)
                ]
            )
        x_t = torch.concat(
            [net_params_to_tensor(self.net, flatten=True, copy=True), slack_vars]
        )
        G_hat += f_grad + c_grad.T @ _lambda + rho * (c_grad.T @ c_val_estimate) + mu * (x_t - z)

        x_t1 = self.project(x_t - tau * G_hat, m)
        with torch.no_grad():
            _set_weights(self.net, x_t1)

        current_time = timeit.default_timer()
        self.history["w"].append(deepcopy(self.net.state_dict()))
        self.history["time"].append(current_time - run_start)
        self.history["n_samples"].append(batch_size * 3)

        return self.history

    def _c_value_estimate(self, slack_vars, c, c_sample):
        c_val = [
                ci.eval(self.net, c_sample[i]).reshape(1) + slack_vars[i]
                for i, ci in enumerate(c)
            ]
        
        return c_val

    def _objective_estimate(self, f_inputs, f_labels):
        m = len(self.constraints)
        outputs = self.net(f_inputs)
        if f_labels.dim() < outputs.dim():
            f_labels = f_labels.unsqueeze(1)
        loss_eval = self.loss_fn(outputs, f_labels)
        f_grad = torch.autograd.grad(loss_eval, self.net.parameters())
        f_grad = torch.concat([*[g.flatten() for g in f_grad], torch.zeros(m)])

        return loss_eval, f_grad

    def _constraint_grad_estimate(self, slack_vars, c):
        c_grad = []
        for ci in c:
            ci_grad = torch.autograd.grad(ci, self.net.parameters())
            slack_grad = torch.autograd.grad(ci, slack_vars)
            c_grad.append(torch.concat([*[g.flatten() for g in ci_grad], *slack_grad]))
            self.net.zero_grad()
            slack_vars.grad = None
        c_grad = torch.stack(c_grad)
        return c_grad
