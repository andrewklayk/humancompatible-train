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
        tau=0.01,
        eta=0.05,
        lambda_bound=25.,
        rho=1.,
        mu=2.,
        beta=0.5,
        tau_mult=1.,
        eta_mult=1.,
        batch_size=16,
        epochs=None,
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
        # self.history['vars_full'] = {'G': {}, 'f': {}, 'fg': {}, 'c': {}, 'cg': {}}
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
        z_par = torch.narrow(z, 0, 0, z.shape[-1] - m)

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
            gen = gen.manual_seed(seed)
        loss_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, shuffle=True, generator=gen
        )
        loss_iter = iter(loss_loader)

        epoch = 0
        iteration = 0
        total_iters = 0

        ### initial f and f_grad estimate ###
        f_grad_estimate = 0
        pre_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, shuffle=True, generator=gen
        )
        pre_iter = iter(pre_loader)
        (f_inputs, f_labels) = next(pre_iter)
        _, f_grad_estimate = self._objective_estimate(f_inputs, f_labels)
        self.net.zero_grad()

        ### initial c_val and c_grad estimate ###
        c_sample = [ci.sample_dataset(batch_size) for ci in c]
        _c_val_estimate = self._c_value_estimate(slack_vars, c, c_sample)
        c_val_estimate = torch.concat(_c_val_estimate)
        c_grad_estimate = self._constraint_grad_estimate(slack_vars, _c_val_estimate)

        ### c_val estimate ###
        if use_unbiased_penalty_grad:
            c_sample = [ci.sample_dataset(batch_size) for ci in c]
            c_val_estimate_2 = torch.concat(self._c_value_estimate(slack_vars, c, c_sample))
        else:
            c_val_estimate_2 = c_val_estimate

        n_iters_c_satisfied = 0
        percent_iters_c_satisfied = 0

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

                percent_iters_c_satisfied = n_iters_c_satisfied / total_iters

            try:
                (f_inputs, f_labels) = next(loss_iter)
            except StopIteration:
                epoch += 1
                iteration = 0
                gen = gen
                loss_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size, shuffle=True, generator=gen
                )
                loss_iter = iter(loss_loader)
                (f_inputs, f_labels) = next(loss_iter)
                tau *= tau_mult
                eta *= eta_mult
                # rho *= rho_mult

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
            )

            if mu > 0:
                smoothing = mu * (x_t - z)
                G += smoothing

            x_t1 = self.project(x_t - tau * G, m)

            if mu > 0:
                z += beta * (x_t - z)

            ###################
            ## UPDATE PARAMS ##
            ###################

            with torch.no_grad():
                _set_weights(self.net, x_t1)
                for i in range(len(slack_vars)):
                    slack_vars[i] = x_t1[i - len(slack_vars)]
            # objective gradient
            loss_eval, f_grad_1 = self._objective_estimate(f_inputs, f_labels)
            self.net.zero_grad()

            # constraint value abd grad (1)
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

            if torch.all(c_val_1 <= 0):
                n_iters_c_satisfied += 1

            if verbose:
                with np.printoptions(
                    precision=3,
                    suppress=True,
                    floatmode="fixed",
                    sign=" ",
                    linewidth=200,
                ):
                    print(
                        f"{epoch:2}|{iteration:5}|{tau:.3f}|"
                        # f"{loss_eval.detach().cpu().numpy():1.3f}|"
                        f"{_lambda.detach().cpu().numpy()}|"
                        f"{c_val_estimate.detach().cpu().numpy() - slack_vars.detach().cpu().numpy()}|",
                        # f"{slack_vars.detach().cpu().numpy()} | {100*percent_iters_c_satisfied:2.1f}%",
                        end="\r",
                    )

        return self.state_history



    def _c_value_estimate(self, slack_vars, c, c_sample):
        c_val = [
            ci.eval(self.net, c_sample[i]).reshape(1) + slack_vars[i]
            for i, ci in enumerate(c)
        ]

        return c_val

    def _objective_estimate(self, f_inputs, f_labels):
        m = len(self.constraints)
        # breakpoint()
        outputs = self.net(f_inputs)
        # if f_labels.dim() < outputs.dim():
        #     f_labels = f_labels.unsqueeze(1)
        loss_eval = self.loss_fn(outputs.squeeze(), f_labels)
        f_grad = torch.autograd.grad(loss_eval, self.net.parameters())
        f_grad = torch.concat([*[g.flatten() for g in f_grad], torch.zeros(m)])

        return loss_eval, f_grad

    def _constraint_grad_estimate(self, slack_vars, c):
        c_grad = []
        for ci in c:
            ci_grad = torch.autograd.grad(ci, self.net.parameters())
            if slack_vars is None:
                c_grad.append(torch.concat([g.flatten() for g in ci_grad]))
            else:
                slack_grad = torch.autograd.grad(ci, slack_vars, materialize_grads=True)
                # if torch.sum(slack_grad[0]) != 1:
                #     breakpoint()
                c_grad.append(
                    torch.concat([*[g.flatten() for g in ci_grad], *slack_grad])
                )
                slack_vars.grad = None
                # slack_vars.zero_grad_

            self.net.zero_grad()
        c_grad = torch.stack(c_grad)
        return c_grad