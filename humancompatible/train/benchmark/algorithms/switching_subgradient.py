import timeit
from copy import deepcopy
from typing import Callable

import numpy as np
import torch

from humancompatible.train.algorithms.Algorithm import Algorithm
from humancompatible.train.algorithms.utils import net_params_to_tensor


class SSG(Algorithm):
    def __init__(
        self, net, data, loss, constraints, custom_project_fn: Callable = None
    ):
        super().__init__(net, data, loss, constraints)
        self.project = custom_project_fn if custom_project_fn else self.project_fn

    @staticmethod
    def project_fn(x, m):
        return x

    def optimize(
        self,
        ctol_rule,
        ctol,
        f_stepsize_rule,
        f_stepsize,
        c_stepsize_rule,
        c_stepsize,
        batch_size,
        epochs=None,
        save_iter=None,
        device="cpu",
        seed=None,
        verbose=True,
        max_runtime=None,
        max_iter=None,
        save_state_interval=100
    ):
        self.state_history = {}
        self.state_history["params"] = {"w": {}}
        self.state_history["values"] = {"G": {}, "f": {}, "c": {}}
        self.state_history["time"] = {}

        f_eta_t = f_stepsize
        c_eta_t = c_stepsize

        loss_eval = None
        c_t = None
        eta_f_sum = total_iters = iteration = f_iters = c_iters = epoch = 0
        eta_f_list = []
        _ctol = ctol
        if epochs is None:
            epochs = np.inf
        if max_iter is None:
            max_iter = np.inf

        gen = torch.Generator(device=device)
        loss_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, shuffle=True, generator=gen
        )
        loss_iter = iter(loss_loader)

        run_start = timeit.default_timer()
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

            try:
                f_sample = next(loss_iter)
            except StopIteration:
                epoch += 1
                iteration = 0
                loss_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size, shuffle=True, generator=gen
                )
                loss_iter = iter(loss_loader)
                f_sample = next(loss_iter)

            self.net.zero_grad()
            if ctol_rule == 'dimin':
                _ctol = ctol / np.sqrt(total_iters)

            if save_iter is not None and total_iters >= save_iter:
                eta_f_list.append(f_eta_t)
                eta_f_sum += f_eta_t

            # generate sample of constraints
            c_sample = [ci.sample_loader() for ci in self.constraints]
            # calc constraints and update multipliers (line 3)
            with torch.no_grad():
                c_t = np.array(
                    [
                        ci.eval(self.net, c_sample[i]).reshape(1)
                        for i, ci in enumerate(self.constraints)
                    ]
                ).flatten()
                c_argmax = np.argmax(c_t)
                c_max = c_t[c_argmax]

            x_t = net_params_to_tensor(self.net, flatten=True, copy=True)

            if c_max >= _ctol:
                c_iters += 1
                c_max2 = self.constraints[c_argmax].eval(self.net, c_sample[c_argmax]).reshape(1)

                c_grad = torch.autograd.grad(c_max2, self.net.parameters())
                c_grad = torch.concat([cg.flatten() for cg in c_grad])

                if c_stepsize_rule == "adaptive":
                    c_eta_t = c_max / (1e-6 + torch.norm(c_grad) ** 2)
                elif c_stepsize_rule == "const":
                    c_eta_t = c_stepsize
                elif c_stepsize_rule == "dimin":
                    c_eta_t = c_stepsize / np.sqrt(total_iters)

                x_t1 = self.project(x_t - c_eta_t * c_grad, m=len(self.constraints))

            else:
                f_iters += 1
                f_inputs, f_labels = f_sample
                outputs = self.net(f_inputs)
                if f_labels.dim() < outputs.dim():
                    f_labels = f_labels.unsqueeze(1)
                loss_eval = self.loss_fn(outputs, f_labels)

                f_grad = torch.autograd.grad(loss_eval, self.net.parameters())
                f_grad = torch.concat([fg.flatten() for fg in f_grad])

                if f_stepsize_rule == "dimin":
                    f_eta_t = f_stepsize / np.sqrt(total_iters)
                elif f_stepsize_rule == "const":
                    f_eta_t = f_stepsize
                x_t1 = self.project(x_t - f_eta_t * f_grad, m=len(self.constraints))

            start = 0
            with torch.no_grad():
                w = net_params_to_tensor(self.net, flatten=False, copy=False)
                for i in range(len(w)):
                    end = start + w[i].numel()
                    w[i].set_(x_t1[start:end].reshape(w[i].shape))
                    start = end

            if total_iters % save_state_interval == 0:
                if c_max is not None:
                    self.state_history["values"]["c"][total_iters] = (
                        c_t
                    )
                if loss_eval is not None:
                    self.state_history["values"]["f"][total_iters] = (
                        loss_eval.cpu().detach().numpy()
                    )

            if verbose and loss_eval is not None and c_t is not None:
                with np.printoptions(
                    precision=3,
                    suppress=True,
                    floatmode="fixed",
                    sign=" ",
                    linewidth=100,
                ):
                    print(
                        f"{epoch:2}|"
                        f"{_ctol:.3}|"
                        f"{iteration:5}|"
                        f"{loss_eval.detach().cpu().numpy():.5f}|"
                        f"{c_t}",
                        end="\r",
                    )

        ######################
        ### POSTPROCESSING ###
        ######################

        if save_iter is not None:
            model_ind = np.random.default_rng(seed=seed).choice(
                np.arange(start=save_iter, stop=total_iters),
                p=np.array(eta_f_list) / np.sum(eta_f_list),
            )
            self.net.load_state_dict(self.state_history["params"]["w"].iloc[model_ind])

        return self.state_history