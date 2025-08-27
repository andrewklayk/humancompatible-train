import timeit
from copy import deepcopy

import numpy as np
import torch

from humancompatible.train.algorithms.Algorithm import Algorithm


class SGD(Algorithm):
    def __init__(self, net, data, loss, constraints):
        super().__init__(net, data, loss, constraints)

    def optimize(
        self,
        lr,
        batch_size,
        epochs=None,
        max_runtime=None,
        max_iter=None,
        seed=None,
        device="cpu",
        verbose=True,
        save_state_interval=1000,
    ):
        self.state_history = {}
        self.state_history["params"] = {"w": {}}
        self.state_history["values"] = {"f": {}, "fg": {}}
        self.state_history["time"] = {}

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

        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

        while True:
            elapsed = timeit.default_timer() - run_start
            iteration += 1
            total_iters += 1
            if epoch >= epochs or total_iters >= max_iter or elapsed > max_runtime:
                break

            if total_iters % save_state_interval == 0:
                self.state_history["params"]["w"][total_iters] = deepcopy(
                    self.net.state_dict()
                )
                self.state_history["time"][total_iters] = elapsed

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
                # lr  *= 0.8

            ########################
            ## UPDATE MULTIPLIERS ##
            ########################
            self.net.zero_grad()
            outputs = self.net(f_inputs)
            loss = self.loss_fn(outputs.squeeze(), f_labels)
            loss.backward()
            optimizer.step()

            # f_grad_estimate =

            with torch.no_grad():
                if total_iters % save_state_interval == 0:
                    self.state_history["values"]["f"][total_iters] = (
                        loss.detach().cpu().numpy()
                    )
                # self.state_history['values']['fg'][total_iters] = torch.norm(f_grad_estimate).detach().cpu().numpy()

            if verbose:
                with np.printoptions(
                    precision=8, suppress=True, floatmode="fixed", sign=" "
                ):
                    print(
                        f"""{epoch:2}|{iteration:5} | {lr} | {loss.detach().cpu().numpy():1.5f}""",
                        end="\r",
                    )

        return self.state_history
