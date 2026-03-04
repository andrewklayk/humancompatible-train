import timeit
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import Iterable, Callable, Dict, Tuple

class TrainTracker:
    def __init__(self, max_time, max_iters):
        self.max_time = max_time
        self.max_iters = max_iters
        self.start()
        self.best_feasible_val_loss = np.inf
        self.no_feasible_improvement = 0

    # TODO: add early stopping to wrapper
    # def check_model_performance(self, loss, constraints, c_tol):
    #     if max(constraints) < c_tol and loss < self.best_feasible_val_loss:
    #         self.best_feasible_val_loss = loss
    #     elif 

    def start(self):
        self.start_time = timeit.default_timer()
        self.time = 0
        self.total_iters = 0
        self.pause_time = 0

    def pause(self):
        self._pause_start_time = timeit.default_timer()

    def resume(self):
        self.pause_time += timeit.default_timer() - self._pause_start_time

    def update(self):
        self.time = timeit.default_timer() - self.pause_time - self.start_time
        self.total_iters += 1

    def check(self):
        stats = {'time': self.time, 'iters': self.total_iters}
        if self.time > self.max_time or self.total_iters > self.max_iters:
            return (True, stats)
        else:
            return False
        
    def update_check(self):
        self.update()
        return self.check()


class OptimLoopWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        # fwd: Callable,
        eval: Callable,
        train_data: torch.utils.data.DataLoader | torch.utils.data.Dataset,
        eval_data: torch.utils.data.DataLoader | torch.utils.data.Dataset,
        primal_optimizer: torch.optim.Optimizer,
        dual_optimizer: torch.optim.Optimizer,
        train_save_interval: int = 1,
        fwd_c: Callable = None,
        train_iter: Callable = None,
        constraint_names: Iterable = None,
        verbose: bool = True,
        after_epoch_actions: Callable = None,
    ):
        """
        fwd_loss and fwd_c used to time loss and not constraints while still calculating constraints in the unconstrained case
        """
        self.model = model
        self.primal_opt = primal_optimizer
        self.dual_opt = dual_optimizer
        if not isinstance(train_data, torch.utils.data.DataLoader):
            raise ValueError("Expected a training DataLoader")
        else:
            self.train_data = train_data
        self.eval_data = eval_data
        self.train_history = []
        self.val_history = []
        self.current_state = {}
        self.train_save_interval = train_save_interval
        self.constraint_names = constraint_names
        if train_iter is not None:
            self.mode = None
            self.train_iter = train_iter
        elif fwd is not None and self.dual_opt is not None:
            self.mode = 'constrained'
            self.fwd = fwd
        else:
            self.mode = 'unconstrained'
            self.fwd = fwd
            self.fwd_c = fwd_c
        
        self.after_epoch_actions = after_epoch_actions
        self.eval = eval
        self.verbose = verbose

    # pre-packaged torch iteration
    def unconstrained_training_iter(self, batch):
        loss = self.fwd(self.model, batch)
        loss.backward()
        self.primal_opt.step()
        self.primal_opt.zero_grad()

        return loss

    # pre-packaged humancompatible.train iteration
    def constrained_training_iter(self, batch):

        loss, constraints = self.fwd(self.model, batch)
        lagrangian = self.dual_opt.forward_update(loss, constraints)
        lagrangian.backward()
        self.primal_opt.step()
        self.primal_opt.zero_grad()

        return loss, constraints
    

    def save_logs(self, mode, loss, constraints, other_stats, epoch, epoch_iters, total_iters, time):
        eval_dict = {
            "epoch": epoch,
            "iteration": epoch_iters,
            "total_iters": total_iters,
            "time": time,
            "loss": loss
        }
        
        if not isinstance(constraints, Iterable):
            constraints = [constraints]
        
        eval_dict = eval_dict | {
            f"c_{j}": c for j, c in enumerate(constraints)
        }

        eval_dict = eval_dict | (other_stats or {})
        
        # save dual vars if optimizer has them
        if self.dual_opt is not None and hasattr(self.dual_opt, "duals"):
            eval_dict = eval_dict | {f"dual_{j}": l.detach().numpy().copy().item() for j, l in enumerate(self.dual_opt.duals)}

        log = self.train_history if mode == 'train' else self.val_history
        log.append(eval_dict)


    def training_loop(
        self,
        epochs: int,
        max_iter: int,
        max_runtime: int,
        eval_every: str | int | None = None, # 'epoch' or number
        save_every: str | int | None = None, # 'epoch' or number; -1 to only save in the end
    ):
        
        tracker = TrainTracker(max_time=max_runtime, max_iters=max_iter)

        for epoch in range(epochs):
            epoch_iters = 0
            i = tracker.total_iters

            for batch in self.train_data:
                epoch_iters += 1
                tracker.update_check()
                save_train_this_iter = tracker.total_iters % self.train_save_interval == 0

                # save either after 1st iter OR every ... iters OR every epoch
                if tracker.total_iters > 1 and (tracker.total_iters == 2 or (isinstance(eval_every, str) and epoch_iters == 1) or (isinstance(eval_every, int) and i % eval_every == 0)):
                    tracker.pause()
                    loss, constraints, *other_stats = self.eval(self.model, self.eval_data)
                    other_stats = other_stats[0] if other_stats else None
                    self.save_logs('val', loss, constraints, other_stats, epoch, epoch_iters, tracker.total_iters, tracker.time)
                    if self.verbose:
                        with np.printoptions(precision=4, suppress=True):
                            print("TRAIN " + " ".join([(f"{k} {v:.4f}"  if isinstance(v, float) else f"{k} {v:5}") for k, v in self.train_history[-1].items()]))
                            print("VAL " + " ".join([(f"{k} {v:.4f}"  if isinstance(v, float) else f"{k} {v:5}") for k, v in self.val_history[-1].items()]))
                    tracker.resume()
                
                if self.mode == 'unconstrained':
                    # log constraints at iteration without timing them
                    constraints = None
                    if save_train_this_iter:
                        tracker.pause()
                        with torch.inference_mode():
                            constraints = self.fwd_c(self.model, batch)
                        tracker.resume()
                    loss = self.unconstrained_training_iter(batch)

                elif self.mode == 'constrained': # if standard constrained iteration
                    loss, constraints = self.constrained_training_iter(batch)

                elif self.train_iter: # if nonstandard iteration
                    loss, constraints = self.train_iter(self.primal_opt, self.dual_opt, model=self.model, batch=batch, timer=tracker)

                if save_train_this_iter:
                    self.save_logs('train', loss.detach().numpy().item(), constraints.detach().numpy(), None, epoch, epoch_iters, tracker.total_iters, tracker.time)
            
            if self.after_epoch_actions is not None:
                self.after_epoch_actions(self.model, self.primal_opt, self.dual_opt, epoch)
            if hasattr(self.dual_opt, "update_penalties"):
                self.dual_opt.update_penalties()
