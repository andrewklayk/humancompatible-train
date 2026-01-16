import timeit
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import Iterable, Callable, Dict, Tuple

class StoppingTracker:
    def __init__(self, max_time, max_iters):
        self.max_time = max_time
        self.max_iters = max_iters
        self.start()

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
        fwd: Callable,
        eval: Callable,
        train_data: torch.utils.data.DataLoader | torch.utils.data.Dataset,
        eval_data: torch.utils.data.DataLoader | torch.utils.data.Dataset,
        optimizer: torch.optim.Optimizer,
        train_iter: Callable = None,
        constraint_names: Iterable = None,
        use_vanilla_torch: bool = False,
        verbose: bool = True
        # eval_samples: int = None # how many samples to take from each dataset in val_data 
    ):
        self.model = model
        self.optimizer = optimizer
        if not isinstance(train_data, torch.utils.data.DataLoader):
            raise ValueError("Expected a training DataLoader")
        else:
            self.train_data = train_data
        self.eval_data = eval_data
        self.history = []
        self.current_state = {}
        self.constraint_names = constraint_names
        self.fwd = fwd
        self.eval = eval
        self.train_iter = train_iter
        self._use_vanilla_torch = use_vanilla_torch
        self.verbose = verbose

    
    # add self.history as a property with the getter making it into a dataframe
    # def get_loss_numpy(self):
    #     df = pd.DataFrame(self.history)
        

    # def get_constraints_numpy(self):


    def torch_training_iter(self, batch):
        loss = self.fwd(self.model, batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def constrained_training_iter(self, batch):

        loss, constraints = self.fwd(self.model, batch)
        for i, constraint in enumerate(constraints):
            self.optimizer.dual_step(i, c_val=constraint)

        self.optimizer.step(loss)
        self.optimizer.zero_grad(set_to_none=True)
    
    @torch.inference_mode
    def eval_save_report(self, epoch, iters, total_iters, time):
        # save iteration info
        eval_dict = {
            "epoch": epoch,
            "iteration": iters,
            "total_iters": total_iters,
            "time": time
        }   



        # evaluate model on all eval datasets given in constructor
        for key, data in self.eval_data.items() if isinstance(self.eval_data, dict) else enumerate(self.eval_data):
            loss, constraints = self.eval(self.model, data)
            eval_dict[f"{key}_loss"] = loss
            eval_dict = eval_dict | {
                f"c_{j}_{key}": c for j, c in enumerate(constraints)
            }

        # save dual vars if optimizer has them
        if hasattr(self.optimizer, "_dual_vars"):
            eval_dict = eval_dict | {f"dual_{j}": l.detach().numpy().copy().item() for j, l in enumerate(self.optimizer._dual_vars)}
            
        self.history.append(eval_dict)
        if self.verbose:
            with np.printoptions(precision=4, suppress=True):
                print(" ".join([(f"{k} {v:.4f}"  if isinstance(v, float) else f"{k} {v:5}") for k, v in self.history[-1].items()]))

    def training_loop(
        self,
        epochs: int,
        max_iter: int,
        max_runtime: int,
        eval_every: str | int | None = None, # 'epoch' or number
        save_every: str | int | None = None, # 'epoch' or number; -1 to only save in the end
    ):
        
        tracker = StoppingTracker(max_time=max_runtime, max_iters=max_iter)

        for epoch in range(epochs):
            epoch_iters = 0
            i = tracker.total_iters
            if eval_every == 'epoch':
                tracker.pause()
                self.eval_save_report(epoch, epoch_iters, i, tracker.time)
                tracker.resume()

            for batch in self.train_data:
                epoch_iters += 1
                if not isinstance(eval_every, str) and i % eval_every == 0:
                    tracker.pause()
                    self.eval_save_report(epoch, epoch_iters, i, tracker.time)
                    tracker.resume()
                    
                if self.train_iter:
                    self.train_iter(optimizer=self.optimizer, model=self.model, batch=batch)
                elif self._use_vanilla_torch:
                    self.torch_training_iter(batch)
                else:
                    self.constrained_training_iter(batch)
                tracker.update_check()

    # def save_checkpoint(
    #         self,
    #         path: None,
    # ):
    #     save_checkpoint(
    #         checkpoint_dir,
    #         step,
    #         orig_model.state_dict(), # model parameters
    #         [opt.state_dict() for opt in optimizers], # optimizer states
    #         { # metadata saved as json
    #             "step": step,
    #             "val_bpb": val_bpb, # loss at last step
    #             "model_config": model_config_kwargs,
    #             "user_config": user_config, # inputs to the training script
    #             "device_batch_size": device_batch_size,
    #             "max_seq_len": max_seq_len,
    #             "dataloader_state_dict": dataloader_state_dict,
    #             "loop_state": { # all loop state (other than step) so that we can resume training
    #                 "min_val_bpb": min_val_bpb,
    #                 "smooth_train_loss": smooth_train_loss,
    #                 "total_training_time": total_training_time,
    #             },
    #         },