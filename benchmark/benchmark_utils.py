### SETUP FOR BENCHMARK TASK - common elements for all optimizers
from typing import Callable, Tuple

from optim_loop_wrapper import OptimLoopWrapper, TrainTracker
from torch.nn import Sequential
import numpy as np
import torch
from humancompatible.train.dual_optim import MoreauEnvelope
from fairret.statistic import TruePositiveRate, FalsePositiveRate

seed_n = 1
torch.manual_seed(seed_n)
criterion = torch.nn.functional.binary_cross_entropy_with_logits


def separation(preds, sens, labels):
    if sens.shape[1] > 2:
        raise ValueError("Separation is not defined for more than 2 groups!")
    
    tpr = TruePositiveRate(preds, sens, labels)
    fpr = FalsePositiveRate(preds, sens, labels)

    return abs(tpr[0] - tpr[1]) + abs(fpr[0] - fpr[1])


def create_model(input_shape, latent_size1=64, latent_size2=32):
    model = Sequential(
        torch.nn.Linear(input_shape, latent_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size1, latent_size2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size2, 1),
    )

    return model


# forward functions: expects to have (model, batch) as arguments
def fwd_unconstrained(model, batch):
    batch_inputs, _, batch_labels = batch
    logits = model(batch_inputs)
    loss = criterion(logits, batch_labels)
    return loss

def fwd_constrained(model, batch, constraint_fn, bounds, eq = True, slack_vars = None, loss_constraint = False):
    if slack_vars is not None:
        with torch.no_grad():
            for s in slack_vars:
                if s < 0:
                    s.zero_()

    batch_inputs, batch_sens, batch_labels = batch
    logits = model(batch_inputs)
    if loss_constraint:
        loss = criterion(logits, batch_labels, reduction='none')
        constraints = constraint_fn(model, logits, batch_sens, batch_labels, loss)
        loss = torch.mean(loss)
    else:
        loss = criterion(logits, batch_labels)
        constraints = constraint_fn(model, logits, batch_sens, batch_labels)

    # into equality with slacks
    if eq and slack_vars is not None:
        constraints_bounded_eq = (constraints - torch.tensor(bounds) + slack_vars).to(torch.float)
    # into equality with max
    elif eq:
        constraints_bounded_eq = torch.max(constraints - torch.tensor(bounds), torch.zeros_like(constraints))
    # no equality, just subtract bound
    else:
        constraints_bounded_eq = (constraints - torch.tensor(bounds)).to(torch.float)

    return loss, constraints_bounded_eq

# one iteration of unconstrained optimization
def train_iter_torch(primal_opt, dual_opt, model, batch, tracker: TrainTracker, *, c_fn, p=None, fuse_loss_constraint = False):

    is_reg = not (p is None or torch.all(p == 0))
    batch_inputs, batch_sens, batch_labels = batch
    logits = model(batch_inputs)
    
    if fuse_loss_constraint:
        loss = criterion(logits, batch_labels, reduction="none")
        constraints = c_fn(model, logits, batch_sens, batch_labels, loss)
        loss = torch.mean(loss)
    else:
        loss = criterion(logits, batch_labels)
        if is_reg:
            constraints = c_fn(model, logits, batch_sens, batch_labels)
        else:
            ###### do NOT time constraints, calculate only for logging ######
            tracker.pause()
            with torch.inference_mode():
                constraints = c_fn(model, logits, batch_sens, batch_labels)
            tracker.resume()

    reg_term = p @ constraints if is_reg else 0

    _loss = criterion(logits, batch_labels) + reg_term
    _loss.backward()
    primal_opt.step()
    
    primal_opt.zero_grad()

    return loss, constraints

# one iteration of constrained optimization with PBM or ALM or other lagrangian-adjacent algorithm
def train_iter_hc(primal_opt, dual_opt, model, batch, tracker: TrainTracker, *, c_fn, bounds, eq = True, slack_vars = None, fuse_loss_constraint = False):
    loss, constraints = fwd_constrained(model, batch, c_fn, bounds, eq, slack_vars, fuse_loss_constraint)
    lagrangian = dual_opt.forward_update(loss, constraints)
    lagrangian.backward()
    primal_opt.step()

    primal_opt.zero_grad()

    return loss, constraints

# one iteration of (generalized) Switching Subgradient
def train_iter_sw(primal_opt, dual_opt, model, batch, tracker: TrainTracker, *, c_fn, bounds, constraint_tol):
    batch_inputs, batch_sens, batch_labels = batch
    logits = model(batch_inputs)
    # don't even need to evaluate loss if constraints are not satisfied
    constraints = c_fn(model, logits, batch_sens, batch_labels)
    constraint = max([con - bounds[i] for i, con in enumerate(constraints)])
    if constraint > constraint_tol:
        constraint.backward()
        dual_opt.step()
        #TODO what to do with loss in sw?
        tracker.pause()
        loss = criterion(logits, batch_labels)
        tracker.resume()
    else:
        loss = criterion(logits, batch_labels)
        loss.backward()
        primal_opt.step()

    model.zero_grad()

    return loss, constraints

# eval function that evaluates loss and constraints without keeping track of gradients 
# and without transforming constraints to equality 
@torch.inference_mode
def eval(model, batch, constraint_fn):
    inputs, sens, labels = batch
    logits = model(inputs)
    loss = criterion(logits, labels)
    constraints = constraint_fn(model, logits, sens, labels)
    constraints = [c.detach().numpy().item() for c in constraints]
    return loss.detach().numpy().item(), constraints


def run_train(
        m,
        primal_opt,
        dual_opt,
        param_set: dict,
        data_train: Tuple,
        dataloader: torch.utils.data.DataLoader,
        data_val: Tuple,
        n_epochs: int,
        c_fn: Callable,
        constraint_bound: float,
        mode: str,
        verbose: bool,
        reg_penalty: float | torch.Tensor = None,
        constraints_to_eq: bool = True,
        use_slack: bool = True,
        constraint_tol: float = 0.,
        fuse_loss_constraint: bool = False,
    ):
    
    (features_train, _, _) = data_train
    (features_val, sens_val, labels_val) = data_val

    model = create_model(features_train.shape[-1])

    primal_params = {k.removeprefix('primal__'): v for k, v in param_set.items() if k.startswith('primal__')}
    dual_params = {k.removeprefix('dual__'): v for k, v in param_set.items() if k.startswith('dual__')}
    moreau_params = {k.removeprefix('moreau__'): v for k, v in param_set.items() if k.startswith('moreau__')}
    # set up primal optimizer
    primal_optimizer = MoreauEnvelope(primal_opt(model.parameters(), **primal_params), **moreau_params)
    # set up slack variables if needed
    if use_slack:
        slack_vars = torch.zeros(m, requires_grad=True)
        primal_optimizer.add_param_group(param_group={"params": slack_vars, "name": "slack"})
    else:
        slack_vars = None

    if reg_penalty is not None and mode != 'torch':
        raise ValueError('Regularization only supported for vanilla PyTorch training. Do not pass`reg_penalty`when training with constrained algorithms.')

    bounds = [constraint_bound]*m

    if mode == 'torch':
        if isinstance(reg_penalty, float):
            reg_penalty = torch.ones(m) * reg_penalty
        train_iter = lambda primal, dual, model, batch, timer: train_iter_torch(primal, dual, model, batch, timer, c_fn=c_fn, p=reg_penalty, fuse_loss_constraint=fuse_loss_constraint)
        dual_optimizer = None
    elif mode == 'hc':
        train_iter = lambda primal, dual, model, batch, timer: train_iter_hc(primal, dual, model, batch, timer, c_fn=c_fn, bounds=bounds, eq=constraints_to_eq, slack_vars=slack_vars, fuse_loss_constraint=fuse_loss_constraint)
        dual_optimizer = dual_opt(m=m, **dual_params)
    elif mode == 'sw':
        train_iter = lambda primal, dual, model, batch, timer: train_iter_sw(primal, dual, model, batch, timer, c_fn=c_fn, bounds=bounds, constraint_tol=constraint_tol)
        dual_optimizer = dual_opt(model.parameters(), **dual_params)
    else:
        raise ValueError(f'Expected`mode`to be one of "torch", "hc", "sw"; got "{mode}"')

    # first backward pass sometimes takes ages, do it before training
    model(features_train[0]).backward()
    model.zero_grad()

    problem = OptimLoopWrapper(
            model = model,
            train_iter = train_iter,
            eval = lambda model, batch: eval(model, batch, c_fn),
            train_data = dataloader,
            eval_data = (features_val, sens_val, labels_val),
            primal_optimizer = primal_optimizer,
            dual_optimizer = dual_optimizer,
            verbose=verbose)
    
    problem.training_loop(epochs=n_epochs, max_iter=np.inf, max_runtime=np.inf, eval_every='epoch', save_every=-1)
    
    return problem.model, problem.train_history, problem.val_history


def run_grid(
        m,
        primal_opt,
        dual_opt,
        param_grid,
        n_epochs,
        constraint_fn,
        constraint_bound,
        dataloader,
        data_train,
        data_val,
        mode: str,
        verbose: bool,
        constraints_to_eq: bool = False,
        use_slack: bool = False,
        constraint_tol: float = 0.
    ):
    train_logs = []
    val_logs = []
    models = []
    if mode not in ['unconstrained', 'hc', 'sw']:
        raise ValueError(f"Expected`mode`to be one of (unconstrained, hc, sw), got {mode}")
    
    for i, param_set in enumerate(param_grid):
        print(f"starting {i+1}/{len(param_grid)}: {param_set}")

        model, train_history, val_history = run_train(
            m,
            primal_opt,
            dual_opt,
            param_set,
            data_train,
            dataloader,
            data_val,
            n_epochs,
            constraint_fn,
            constraint_bound,
            mode,
            verbose,
            constraints_to_eq,
            use_slack,
            constraint_tol
        )
        train_logs.append(train_history)
        val_logs.append(val_history)
        models.append(model)

    return models, train_logs, val_logs 