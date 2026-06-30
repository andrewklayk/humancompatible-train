"""The single training loop.

Replaces train_loop_primal_dual / train_loop_sw / train_loop_adam. The per-batch
update is delegated to ``algorithm.step``; everything else (epoch-0 eval pass,
metric logging, validation) is shared. Slack variables (``use_slack``, never
enabled in the benchmark) were dropped.
"""
import time

import numpy as np
import torch


def calc_constraints(constraint_fn, bounds, fuse, constraints_to_eq, model, out, sens, labels, loss):
    """Raw constraints and their bounded(/equality) form (c - bound)."""
    if fuse:
        constraints = constraint_fn(model, out, sens, labels, loss=loss)
    else:
        constraints = constraint_fn(model, out, sens, labels)
    if constraints_to_eq:
        constraints_bounded_eq = torch.max(constraints - bounds, torch.zeros_like(constraints))
    else:
        constraints_bounded_eq = constraints - bounds
    return constraints, constraints_bounded_eq


def calc_perclass_acc(group_onehot, labels, out):
    """Per-class/-group accuracy (faithful copy of the original helper)."""
    predictions = torch.argmax(out, dim=1)
    if labels.ndim > predictions.ndim:
        labels = labels.squeeze()
    correct = (predictions == labels).float()
    n_per = group_onehot.sum(dim=0)
    correct_per = group_onehot.T @ correct
    return (correct_per / (n_per + 1e-8)).detach().cpu().numpy()


def validate_model(model, val_data, loss_fn, constraint_fn, device):
    losses, constraints_list, acc_list = [], [], []
    with torch.no_grad():
        if isinstance(val_data, torch.utils.data.DataLoader):
            batches = val_data
        else:
            batches = [val_data]  # single full-dataset "batch"
        for feats, sens, labels in batches:
            feats = feats.to(device) if torch.is_tensor(feats) else feats
            sens = sens.to(device) if torch.is_tensor(sens) else sens
            labels = labels.to(device) if torch.is_tensor(labels) else labels
            out = model(feats)
            loss = loss_fn(out, labels)
            if loss.dim() > 0:
                loss = loss.mean()
            c = constraint_fn(model, out, sens, labels)
            acc = calc_perclass_acc(sens, labels, out)
            losses.append(loss.detach().cpu().numpy().item())
            constraints_list.append(c.detach().cpu().numpy())
            acc_list.append(acc)
    return np.mean(losses), np.mean(constraints_list, axis=0), np.mean(acc_list, axis=0)


def _epoch_record(epoch, train_time, losses, constraints, accs):
    return {
        "epoch": epoch,
        "time": train_time,
        "loss": np.mean(losses),
        "acc": np.mean(accs, axis=0),
    } | {f"c_{j}": c for j, c in enumerate(np.mean(constraints, axis=0))}


def train(model, algorithm, task, bundle, n_epochs, device):
    """Run training; returns (history_train, history_val, history_test) as lists of dicts."""
    model.to(device)
    bounds = torch.tensor([task.bound] * task.m).to(device)
    constraint_fn = task.constraint_fn
    loss_fn = task.loss_fn
    fuse = task.fuse_loss_constraint
    c_to_eq = algorithm.constraints_to_eq

    history_train, history_val, history_test = [], [], []

    for epoch in range(n_epochs + 1):
        losses, constraints, accs = [], [], []
        train_start = time.perf_counter()

        if epoch == 0:
            model.eval()
            for feats, sens, labels in bundle.train_loader:
                feats, sens, labels = feats.to(device), sens.to(device), labels.to(device)
                out = model(feats)
                loss = loss_fn(out, labels)
                c, _ = calc_constraints(constraint_fn, bounds, fuse, c_to_eq, model, out, sens, labels, loss)
                acc = calc_perclass_acc(sens, labels, out)
                losses.append((loss.mean() if loss.dim() > 0 else loss).detach().cpu().numpy().item())
                constraints.append(c.detach().cpu().numpy())
                accs.append(acc)
        else:
            model.train()
            for feats, sens, labels in bundle.train_loader:
                feats, sens, labels = feats.to(device), sens.to(device), labels.to(device)
                algorithm.zero_grad()
                out = model(feats)
                loss = loss_fn(out, labels)
                loss_for_c = loss if algorithm.passes_loss_to_constraints else None
                c, c_eq = calc_constraints(constraint_fn, bounds, fuse, c_to_eq, model, out, sens, labels, loss_for_c)
                loss_mean = loss.mean() if loss.dim() > 0 else loss
                algorithm.step(loss_mean, c_eq)
                losses.append(loss_mean.detach().cpu().numpy().item())
                constraints.append(c.detach().cpu().numpy())
                accs.append(acc := calc_perclass_acc(sens, labels, out))

        train_time = time.perf_counter() - train_start
        history_train.append(_epoch_record(epoch, train_time, losses, constraints, accs))

        if bundle.val is not None:
            model.eval()
            v_loss, v_c, v_acc = validate_model(model, bundle.val, loss_fn, constraint_fn, device)
            history_val.append({"epoch": epoch, "time": train_time, "loss": v_loss, "acc": v_acc}
                               | {f"c_{j}": c for j, c in enumerate(v_c)})

        if bundle.test is not None:
            model.eval()
            t_loss, t_c, t_acc = validate_model(model, bundle.test, loss_fn, constraint_fn, device)
            history_test.append({"epoch": epoch, "time": train_time, "loss": t_loss, "acc": t_acc}
                                | {f"c_{j}": c for j, c in enumerate(t_c)})

    return history_train, history_val, history_test
