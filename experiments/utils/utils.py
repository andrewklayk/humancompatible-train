import importlib
from itertools import combinations
import os
from typing import Callable, Iterable
from scipy.io.arff import loadarff
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from humancompatible.train.benchmark.constraints import FairnessConstraint
import torch


def run_summary_full_set(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    constraints: Iterable[FairnessConstraint],
    loss_fn: Callable,
    path: str = None,
    state_dicts: Iterable[dict] = None,
) -> pd.DataFrame:
    """Takes a path, reads all .pt files, evaluates models and constraints on the dataset provided;"""

    c_val_vec, c_grads_mat = [], []

    if state_dicts is not None:
        raise NotImplementedError

    full_eval = []
    dict_paths = []
    for file in os.listdir(path):
        if file.endswith(".pt"):
            dict_paths.append(file)

    for state_dict in dict_paths:
        # if state_dicts is Non
        state_dict = torch.load(os.path.join(path, file))
        model.load_state_dict(state_dict)
        val_dict = {}

        c_val_vec = []
        for i, c_i in enumerate(constraints):
            data_c = c_i.sample_dataset(N=np.inf)
            cv = c_i.eval(model, data_c)
            c_val_vec.append(cv)
            # cv.backward()
            # cg = net_grads_to_tensor(net, flatten=True, device=device)
            model.zero_grad()
            # c_grads_mat.append(cg)
        c_val_vec = torch.tensor(c_val_vec)
        # c_grads_mat = torch.stack(c_grads_mat)

        # TODO: why list here
        val_dict["c"] = [c_val_vec.detach().cpu().numpy()]
        # full_eval.loc[*index_to_save]["cg"] = [c_grads_mat.detach().cpu().numpy()]

        X_tensor, y_tensor = dataset.tensors
        outs = model(X_tensor)
        if y_tensor.ndim < outs.ndim:
            y_tensor = y_tensor.unsqueeze(1)
        loss = loss_fn(outs, y_tensor)
        # loss.backward()
        # fg = net_grads_to_tensor(net, flatten=True, device=device)
        # net.zero_grad()

        val_dict["f"] = loss.detach().cpu().numpy()

        full_eval.append(val_dict)

    return pd.DataFrame(full_eval)


def create_constraint_from_cfg(
    cfg: DictConfig,
    dataset: torch.utils.data.Dataset,
    group_indices: Iterable[Iterable[int]],
    loss_fn: Callable,
    device: str,
    seed: int = None,
) -> FairnessConstraint:
    constraint_fn_module = importlib.import_module(
        "humancompatible.train.benchmark.constraints"
    )
    constraint_fn = getattr(constraint_fn_module, cfg.constraint.import_name)
    if cfg.constraint.type == "one_vs_mean":
        c = [
            FairnessConstraint(
                dataset,
                [_group_ind, np.concat(group_indices)],
                fn=lambda net, inputs: constraint_fn(loss_fn, net, inputs)
                - cfg.constraint.bound,
                batch_size=cfg.constraint.c_batch_size,
                seed=seed,
            )
            for _group_ind in group_indices
        ]
        if cfg.constraint.add_negative:
            c.extend(
                [
                    FairnessConstraint(
                        dataset,
                        [group_ind, np.concat(group_indices)],
                        fn=lambda net, inputs: -constraint_fn(loss_fn, net, inputs)
                        - cfg.constraint.bound,
                        batch_size=cfg.constraint.c_batch_size,
                        seed=seed,
                    )
                    for group_ind in group_indices
                ]
            )
    elif cfg.constraint.type == "one_vs_each":
        c = [
            FairnessConstraint(
                dataset,
                _group_ind,
                fn=lambda net, inputs: constraint_fn(loss_fn, net, inputs)
                - cfg.constraint.bound,
                batch_size=cfg.constraint.c_batch_size,
                device=device,
                seed=seed,
            )
            for _group_ind in combinations(group_indices, 2)
        ]
        if cfg.constraint.add_negative:
            c.extend(
                [
                    FairnessConstraint(
                        dataset,
                        _group_ind,
                        fn=lambda net, inputs: -constraint_fn(loss_fn, net, inputs)
                        - cfg.constraint.bound,
                        batch_size=cfg.constraint.c_batch_size,
                        device=device,
                        seed=seed,
                    )
                    for _group_ind in combinations(group_indices, 2)
                ]
            )
    return c
