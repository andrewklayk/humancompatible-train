import importlib
from itertools import combinations
import os
import timeit
import warnings
import hydra
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from omegaconf import DictConfig, OmegaConf
from torch import nn, tensor
from utils.load_dutch import prepare_dutch
from utils.network import SimpleNet
from humancompatible.train.benchmark.algorithms.utils import net_grads_to_tensor
from itertools import combinations
from humancompatible.train.benchmark.constraints import FairnessConstraint



@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def run(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(OmegaConf.to_yaml(cfg))
    N_RUNS = cfg.n_runs
    FT_STATE = 'dutch'
    FT_TASK = 'dutch'
    DOWNLOAD_DATA = cfg.data.download
    DATA_PATH = cfg.data.path
    
    if "constraint" in cfg.keys():
        CONSTRAINT = cfg.constraint.import_name
        LOSS_BOUND = cfg.constraint.bound
    else:
        CONSTRAINT = "unconstr"
        LOSS_BOUND = 0

    if cfg.device == "cpu":
        device = "cpu"
    elif cfg.alg == "ghost":
        device = "cpu"
        print("CUDA not supported for Stochastic Ghost")
    elif torch.cuda.is_available():
        device = "cuda"
        print("CUDA found")
    else:
        device = "cpu"
        print("CUDA not found")

    print(f"{device = }")
    torch.set_default_device(device)

    DTYPE = torch.float32

    ## load data ##

    torch.set_default_dtype(DTYPE)
    DATASET_NAME = FT_TASK + "_" + FT_STATE

    (
        (X_train, X_val, X_test),
        (y_train, y_val, y_test),
        (group_ind_train, group_ind_val, group_ind_test),
        _,
        _
    ) = prepare_dutch(
        random_state=None,
        onehot=False,
        stratify=True,
    )
    print(f'Train: {len(group_ind_train)} groups of size {[len(group) for group in group_ind_train]}')
    print(f'Val: {len(group_ind_val)} groups of size {[len(group) for group in group_ind_val]}')
    print(f'Test: {len(group_ind_test)} groups of size {[len(group) for group in group_ind_test]}')

    X_train_tensor = tensor(X_train, dtype=DTYPE)
    y_train_tensor = tensor(y_train, dtype=DTYPE)
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)

    print(f"Train data loaded: {(FT_TASK, FT_STATE)}")
    print(f"Data shape: {X_train_tensor.shape}")

    ## prepare to save results ##

    if "save_name" in cfg["alg"].keys():
        alg_save_name = cfg.alg.save_name
    else:
        alg_save_name = cfg.alg.import_name

    saved_models_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "utils", "saved_models")
    )
    directory = os.path.join(
        saved_models_path, DATASET_NAME, CONSTRAINT, f"{LOSS_BOUND:.0E}"
    )

    model_name = os.path.join(directory, f"{alg_save_name}_{LOSS_BOUND}")

    if not os.path.exists(directory):
        os.makedirs(directory)

    ## run experiments ##
    histories = []
    for EXP_IDX in range(1, N_RUNS+1):
        print(f'Start run {EXP_IDX}\n')

        ## define constraints ##
        loss_fn = nn.BCEWithLogitsLoss()
        constraint_fn_module = importlib.import_module("humancompatible.train.benchmark.constraints")
        constraint_fn = getattr(constraint_fn_module, cfg.constraint.import_name)
        
        if cfg.constraint.type == 'one_vs_mean':
            c = [
                FairnessConstraint(
                    train_ds,
                    [group_ind, np.concat(group_ind_train)],
                    fn=lambda net, inputs: constraint_fn(loss_fn, net, inputs) - cfg.constraint.bound,
                    batch_size=cfg.constraint.c_batch_size,
                    seed=EXP_IDX
                )
                for group_ind in group_ind_train
            ]
            if cfg.constraint.add_negative:
                c.extend(
                    [
                        FairnessConstraint(
                            train_ds,
                            [group_ind, np.concat(group_ind_train)],
                            fn=lambda net, inputs: -constraint_fn(loss_fn, net, inputs) - cfg.constraint.bound,
                            batch_size=cfg.constraint.c_batch_size,
                            seed=EXP_IDX
                        )
                        for group_ind in group_ind_train
                    ]
                )
        elif cfg.constraint.type == 'one_vs_each':
            c = [
                FairnessConstraint(
                    train_ds,
                    group_idx,
                    fn=lambda net, inputs: constraint_fn(loss_fn, net, inputs) - cfg.constraint.bound,
                    batch_size=cfg.constraint.c_batch_size,
                    device=device,
                    seed=EXP_IDX,
                )
                for group_idx in combinations(group_ind_train, 2)
            ]
            if cfg.constraint.add_negative:
                c.extend(
                    [
                        FairnessConstraint(
                            train_ds,
                            group_idx,
                            fn=lambda net, inputs: -constraint_fn(loss_fn, net, inputs) - cfg.constraint.bound,
                            batch_size=cfg.constraint.c_batch_size,
                            device=device,
                            seed=EXP_IDX,
                        )
                        for group_idx in combinations(group_ind_train, 2)
                    ]
                )

        torch.manual_seed(EXP_IDX)
        net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)
        model_path = model_name + f"_trial{EXP_IDX}.pt"

        optimizer_name = cfg.alg.import_name
        module = importlib.import_module("humancompatible.train.benchmark.algorithms")
        Optimizer = getattr(module, optimizer_name)
        optimizer = Optimizer(net, train_ds, loss_fn, c)
        # inconsequential backward pass cause first pass is very slow
        x = net.forward(X_train_tensor[0])
        x.backward()
        net.zero_grad()
        # train!
        history = optimizer.optimize(
            **cfg.alg.params,
            max_iter=cfg.run_maxiter,
            max_runtime=cfg.run_maxtime,
            device=device,
            # seed=EXP_IDX,
            verbose=True,
        )

        ## SAVE RESULTS ##
        params = pd.DataFrame(history["params"])
        values = pd.DataFrame(history["values"])
        t = pd.Series(history["time"], name="time")
        histories.append(values.join(params, how="outer").join(t, how="outer"))
        
        ## SAVE MODEL ##
        print(f"Model saved to: {model_path}")
        torch.save(net.state_dict(), model_path)
        print("")

    # Save DataFrames to CSV files
    c_name = cfg.constraint.import_name
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "utils", "exp_results", c_name)
    )
    if not os.path.exists(utils_path):
        os.makedirs(utils_path)
        
    if cfg.save_checkpoint_df:
        fname = f"{alg_save_name}_{DATASET_NAME}_{LOSS_BOUND}.csv"
        save_path = os.path.join(utils_path, fname)
        print(f"Saving to: {save_path}")
        histories = pd.concat(histories, keys=range(N_RUNS), names=["trial", "iteration"])
        histories.to_pickle(save_path)
        print("Saved!")
        
    ####################################################
    ### CALCULATE STATS ON EVERY ALGORITHM ITERATION ###
    ####################################################

    loss_fn = nn.BCEWithLogitsLoss()
    constraint_fn_module = importlib.import_module("humancompatible.train.benchmark.constraints")
    constraint_fn = getattr(constraint_fn_module, cfg.constraint.import_name)

    print("----")
    print("")

    exp_iter_indices = [
        histories.loc[exp_idx, :]
        .index.get_level_values("iteration")[histories.loc[exp_idx]["w"].notna()]
        .to_list()
        for exp_idx in histories.index.get_level_values("trial").unique()
    ]
    exp_maxiter = np.argmax([ind[-1] for ind in exp_iter_indices])
    longest_exp_indices = exp_iter_indices[exp_maxiter]
    longest_exp_indices.extend(
        [ei[-1] for ei in exp_iter_indices if ei[-1] not in longest_exp_indices]
    )
    longest_exp_indices = list(set(longest_exp_indices))
    longest_exp_indices.sort()

    index = pd.MultiIndex.from_product(
        [longest_exp_indices, range(N_RUNS)],
        names=("iteration", "trial"),
    )
    full_eval_train = pd.DataFrame(
        index=index, columns=["G", "f", "fg", "c", "cg"]
    ).sort_index()
    full_eval_val = full_eval_train.copy()
    full_eval_test = full_eval_train.copy()

    loss_fn = nn.BCEWithLogitsLoss()
    X_test_tensor = tensor(X_test, dtype=DTYPE).to(device)
    y_test_tensor = tensor(y_test, dtype=DTYPE).to(device)

    X_val_tensor = tensor(X_val, dtype=DTYPE).to(device)
    y_val_tensor = tensor(y_val, dtype=DTYPE).to(device)
    
    X_train_tensor = X_train_tensor.to(device=device)
    y_train_tensor = y_train_tensor.to(device=device)

    save_train = True
    save_val = True
    save_test = True
    histories.dropna(subset=["w"], inplace=True)
    with torch.inference_mode():
        for exp_idx in range(N_RUNS):
            for alg_iteration in histories.loc[exp_idx, :].index:
                print(f"{exp_idx} | {alg_iteration}", end="\r")

                w = histories["w"].loc[exp_idx, alg_iteration]
                net.load_state_dict(w)
                net = net.to(device)
                if save_train:
                    if cfg.constraint.type=="one_vs_mean":
                        data_c = [
                            (
                                (X_train_tensor[g_idx], y_train_tensor[g_idx]),
                                (X_train_tensor, y_train_tensor)
                            )
                            for g_idx in group_ind_train
                        ]
                    elif cfg.constraint.type=="one_vs_each":
                        data_c = [
                            (
                                (X_train_tensor[g_idx_1], y_train_tensor[g_idx_1]),
                                (X_train_tensor[g_idx_2], y_train_tensor[g_idx_2]),
                            )
                            for g_idx_1, g_idx_2 in combinations(group_ind_train, 2)
                        ]
                    calculate_iteration_values(
                        alg=cfg.alg.import_name,
                        full_eval=full_eval_train,
                        index_to_save=[alg_iteration, exp_idx],
                        c=c,
                        loss_fn=loss_fn,
                        data_f=[X_train_tensor, y_train_tensor],
                        data_c=data_c,
                        net=net,
                        device=device,
                        add_negative=cfg.constraint.add_negative,
                        **params,
                    )

                if save_val:
                    if cfg.constraint.type=="one_vs_mean":
                        data_c = [
                            (
                                (X_val_tensor[g_idx], y_val_tensor[g_idx]),
                                (X_val_tensor, y_val_tensor)
                            )
                            for g_idx in group_ind_val
                        ]
                    elif cfg.constraint.type=="one_vs_each":
                        data_c = [
                            (
                                (X_val_tensor[g_idx_1], y_val_tensor[g_idx_1]),
                                (X_val_tensor[g_idx_2], y_val_tensor[g_idx_2]),
                            )
                            for g_idx_1, g_idx_2 in combinations(group_ind_val, 2)
                        ]
                    calculate_iteration_values(
                        alg=cfg.alg.import_name,
                        full_eval=full_eval_val,
                        index_to_save=[alg_iteration, exp_idx],
                        c=c,
                        loss_fn=loss_fn,
                        data_f=[X_val_tensor, y_val_tensor],
                        data_c=data_c,
                        net=net,
                        device=device,
                        add_negative=cfg.constraint.add_negative,
                        **params,
                    )

                if save_test:
                    if cfg.constraint.type=="one_vs_mean":
                        data_c = [
                            (
                                (X_test_tensor[g_idx], y_test_tensor[g_idx]),
                                (X_test_tensor, y_test_tensor)
                            )
                            for g_idx in group_ind_test
                        ]
                    elif cfg.constraint.type=="one_vs_each":
                        data_c = [
                            (
                                (X_test_tensor[g_idx_1], y_test_tensor[g_idx_1]),
                                (X_test_tensor[g_idx_2], y_test_tensor[g_idx_2]),
                            )
                            for g_idx_1, g_idx_2 in combinations(group_ind_test, 2)
                        ]
                    calculate_iteration_values(
                        alg=cfg.alg.import_name,
                        full_eval=full_eval_test,
                        index_to_save=[alg_iteration, exp_idx],
                        c=c,
                        loss_fn=loss_fn,
                        data_f=[X_test_tensor, y_test_tensor],
                        data_c=data_c,
                        net=net,
                        device=device,
                        add_negative=cfg.constraint.add_negative,
                        **params,
                    )

                net.zero_grad()

    fname = f"AFTER_{alg_save_name}_{DATASET_NAME}_{LOSS_BOUND}"
    fext = ".csv"
    if save_train:
        fname_train = fname + "_train" + fext
        save_path = os.path.join(utils_path, fname_train)
        print(f"Saving to: {save_path}")
        full_eval_train.to_pickle(save_path)

    if save_val:
        fname_val = fname + "_val" + fext
        save_path = os.path.join(utils_path, fname_val)
        print(f"Saving to: {save_path}")
        full_eval_val.to_pickle(save_path)

    if save_test:
        fname_test = fname + "_test" + fext
        save_path = os.path.join(utils_path, fname_test)
        print(f"Saving to: {save_path}")
        full_eval_test.to_pickle(save_path)


# helper function to calculate relevant values on full dataset (e.g. constraint gradient, AL function, etc)
# used to calculate those values at different points during algorithms run
def calculate_iteration_values(
    alg,
    full_eval,
    index_to_save,
    c,
    loss_fn,
    data_f,
    data_c,
    net,
    device,
    add_negative,
    **params,
):
    c_val_vec, c_grads_mat = [], []

    for i, c_i in enumerate(c):
        cv = c_i.eval(net, data_c[i // 2 if add_negative else i])
        c_val_vec.append(cv)
        # cv.backward()
        # cg = net_grads_to_tensor(net, flatten=True, device=device)
        net.zero_grad()
        # c_grads_mat.append(cg)
    c_val_vec = torch.tensor(c_val_vec)
    # c_grads_mat = torch.stack(c_grads_mat)
    full_eval.loc[*index_to_save]["c"] = [c_val_vec.detach().cpu().numpy()]
    # full_eval.loc[*index_to_save]["cg"] = [c_grads_mat.detach().cpu().numpy()]

    X_tensor, y_tensor = data_f
    outs = net(X_tensor)
    if y_tensor.ndim < outs.ndim:
        y_tensor = y_tensor.unsqueeze(1)
    loss = loss_fn(outs, y_tensor)
    # loss.backward()
    # fg = net_grads_to_tensor(net, flatten=True, device=device)
    # net.zero_grad()

    full_eval.loc[*index_to_save]["f"] = loss.detach().cpu().numpy()
    # full_eval.loc[*index_to_save]["fg"] = [fg.detach().cpu().numpy()]


def sample_or_restart_iterloader(loader):
    try:
        item = next(loader)
        return item
    except StopIteration:
        loader._reset(loader)
        # loader.gen
        item = next(loader)
        return item


if __name__ == "__main__":
    run()