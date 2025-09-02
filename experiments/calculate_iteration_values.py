from copy import deepcopy
import importlib
from itertools import combinations
import os
import timeit
import warnings
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, tensor
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from humancompatible.train.fairness.constraints.constraint_fns import fairret_stat_equality
from utils.load_folktables import prepare_folktables_multattr
from utils.network import SimpleNet
from humancompatible.train.algorithms.utils import net_grads_to_tensor, net_params_to_tensor
from itertools import combinations
from humancompatible.train.fairness.constraints import FairnessConstraint



# @hydra.main(version_base=None, config_path="conf", config_name="experiment")
def run(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(OmegaConf.to_yaml(cfg))
    N_RUNS = cfg.n_runs
    FT_STATE = cfg.data.state
    FT_TASK = cfg.data.task
    DOWNLOAD_DATA = cfg.data.download
    DATA_PATH = cfg.data.path

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
        X_train,
        y_train,
        group_ind_train,
        group_onehot_train,
        sep_group_ind_train,
        X_test,
        y_test,
        group_ind_test,
        sep_group_ind_test,
        group_onehot_test,
        _
    ) = prepare_folktables_multattr(
        FT_TASK,
        state=FT_STATE.upper(),
        random_state=42,
        onehot=False,
        download=DOWNLOAD_DATA,
        path=DATA_PATH,
        sens_cols=cfg.data.sens_attr,
        binarize=cfg.data.binarize,
        stratify=False,
    )
    print('Groups:')
    print(len(group_ind_train))
    X_train_tensor = tensor(X_train, dtype=DTYPE)
    y_train_tensor = tensor(y_train, dtype=DTYPE)
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)

    print(f"Train data loaded: {(FT_TASK, FT_STATE)}")
    print(f"Data shape: {X_train_tensor.shape}")

    PATH = 

    ## prepare to save results ##

    if "save_name" in cfg["alg"].keys():
        alg_save_name = cfg.alg.save_name
    else:
        alg_save_name = cfg.alg.import_name

    saved_models_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "utils", "saved_models")
    )
    directory = os.path.join(
        saved_models_path, DATASET_NAME, CONSTRAINT, f"{BOUND:.0E}"
    )

    model_name = os.path.join(directory, f"{alg_save_name}_{BOUND}")

    if not os.path.exists(directory):
        os.makedirs(directory)

    ## run experiments ##

    histories = pd.read_csv(cfg.checkpoint_df_path)

    ####################################################
    ### CALCULATE STATS ON EVERY ALGORITHM ITERATION ###
    ####################################################

    loss_fn = nn.BCEWithLogitsLoss()
    constraint_fn_module = importlib.import_module("humancompatible.train.fairness.constraints")
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
    full_eval_test = pd.DataFrame(
        index=index, columns=["G", "f", "fg", "c", "cg"]
    ).sort_index()

    loss_fn = nn.BCEWithLogitsLoss()
    X_test_tensor = tensor(X_test, dtype=DTYPE).to(device)
    y_test_tensor = tensor(y_test, dtype=DTYPE).to(device)
    X_train_tensor = X_train_tensor.to(device=device)
    y_train_tensor = y_train_tensor.to(device=device)

    save_train = True
    save_test = True
    histories.dropna(subset=["w"], inplace=True)

    for exp_idx in range(N_RUNS):
        for alg_iteration in histories.loc[exp_idx, :].index:
            print(f"{exp_idx} | {alg_iteration}", end="\r")

            w = histories["w"].loc[exp_idx, alg_iteration]
            net.load_state_dict(w)
            net = net.to(device)
            if cfg.alg.import_name.lower() == "sslalm":
                x_t = net_params_to_tensor(net, flatten=True, copy=True)
                lambdas = histories["dual_ms"].loc[exp_idx, alg_iteration]
                z = histories["z"].loc[exp_idx, alg_iteration]
                params = {
                    "x_t": x_t,
                    "lambdas": lambdas,
                    "z": z,
                    "rho": cfg.alg.params.rho,
                    "mu": cfg.alg.params.mu,
                }

            if save_train:
                if cfg.constraint.import_name == 'abs_max_dev_from_overall_tpr':
                    data_c = [[
                        (X_train_tensor[g_idx], y_train_tensor[g_idx]) for g_idx in group_ind_train
                    ]]
                elif cfg.constraint.import_name in ['abs_diff_pr', 'abs_diff_tpr']:
                    data_c = [
                        (
                            (X_train_tensor[g_idx], y_train_tensor[g_idx]),
                            (X_train_tensor, y_train_tensor)
                        )
                        for g_idx in group_ind_train
                    ]
                else:
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


            if save_test:
                if cfg.constraint.import_name == 'abs_max_dev_from_overall_tpr':
                    data_c = [[
                        (X_test_tensor[g_idx], y_test_tensor[g_idx]) for g_idx in group_ind_test
                    ]]
                elif cfg.constraint.import_name in ['abs_diff_tpr', 'abs_diff_pr']:
                    data_c = [
                        (
                            (X_test_tensor[g_idx], y_test_tensor[g_idx]),
                            (X_test_tensor, y_test_tensor)
                        )
                        for g_idx in group_ind_test
                    ]
                else:
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

    fname = f"AFTER_{alg_save_name}_{DATASET_NAME}_{BOUND}"
    fext = ".csv"
    if save_train:
        fname_train = fname + "_train" + fext
        save_path = os.path.join(utils_path, fname_train)
        print(f"Saving to: {save_path}")
        full_eval_train.to_pickle(save_path)

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
        cv.backward()
        cg = net_grads_to_tensor(net, flatten=True, device=device)
        net.zero_grad()
        c_grads_mat.append(cg)
    c_val_vec = torch.tensor(c_val_vec)
    c_grads_mat = torch.stack(c_grads_mat)
    full_eval.loc[*index_to_save]["c"] = [c_val_vec.detach().cpu().numpy()]
    full_eval.loc[*index_to_save]["cg"] = [c_grads_mat.detach().cpu().numpy()]

    X_tensor, y_tensor = data_f
    outs = net(X_tensor)
    if y_tensor.ndim < outs.ndim:
        y_tensor = y_tensor.unsqueeze(1)
    loss = loss_fn(outs, y_tensor)
    loss.backward()
    fg = net_grads_to_tensor(net, flatten=True, device=device)
    net.zero_grad()

    full_eval.loc[*index_to_save]["f"] = loss.detach().cpu().numpy()
    full_eval.loc[*index_to_save]["fg"] = [fg.detach().cpu().numpy()]



if __name__ == "__main__":
    run()