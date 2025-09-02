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



@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def run(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(OmegaConf.to_yaml(cfg))
    N_RUNS = cfg.n_runs
    FT_STATE = cfg.data.state
    FT_TASK = cfg.data.task
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

    # experiment loop
    for EXP_IDX in range(N_RUNS):

        net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)

        ## define constraints ##

        criterion = nn.BCEWithLogitsLoss()
        constraint_fn_module = importlib.import_module("humancompatible.train.fairness.constraints")
        try:
            constraint_fn = getattr(constraint_fn_module, cfg.constraint.import_name)
        except:
            constraint_fn = getattr(importlib.import_module("humancompatible.train.fairness.constraints.torch"), "loss_equality")
        
        if cfg.constraint.import_name == 'abs_max_dev_from_overall_tpr':
            c = [FairnessConstraint(
                train_ds,
                group_ind_train,
                fn=lambda net, inputs: constraint_fn(criterion, net, inputs) - cfg.constraint.bound,
                batch_size=cfg.constraint.c_batch_size,
                seed=EXP_IDX
            )]
        elif cfg.constraint.import_name in ['abs_diff_tpr', 'abs_diff_pr']:
            c = [
                FairnessConstraint(
                    train_ds,
                    [group_ind, np.concat(group_ind_train)],
                    fn=lambda net, inputs: constraint_fn(criterion, net, inputs) - cfg.constraint.bound,
                    batch_size=cfg.constraint.c_batch_size,
                    seed=EXP_IDX
                )
                for group_ind in group_ind_train
            ]
        elif cfg.constraint.import_name in ['abs_diff_fpr', 'abs_diff_pr']:
            c = [
                FairnessConstraint(
                    train_ds,
                    [group_ind, np.concat(group_ind_train)],
                    fn=lambda net, inputs: constraint_fn(criterion, net, inputs) - cfg.constraint.bound,
                    batch_size=cfg.constraint.c_batch_size,
                    seed=EXP_IDX
                )
                for group_ind in group_ind_train
            ]
            constraint_fn1 = getattr(constraint_fn_module, 'abs_diff_tpr')
            c += [
                FairnessConstraint(
                    train_ds,
                    [group_ind, np.concat(group_ind_train)],
                    fn=lambda net, inputs: constraint_fn1(criterion, net, inputs) - cfg.constraint.bound,
                    batch_size=cfg.constraint.c_batch_size,
                    seed=EXP_IDX
                )
                for group_ind in group_ind_train
            ]
        else:
            c = construct_constraints(
                bound=cfg.constraint.bound,
                add_negative=cfg.constraint.add_negative,
                batch_size=cfg.constraint.c_batch_size,
                device=device,
                constraint_groups=[group_ind_train],
                dataset=train_ds,
                seed=EXP_IDX,
                constraint_fn=lambda net, inputs: constraint_fn(criterion, net, inputs),
                max_0 = False
            )
        # breakpoint()

        torch.manual_seed(EXP_IDX)
        model_path = model_name + f"_trial{EXP_IDX}.pt"

        net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)

        history = {
            "params": {"w": {}, "slack": {}, "dual_ms": {}},
            "values": {"f": {}, "c": {}, "G": {}},
            "time": {},
        }

        ## train ##

        if cfg.alg.import_name.startswith("fairret"):
            m = len(group_ind_train)

            _fairret_loss_module = importlib.import_module("fairret.loss")
            _fairret_statistic_module = importlib.import_module("fairret.statistic")
            fstat = getattr(_fairret_statistic_module, cfg.alg.params.statistic)()
            floss = getattr(_fairret_loss_module, cfg.alg.params.loss)(fstat, p=1)

            run_start = timeit.default_timer()

            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=cfg.alg.params.lr)
            c_batch_size = cfg.alg.params.c_batch_size
            obj_batch_size = cfg.alg.params.obj_batch_size
            mult = cfg.alg.params.pmult

            total_iters = 0
            gen = torch.Generator(device=device)
            gen.manual_seed(EXP_IDX)
            obj_loader = iter(
                torch.utils.data.DataLoader(
                    train_ds,
                    obj_batch_size,
                    shuffle=True,
                    generator=gen,
                    drop_last=True,
                )
            )

            constr_dataloaders = []
            for group_indices in group_ind_train:
                sampler = SubsetRandomSampler(group_indices, gen)
                constr_dataloaders.append(
                    iter(
                        DataLoader(
                            train_ds, c_batch_size, sampler=sampler, drop_last=True
                        )
                    )
                )

            epoch = 0
            iteration = 0
            total_iters = 0

            group_ind_onehot = torch.empty(m, (c_batch_size * m))
            for j in range(1, m):
                group_ind_onehot[j][c_batch_size * (j - 1) : c_batch_size * j] = (
                    torch.ones(c_batch_size)
                )
            group_ind_onehot = group_ind_onehot.T

            while True:
                elapsed = timeit.default_timer() - run_start
                iteration += 1
                total_iters += 1
                if (
                    cfg.run_maxiter is not None and total_iters >= cfg.run_maxiter
                ) or elapsed > cfg.run_maxtime:
                    break
                if total_iters % cfg.alg.params.save_state_interval == 0:
                    history["params"]["w"][total_iters] = deepcopy(net.state_dict())
                history["time"][total_iters] = elapsed

                net.zero_grad()

                inputs, labels = sample_or_restart_iterloader(obj_loader)
                outputs = net(inputs)
                loss_obj = criterion(outputs.squeeze(), labels)

                c_inputs, c_labels = [], []
                for j in range(m):
                    group_inputs, group_labels = sample_or_restart_iterloader(
                        constr_dataloaders[j]
                    )
                    c_inputs.append(group_inputs)
                    c_labels.append(group_labels)

                c_inputs = torch.concat(c_inputs)
                c_labels = torch.concat(c_labels)

                outputs_c = net(c_inputs).squeeze()
                loss_c = floss(
                    outputs_c.unsqueeze(1), group_ind_onehot, c_labels.unsqueeze(1)
                )

                loss = loss_obj + mult * loss_c

                loss.backward()
                optimizer.step()

                with np.printoptions(precision=6, suppress=True):
                    print(
                        f"{epoch:2} | {iteration:5} | {loss_obj.detach().cpu().numpy():.4} | {loss_c.detach().cpu().numpy():.4}",
                        end="\r",
                    )
        elif cfg.alg.import_name.startswith("TorchSSLALM"):
            epochs = 1000
            avg_epoch_c_log = []
            avg_epoch_loss_log = []
            m = len(list(combinations(group_ind_train, 2)))*2
            
            from fairret.statistic import TruePositiveRate, PositiveRate
            from fairret.loss import NormLoss

            # breakpoint()
            train_ds_sens = TensorDataset(
                X_train_tensor,
                group_onehot_train,
                y_train_tensor
                )

            
            slack_vars = torch.zeros(m, requires_grad=True)
            obj_batch_size = 16
            c_batch_size = cfg.constraint.c_batch_size

            from humancompatible.train.algorithms.torch import SSLALM
            optimizer = SSLALM(
                net.parameters(),
                lr=0.01,
                dual_lr=0.1,
                rho=1.0,
                mu=2.0,
                beta=0.5,
                m=m,
            )
            c_bound = torch.tensor([cfg.constraint.bound]*m)
            optimizer.add_param_group(param_group={"params": slack_vars, "name": "slack"})
            # constr = FalseNegativeFalsePositiveFraction()
            # constr = PositiveRate()
            constr = constraint_fn
            # fair_loss = NormLoss(constr)

            time = timeit.default_timer()
            total_iters = 0
            c_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            
            for epoch in range(epochs):
                elapsed = timeit.default_timer()
                if elapsed - time > cfg.run_maxtime:
                    break
                loss_log = []
                c_log = []
                gen = torch.Generator(device=device)
                gen.manual_seed(EXP_IDX + epoch)
                from humancompatible.train.fairness.utils import BalancedBatchSampler
                
                sampler = BalancedBatchSampler(
                    # subgroup_indices=group_ind_train,
                    subgroup_onehot=group_onehot_train,
                    batch_size=c_batch_size,
                    drop_last=True
                )
                dataloader = DataLoader(
                    train_ds_sens,
                    batch_sampler=sampler
                )
                # c_loader = iter(dataloader)
                for batch_input, batch_sens, batch_label in dataloader:
                    elapsed = timeit.default_timer()
                    if elapsed - time > cfg.run_maxtime:
                        break
                    history["time"][total_iters] = elapsed - time

                    # evaluate constraints and constraint grads
                    c_vals = []
                    c_vals_raw = []
                    
                    c_inputs = batch_input[::2]
                    c_labels = batch_label[::2]
                    c_sens = batch_sens[::2]
                    c_sens_norm = c_sens.div(torch.sum(c_sens, axis=0))

                    # calculate loss for each group
                    c_loss = c_criterion(net(c_inputs).squeeze(), c_labels) @ c_sens_norm
                    c_val_raw_vec = []
                    for (l1, l2) in combinations(c_loss, 2):
                        c_val_raw_vec.append(l1-l2)
                        c_val_raw_vec.append(l2-l1)


                    # c_outputs = torch.nn.functional.sigmoid(net(c_inputs))
                    # c_outputs_pos_idx = (c_outputs >= 0).squeeze()
                    # c_overall = constr(c_outputs[c_outputs_pos_idx], None
                    #                    , c_labels[c_outputs_pos_idx].unsqueeze(1)
                    #                    )
                    # c_val_raw_vec = constr(c_outputs[c_outputs_pos_idx], c_sens[c_outputs_pos_idx]
                    #                        , c_labels[c_outputs_pos_idx].unsqueeze(1)
                    #                        )
                    # c_val_raw_vec = torch.abs(c_val_raw_vec - c_overall)

                    for i in range(m):
                        optimizer.zero_grad()
                        c_val = c_val_raw_vec[i] + slack_vars[i] - c_bound[i]
                        # retain_graph in all but last iteration to calculate grads
                        c_val.backward(retain_graph = i < m-1)
                        optimizer.dual_step(i, c_val)
                    
                        c_vals.append(c_val.detach())
                        c_vals_raw.append(c_val_raw_vec[i].detach())
                    

                    optimizer.zero_grad()
                    # evaluate loss and loss grad
                    logits = net(batch_input)
                    loss = criterion(logits.squeeze(), batch_label) + torch.zeros_like(slack_vars) @ slack_vars # SLACK
                    loss.backward()

                    if cfg.alg.params.use_unbiased_penalty_grad:
                        with torch.no_grad():
                            c_inputs = batch_input[1::2]
                            c_labels = batch_label[1::2]
                            c_sens = batch_sens[1::2]
                            c_sens_norm = c_sens.div(torch.sum(c_sens, axis=0))
                            c_loss = c_criterion(net(c_inputs).squeeze(), c_labels) @ c_sens_norm
                            c_val_raw_vec = []
                            for (l1, l2) in combinations(c_loss, 2):
                                c_val_raw_vec.append(l1-l2)
                                c_val_raw_vec.append(l2-l1)
                                
                            # c_vals = []
                            # c_vals_raw = []
                            # c_inputs = batch_input[1::2]
                            # c_labels = batch_label[1::2]
                            # c_sens = batch_sens[1::2]
                            # c_outputs = torch.nn.functional.sigmoid(net(c_inputs))
                            # c_outputs_pos_idx = (c_outputs >= 0).squeeze()
                            # c_overall = constr(c_outputs[c_outputs_pos_idx], None
                            #                    ,c_labels[c_outputs_pos_idx].unsqueeze(1)
                            #                 )
                            # c_val_raw_vec = constr(c_outputs[c_outputs_pos_idx], c_sens[c_outputs_pos_idx]
                            #                        , c_labels[c_outputs_pos_idx].unsqueeze(1)
                            #                        )
                            # c_val_raw_vec = torch.abs(c_val_raw_vec - c_overall)
                            # c_val = c_val_raw_vec + slack_vars - c_bound
                            # c_vals.append(c_val.detach())

                            # c_vals_raw = c_val_raw_vec.detach()

                    optimizer.step(c_vals)
                    optimizer.zero_grad()
                    if total_iters % cfg.alg.params.save_state_interval == 0:
                        history["params"]["w"][total_iters] = deepcopy(net.state_dict())
                        history["time"][total_iters] = elapsed - time
                    
                    total_iters += 1

                    with torch.no_grad():
                        for s in slack_vars:
                            if s < 0:
                                s.zero_()

                    loss_log.append(loss.item())
                    c_log.append([c.item() for c in c_vals_raw])

                # print(optimizer._dual_vars)
                avg_epoch_loss_log.append(np.mean(loss_log))
                avg_epoch_c_log.append(np.mean(c_log, axis=0))
                with np.printoptions(precision=4):
                    print(
                        f"Epoch: {epoch}, loss: {avg_epoch_loss_log[-1]}, constraints: {avg_epoch_c_log[-1]}, dual: {optimizer._dual_vars.detach().numpy()}"
                    )
        elif cfg.alg.import_name.startswith("TorchSSG"):
            epochs = 100000
            avg_epoch_c_log = []
            avg_epoch_loss_log = []
            m = 1
            
            from fairret.statistic import PositiveRate

            # breakpoint()
            train_ds_sens = TensorDataset(X_train_tensor, group_onehot_train, y_train_tensor)

            obj_batch_size = 16
            c_batch_size = cfg.constraint.c_batch_size

            from humancompatible.train.algorithms.torch import SSG
            optimizer = SSG(
                net.parameters(),
                lr=0.05,
                dual_lr=0.05,
                m=m,
            )
            c_bound = torch.tensor([cfg.constraint.bound]*5)
            c_tol = torch.tensor([cfg.constraint.bound*2]*5)

            time = timeit.default_timer()
            total_iters = 0
            
            for epoch in range(epochs):
                elapsed = timeit.default_timer()
                if elapsed - time > cfg.run_maxtime:
                    break
                loss_log = []
                c_log = []
                gen = torch.Generator(device=device)
                gen.manual_seed(EXP_IDX + epoch)
                obj_loader = iter(
                    torch.utils.data.DataLoader(
                        train_ds,
                        obj_batch_size,
                        shuffle=True,
                        generator=gen,
                    )
                )

                gen = torch.Generator(device=device)
                gen.manual_seed(EXP_IDX + epoch)
                from humancompatible.train.fairness.utils import BalancedBatchSampler
                
                sampler = BalancedBatchSampler(
                    subgroup_indices=group_ind_train,
                    batch_size=c_batch_size,
                    drop_last=True)
                dataloader = DataLoader(train_ds_sens, batch_sampler=sampler)
                max_idx_log = []
                for batch_input, batch_sens, batch_label in dataloader:
                    elapsed = timeit.default_timer()
                    if elapsed - time > cfg.run_maxtime:
                        break
                    history["time"][total_iters] = elapsed - time

                    # evaluate constraints and largest constraint grad
                    c_vals = []
                    c_vals_raw = []
                    
                    c_inputs = batch_input
                    c_sens = batch_sens
                    c_labels = batch_label
                    c_outputs = torch.nn.functional.sigmoid(net(c_inputs))

                    constr = PositiveRate()
                    c_overall = constr(c_outputs, None)
                    c_val_raw_vec = constr(c_outputs, c_sens)
                    # breakpoint()
                    c_val_raw_vec = torch.abs(c_val_raw_vec - c_overall)

                    c_val = c_val_raw_vec - c_bound
                    c_max_viol_idx = torch.argmax(c_val - c_tol)
                    c_max_viol = c_val[c_max_viol_idx]
                    c_max_viol.backward()
                    max_idx_log.append(c_max_viol_idx)

                    optimizer.dual_step(i=0)
                    
                    c_vals = c_max_viol
                    c_vals_raw.append(c_val_raw_vec.detach())

                    optimizer.zero_grad()
                    # evaluate loss and loss grad
                    logits = net(batch_input)
                    loss = criterion(logits.squeeze(), batch_label)
                    loss.backward()

                    optimizer.step(c_vals)
                    optimizer.zero_grad()

                    total_iters += 1
                    c_tol /= np.sqrt(total_iters)

                    if total_iters % cfg.alg.params.save_state_interval == 0:
                        history["params"]["w"][total_iters] = deepcopy(net.state_dict())
                    
                    
                    loss_log.append(loss.item())
                    c_log.append([
                        c_val.detach()
                    ])

                # print(optimizer._dual_vars)
                avg_epoch_loss_log.append(np.mean(loss_log))
                avg_epoch_c_log.append(np.mean(c_log, axis=0))
                print(
                    f"Epoch: {epoch}, loss: {avg_epoch_loss_log[-1]}, constraints: {avg_epoch_c_log[-1]}"
                )  
        elif cfg.alg.import_name.startswith("SGD"):
            
            optimizer = torch.optim.Adam(params=net.parameters())

            time = timeit.default_timer()
            total_iters = 0
            train_ds_sens = TensorDataset(X_train_tensor, group_onehot_train, y_train_tensor)
            epochs = 1000000
            avg_epoch_loss_log = []
            
            for epoch in range(epochs):
                elapsed = timeit.default_timer()
                if elapsed - time > cfg.run_maxtime:
                    break
                loss_log = []
                gen = torch.Generator(device=device)
                gen.manual_seed(EXP_IDX + epoch)

                gen = torch.Generator(device=device)
                gen.manual_seed(EXP_IDX + epoch)
                from humancompatible.train.fairness.utils import BalancedBatchSampler
                
                sampler = BalancedBatchSampler(
                    subgroup_indices=group_ind_train,
                    batch_size=cfg.constraint.c_batch_size,
                    drop_last=True)
                dataloader = DataLoader(train_ds_sens, batch_sampler=sampler)

                for batch_input, batch_sens, batch_label in dataloader:
                    elapsed = timeit.default_timer()
                    if elapsed - time > cfg.run_maxtime:
                        break
                    history["time"][total_iters] = elapsed - time
                    logits = net(batch_input)
                    loss = criterion(logits.squeeze(), batch_label)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    if total_iters % cfg.alg.params.save_state_interval == 0:
                        history["params"]["w"][total_iters] = deepcopy(net.state_dict())
                    
                    total_iters += 1
                    
                    loss_log.append(loss.item())
                    
                avg_epoch_loss_log.append(np.mean(loss_log))
                
                print(
                    f"Epoch: {epoch}, loss: {avg_epoch_loss_log[-1]}"
                )  
        else:
            optimizer_name = cfg.alg.import_name
            module = importlib.import_module("humancompatible.train.algorithms")
            Optimizer = getattr(module, optimizer_name)

            optimizer = Optimizer(net, train_ds, criterion, c)
            history = optimizer.optimize(
                **cfg.alg.params,
                max_iter=cfg.run_maxiter,
                max_runtime=cfg.run_maxtime,
                device=device,
                seed=EXP_IDX,
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
    if cfg.alg.import_name.lower() == "sgd":
        c_name = "unconstrained"
    else:
        c_name = cfg.constraint.import_name
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "utils", "exp_results", c_name)
    )
    if not os.path.exists(utils_path):
        os.makedirs(utils_path)
    fname = f"{alg_save_name}_{DATASET_NAME}_{LOSS_BOUND}.csv"
    save_path = os.path.join(utils_path, fname)
    print(f"Saving to: {save_path}")
    histories = pd.concat(histories, keys=range(N_RUNS), names=["trial", "iteration"])
    histories.to_pickle(save_path)
    print("Saved!")

    ####################################################
    ### CALCULATE STATS ON EVERY ALGORITHM ITERATION ###
    ####################################################

    criterion = nn.BCEWithLogitsLoss()
    # constraint_fn_module = importlib.import_module("humancompatible.train.fairness.constraints")
    # constraint_fn = getattr(constraint_fn_module, cfg.constraint.import_name)
    # if cfg.constraint.import_name != 'abs_max_dev_from_overall_tpr':
    #     c = construct_constraints(
    #         bound=cfg.constraint.bound,
    #         add_negative=cfg.constraint.add_negative,
    #         batch_size=cfg.constraint.c_batch_size,
    #         device=device,
    #         constraint_groups=[group_ind_train],
    #         dataset=train_ds,
    #         seed=EXP_IDX,
    #         constraint_fn=lambda net, inputs: constraint_fn(loss_fn, net, inputs),
    #         max_0 = False
    #     )
    # else:
    #     c = [FairnessConstraint(
    #         train_ds,
    #         group_ind_train,
    #         fn=lambda net, inputs: constraint_fn(loss_fn, net, inputs) - cfg.constraint.bound,
    #         batch_size=cfg.constraint.c_batch_size // len(group_ind_train),
    #         seed=EXP_IDX
    #     )]

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

    criterion = nn.BCEWithLogitsLoss()
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
                    loss_fn=criterion,
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
                    loss_fn=criterion,
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

    if save_test:
        fname_test = fname + "_test" + fext
        save_path = os.path.join(utils_path, fname_test)
        print(f"Saving to: {save_path}")
        full_eval_test.to_pickle(save_path)


# helper function to construct pairwise constraints for every combination of provided groups
def construct_constraints(
    constraint_fn,
    bound,
    dataset,
    constraint_groups,
    batch_size,
    add_negative,
    device,
    seed,
    max_0 = False
):
    c = []

    for group_indices in constraint_groups:
        for group_idx in combinations(group_indices, 2):
            c1 = FairnessConstraint(
                dataset,
                group_idx,
                fn=lambda net, d: torch.max(constraint_fn(net, d) - bound, torch.zeros(1)) if max_0 else constraint_fn(net, d) - bound,
                batch_size=batch_size // 2,
                device=device,
                seed=seed,
            )
            c.append(c1)

            if add_negative:
                c2 = FairnessConstraint(
                    dataset,
                    group_idx,
                    fn=lambda net, d: torch.max(-constraint_fn(net, d) - bound, torch.zeros(1)) if max_0 else -constraint_fn(net, d) - bound,
                    batch_size=batch_size // 2,
                    device=device,
                    seed=seed,
                )
                c.append(c2)

    return c

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

    # if alg.lower() == "sgd":
    #     return

    # elif alg.lower() == "sslalm":
    #     x_t, z, rho, mu, lambdas = (
    #         params["x_t"],
    #         params["z"],
    #         params["rho"],
    #         params["mu"],
    #         params["lambdas"],
    #     )
    #     G = (
    #         fg
    #         + c_grads_mat.T @ lambdas
    #         + rho * (c_grads_mat.T @ c_val_vec)
    #         + mu * (x_t - z)
    #     )

    #     full_eval.loc[*index_to_save]["G"] = [G.detach().cpu().numpy()]


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