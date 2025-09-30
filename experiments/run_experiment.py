import importlib
import os
import warnings
import hydra
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from omegaconf import DictConfig, OmegaConf
from torch import nn, tensor
from utils.load_folktables import prepare_folktables_multattr
from utils.network import SimpleNet
from utils.utils import create_constraint_from_cfg, run_summary_full_set
from humancompatible.train.benchmark.algorithms.utils import net_grads_to_tensor
from humancompatible.train.benchmark.constraints import FairnessConstraint


@hydra.main(version_base=None, config_path="conf", config_name="experiment")
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

    torch.set_default_dtype(DTYPE)
    (
        (X_train, X_val, X_test),
        (y_train, y_val, y_test),
        (group_ind_train, group_ind_val, group_ind_test),
        _,
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
        stratify=cfg.data.stratify,
    )

    print(f'Train: {len(group_ind_train)} groups of size {[len(group) for group in group_ind_train]}')
    print(f'Val: {len(group_ind_val)} groups of size {[len(group) for group in group_ind_val]}')
    print(f'Test: {len(group_ind_test)} groups of size {[len(group) for group in group_ind_test]}')

    X_train_tensor = tensor(X_train, dtype=DTYPE)
    y_train_tensor = tensor(y_train, dtype=DTYPE)
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)

    print(f"Train data loaded: {(FT_TASK, FT_STATE)}")
    print(f"Data shape: {X_train_tensor.shape}")

    #TODO change to actual current date time lol
    EXPERIMENT_NAME = 'CURRENT_DATE_TIME'
    exp_save_path = os.path.join(os.path.dirname(__file__), EXPERIMENT_NAME)
    os.makedirs(exp_save_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(exp_save_path, 'config.yaml'))

    for RUN_IDX in range(1, N_RUNS+1):

        print(f'Start run {RUN_IDX}\n')
        
        ## prepare files ##
        run_save_path = os.path.abspath(
            os.path.join(exp_save_path, str(RUN_IDX))
        )
        if not os.path.exists(run_save_path):
            os.makedirs(run_save_path)

        ## define constraints ##
        loss_fn = nn.BCEWithLogitsLoss()
        c = create_constraint_from_cfg(
            cfg=cfg,
            dataset=train_ds,
            group_indices=group_ind_train,
            loss_fn=loss_fn,
            device=device,
            seed=RUN_IDX)
        
        ## define network ##
        # TODO: add a choice of net to cfg
        net = SimpleNet(in_shape=X_test.shape[1], out_shape=1, dtype=DTYPE).to(device)

        ## define optimizer
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
            verbose=True,
        )

        ## Process and save results ##
        model_checkpoints_save_path = os.path.join(run_save_path, 'model_states')
        os.makedirs(model_checkpoints_save_path, exist_ok=True)
        for iteration, state_dict in history['params']['w'].items():
            # save model
            torch.save(state_dict, os.path.join(model_checkpoints_save_path, f'{iteration}.pt'))
        
        # save optimizer states
        params = pd.DataFrame(history["params"])
        values = pd.DataFrame(history["values"])
        t = pd.Series(history["time"], name="time")
        alg_states_df = pd.concat([t, params, values], axis=1)
        alg_states_df.to_csv(os.path.join(run_save_path, 'alg_states.csv'))

        ## SAVE MODEL ##
        torch.save(net.state_dict(), os.path.join(run_save_path, 'model.pt'))
        print(f"Model saved to: {run_save_path} as model.pt")
        print("")
        
        ####################################################
        ### CALCULATE STATS ON EVERY ALGORITHM ITERATION ###
        ####################################################
        save_train = True
        save_val = True
        save_test = True

        save_states_index = alg_states_df.index % cfg.alg.params.save_state_interval == 0

        if save_train:
            # TODO: join with table of iters and times based on checkpoint save interval
            table_train = run_summary_full_set(model=net, dataset=train_ds, path=model_checkpoints_save_path, constraints=c, loss_fn=torch.nn.BCEWithLogitsLoss())
            table_train.index = alg_states_df.index[save_states_index]
            table_train['time'] = alg_states_df['time'].loc[save_states_index]
        if save_val:
            X_val_tensor, y_val_tensor = tensor(X_val, dtype=DTYPE), tensor(y_val, dtype=DTYPE)
            val_ds = TensorDataset(X_val_tensor, y_val_tensor)
            c_val = create_constraint_from_cfg(cfg, dataset=val_ds, group_indices=group_ind_val, loss_fn=torch.nn.BCEWithLogitsLoss(), device=device)

            table_val = run_summary_full_set(model=net, dataset=val_ds, path=model_checkpoints_save_path, constraints=c_val, loss_fn=torch.nn.BCEWithLogitsLoss())
            table_val.index = alg_states_df.index[save_states_index]
            table_val['time'] = alg_states_df['time'].loc[save_states_index]
        if save_test:
            X_test_tensor, y_test_tensor = tensor(X_test, dtype=DTYPE), tensor(y_test, dtype=DTYPE)
            test_ds = TensorDataset(X_test_tensor, y_test_tensor)
            c_test = create_constraint_from_cfg(cfg, dataset=test_ds, group_indices=group_ind_test, loss_fn=torch.nn.BCEWithLogitsLoss(), device=device)

            table_test = run_summary_full_set(model=net, dataset=test_ds, path=model_checkpoints_save_path, constraints=c_test, loss_fn=torch.nn.BCEWithLogitsLoss())
            table_test.index = alg_states_df.index[save_states_index]
            table_test['time'] = alg_states_df['time'].loc[save_states_index]

        fname = f"full_set_eval"
        fext = ".csv"
        if save_train:
            fname_train = fname + "_train" + fext
            save_path = os.path.join(run_save_path, fname_train)
            print(f"Saving to: {save_path}")
            table_train.to_pickle(save_path)

        if save_val:
            fname_val = fname + "_val" + fext
            save_path = os.path.join(run_save_path, fname_val)
            print(f"Saving to: {save_path}")
            table_val.to_pickle(save_path)

        if save_test:
            fname_test = fname + "_test" + fext
            save_path = os.path.join(run_save_path, fname_test)
            print(f"Saving to: {save_path}")
            table_test.to_pickle(save_path)

            
if __name__ == "__main__":
    run()