from utils import *
from utils import create_model, create_conv_model
from humancompatible.train.fairness.utils import BalancedBatchSampler
from itertools import product
import torch
from _data_sources import load_data_FT, load_data_DUTCH, load_data_norm, load_data_cifar10, load_data_cifar100
import pandas as pd
import os
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from humancompatible.train.dual_optim import ALM, PBM, MoreauEnvelope
from constraints import loss_per_group, posrate_per_group, weight_constraint, posrate_fairret_constraint
import json

def save_best_params(best, name, result_dir):
    params, idx, vloss, vc, _ = best
    json.dump({"method": name, "best_params": params,
               "best_idx": int(idx[0]), "val_loss": float(vloss),
               "val_max_c": float(vc)}, open(f"{result_dir}/best_{name}.json","w"))
    

def runs_to_df(runs):
    
    return pd.concat([pd.DataFrame(h).set_index('epoch') for h in runs], keys=range(len(runs)))

### GRID SEARCH
def extract_best_params(runs, param_grid, val_c_tolerance, filter='upper'):
    # concat into one dataframe with kwarg_idx and epoch as indices
    runs = runs_to_df(runs)
    
    # filter out entries with unsatisfied validation constraints
    runs_filt = runs.copy()
    if filter == 'upper':
        runs_filt['max_c'] = runs_filt[[col for col in runs_filt.columns if col.startswith('c_')]].max(axis=1)
        runs_filt = runs_filt[runs_filt['max_c'] <= val_c_tolerance]
    elif filter == 'both':
        runs_filt['max_abs_c'] = runs_filt[[col for col in runs_filt.columns if col.startswith('c_')]].abs().max(axis=1)
        runs_filt = runs_filt[(runs_filt['max_abs_c'] <= val_c_tolerance)]
    elif filter == 'none':
        runs_filt['max_abs_c'] = runs_filt[[col for col in runs_filt.columns if col.startswith('c_')]].abs().max(axis=1)

    # argmin of validation loss
    min_feasible_val_loss_idx = runs_filt['loss'].idxmin(axis=0)
    min_feasible_val_loss_params = param_grid[min_feasible_val_loss_idx[0]]

    min_val_loss = runs_filt['loss'][min_feasible_val_loss_idx]
    min_val_c = (runs_filt['max_abs_c'] if 'max_abs_c' in runs_filt.columns else runs_filt['max_c'])[min_feasible_val_loss_idx]

    return min_feasible_val_loss_params, min_feasible_val_loss_idx, min_val_loss, min_val_c, runs


def main(data_cfg, task_cfg, n_epochs, constraint_cfg, device, seed):
    torch.manual_seed(seed)
    
    dataset = data_cfg['name']
    task = task_cfg.task
    result_dir = "results/" + dataset + '_' + task + str(seed)
    
    os.makedirs(result_dir, exist_ok=True)

    ### load data ###

    if dataset == 'folktables':
        data_source = lambda batch_size: load_data_FT(batch_size, device, **data_cfg['kwargs'])
        # if data_cfg['sens_attrs'] == ['MAR', 'SEX']:
        #     data_source = lambda batch_size: load_data_FT_prod(batch_size, device)
        # elif data_cfg['sens_attrs'] == ['SEX']:
        #     data_source = lambda batch_size: load_data_FT_vec(batch_size, device)
    elif dataset == 'dutch':
        data_source = load_data_DUTCH
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    
    if task == 'weight_norm':
        data_source = lambda batch_size: load_data_norm(batch_size, device)

    batch_size = task_cfg.batch_size
    if task == 'cifar10':
        dataloader_train, dataloader_val, classes, class_ind = load_data_cifar10(device=device)
        features_train, sens_train, labels_train = next(iter(dataloader_train))
        create_model_fn = create_conv_model
        model_kwargs = {}
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    elif task == 'cifar100':
        dataloader_train, dataloader_val, classes, class_ind = load_data_cifar100(device=device)
        features_train, sens_train, labels_train = next(iter(dataloader_train))
        create_model_fn = create_conv_model
        model_kwargs = {}
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val), (features_test, sens_test, labels_test) = data_source(batch_size)
        features_val = features_test
        sens_val = sens_test
        labels_val = labels_test
        create_model_fn = create_model
        model_kwargs = {'input_shape': features_train.shape[1], 'latent_size1': 64, 'latent_size2': 32}
        criterion = torch.nn.functional.binary_cross_entropy_with_logits

    data_val = (features_val, sens_val, labels_val) if task not in ['cifar10', 'cifar100'] else dataloader_val

    # Define hyperparameter grids
    pbm_grid = [
        {
            "primal__lr": lr,
            "dual__penalty_mult": dual_lr,
            "dual__penalty_update": p_update,
            "dual__pbf": pb_func,
            "dual__penalty_range": p_range,
            "dual__gamma": dual_gamma,
            "dual__delta": 1.,
            "moreau__mu": moreau_mu
        }
        for (
                lr,
                dual_lr,
                p_update,
                pb_func,
                p_range,
                dual_gamma,
                # dual_delta,
                moreau_mu
            ) in product (
            [0.001, 0.005, 0.01, 0.05],
            [0., 0.1, 0.2, 0.5],
            ["dimin_adapt"],
            ["quadratic_logarithmic"],
            [[1e-1, 1.], [1e-2, 1.]],
            [0.9],
            # [0.9, 1.0, 1.1],
            [0.0, 1.0, 2.]
            )
    ]

    alm_max_grid = [
        {
            "primal__lr": lr, 
            "dual__lr": dual_lr,
            "dual__penalty": penalty,
            # "dual__momentum": dual_momentum,
            "moreau__mu": moreau_mu
        }
        for (
                lr,
                dual_lr,
                penalty,
                # dual_momentum,
                moreau_mu
            ) in product (
            [0.001, 0.005, 0.01, 0.05],
            [0.001, 0.005, 0.01, 0.05],
            [0., 1.],
            # [0., 0.1, 0.2, 0.5],
            [0.0, 1.0, 2.]
            )
    ]

    alm_proj_grid = [
        {
            "primal__lr": lr, 
            "dual__lr": dual_lr,
            "dual__penalty": penalty,
            "dual__is_ineq": True,
            "moreau__mu": moreau_mu
        }
        for (
                lr,
                dual_lr,
                penalty,
                # dual_momentum,
                moreau_mu
            ) in product (
            [0.001, 0.005, 0.01, 0.05],
            [0.001, 0.005, 0.01, 0.05],
            [0., 1.],
            # [0., 0.1, 0.2, 0.5],
            [0.0, 1.0, 2.]
            )
    ]

    ssg_grid = [
        {
            "primal__lr": lr, 
            "dual__lr": dual_lr,
            "moreau__mu": moreau_mu,
        }
        for (
                lr,
                dual_lr,
                moreau_mu,
            ) in product (
            [0.001, 0.005, 0.01, 0.05],
            [0.001, 0.005, 0.01, 0.05],
            [0., 2.,]
            )
    ]

    adam_grid = [{"primal__lr": lr} for lr in [0.001, 0.005, 0.01, 0.05]]

    # Determine constraint function and parameters based on task
    fuse_loss_constraint = False
    if constraint_cfg is not None:
        
        c = importlib.import_module("constraint_meta").__dict__.get(constraint_cfg['name'])
        
        if constraint_cfg['name'].startswith('Fairret'):
            # if fairret-based, load statistic
            statistic = importlib.import_module("fairret.statistic").__dict__.get(constraint_cfg['statistic'])
            statistic = statistic()
            # load fairret loss if needed
            if constraint_cfg['name'] == 'FairretAgg':
                fair_loss = importlib.import_module("fairret.loss").__dict__.get(constraint_cfg['loss'])
                c = c(loss=fair_loss(statistic), **constraint_cfg.get('kwargs', {}))
            else:
                c = c(statistic=statistic, **constraint_cfg.get('kwargs', {}))
        
        elif constraint_cfg['name'].startswith('Loss'):
            # loss-based, load loss function with no reduction
            loss = importlib.import_module("torch.nn").__dict__.get(constraint_cfg['loss'])
            c = c(loss=loss(reduction='none'), **constraint_cfg.get('kwargs', {}))
            fuse_loss_constraint = True

            m = c.m_fn(sens_train.shape[-1])
            constraint_fn = c.compute_constraints
            constraint_bound = constraint_cfg['bound']

        m = c.m_fn(sens_train.shape[-1])
        constraint_fn = c.compute_constraints
        constraint_bound = constraint_cfg['bound']

    elif task == 'weight_norm':
        constraint_fn = weight_constraint
        constraint_bound = 2.0
        m = 6


    # Run experiments

    if 'adam' in task_cfg.algorithms:
        seed = seed
        torch.manual_seed(seed)
        _, adam_history_train, adam_history_val = run_grid(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=None,
            param_grid=adam_grid,
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            dataloader=dataloader_train,
            data_train=(features_train, sens_train, labels_train),
            data_val=data_val,
            mode='torch',
            verbose=False,
            constraints_to_eq=False,
            use_slack=False,
            fuse_loss_constraint=fuse_loss_constraint,
            model_gen=create_model_fn,
            model_kwargs=model_kwargs,
            device=device,
            criterion = criterion)

        best_adam_params = extract_best_params(adam_history_val, adam_grid, None, filter='none')
        save_best_params(best_adam_params,'adam',result_dir)

        print('\n------------\n')
        print('adam')
        print(best_adam_params[0], best_adam_params[1])
        print(f'loss: {best_adam_params[2]}')
        print(f'max c: {best_adam_params[3]}')
        print('\n------------\n')
        grid_adam = pd.DataFrame(adam_grid)
        runs_adam_train = runs_to_df(adam_history_train)
        runs_adam_train.to_csv(f'{result_dir}/runs_adam_train.csv')
        runs_adam_val = runs_to_df(adam_history_val)
        runs_adam_val.to_csv(f'{result_dir}/runs_adam_val.csv')
        grid_adam.to_csv(f'{result_dir}/grid_adam.csv')
        del adam_history_train, adam_history_val, runs_adam_train, runs_adam_val, grid_adam

    #################################################################
    if 'pbm' in task_cfg.algorithms:
        seed = seed
        torch.manual_seed(seed)
        _, pbm_history_train, pbm_history_val = run_grid(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=PBM,
            param_grid=pbm_grid,
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            dataloader=dataloader_train,
            data_train=(features_train, sens_train, labels_train),
            data_val=data_val,
            mode='hc',
            verbose=False,
            constraints_to_eq=False,
            use_slack=False,
            fuse_loss_constraint=fuse_loss_constraint,
            model_gen=create_model_fn,
            model_kwargs=model_kwargs,
            device=device,
            criterion=criterion)

        best_pbm_params = extract_best_params(pbm_history_val, pbm_grid, constraint_bound*1.1, filter='upper')
        save_best_params(best_pbm_params,'pbm',result_dir)

        print('\n------------\n')
        print('PBM')
        print(best_pbm_params[0], best_pbm_params[1])
        print(f'loss: {best_pbm_params[2]}')
        print(f'max c: {best_pbm_params[3]}')
        print('\n------------\n')
        grid_pbm = pd.DataFrame(pbm_grid)
        runs_pbm_train = runs_to_df(pbm_history_train)
        runs_pbm_train.to_csv(f'{result_dir}/runs_pbm_train.csv')
        runs_pbm_val = runs_to_df(pbm_history_val)
        runs_pbm_val.to_csv(f'{result_dir}/runs_pbm_val.csv')
        grid_pbm.to_csv(f'{result_dir}/grid_pbm.csv')
        del pbm_history_train, pbm_history_val, runs_pbm_train, runs_pbm_val, grid_pbm

    if 'alm_proj' in task_cfg.algorithms:
        seed = seed
        torch.manual_seed(seed)
        _, alm_history_train, alm_history_val = run_grid(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=ALM,
            param_grid=alm_proj_grid,
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            dataloader=dataloader_train,
            data_train=(features_train, sens_train, labels_train),
            data_val=data_val,
            mode='hc',
            verbose=False,
            constraints_to_eq=False,
            use_slack=False,
            fuse_loss_constraint=fuse_loss_constraint,
            model_gen=create_model_fn,
            model_kwargs=model_kwargs,
            device=device,
            criterion = criterion)

        best_alm_params = extract_best_params(alm_history_val, alm_proj_grid, constraint_bound*1.1, filter='upper')
        save_best_params(best_alm_params,'alm_proj',result_dir)

        print('\n------------\n')
        print('ALM')
        print(best_alm_params[0], best_alm_params[1])
        print(f'loss: {best_alm_params[2]}')
        print(f'max c: {best_alm_params[3]}')
        print('\n------------\n')
        grid_alm = pd.DataFrame(alm_proj_grid)
        runs_alm_train = runs_to_df(alm_history_train)
        runs_alm_train.to_csv(f'{result_dir}/runs_alm_proj_train.csv')
        runs_alm_val = runs_to_df(alm_history_val)
        runs_alm_val.to_csv(f'{result_dir}/runs_alm_proj_val.csv')
        grid_alm.to_csv(f'{result_dir}/grid_alm_proj.csv')
        del alm_history_train, alm_history_val, runs_alm_train, runs_alm_val, grid_alm

    if 'alm_max' in task_cfg.algorithms:
        seed = seed
        torch.manual_seed(seed)
        _, alm_history_train, alm_history_val = run_grid(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=ALM,
            param_grid=alm_max_grid,
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            dataloader=dataloader_train,
            data_train=(features_train, sens_train, labels_train),
            data_val=data_val,
            mode='hc',
            verbose=False,
            constraints_to_eq=True,
            use_slack=False,
            fuse_loss_constraint=fuse_loss_constraint,
            model_gen=create_model_fn,
            model_kwargs=model_kwargs,
            device=device,
            criterion=criterion)

        best_alm_params = extract_best_params(alm_history_val, alm_max_grid, constraint_bound*1.1, filter='upper')
        save_best_params(best_alm_params,'alm_max',result_dir)

        print('\n------------\n')
        print('ALM')
        print(best_alm_params[0], best_alm_params[1])
        print(f'loss: {best_alm_params[2]}')
        print(f'max c: {best_alm_params[3]}')
        print('\n------------\n')
        grid_alm = pd.DataFrame(alm_max_grid)
        runs_alm_train = runs_to_df(alm_history_train)
        runs_alm_train.to_csv(f'{result_dir}/runs_alm_max_train.csv')
        runs_alm_val = runs_to_df(alm_history_val)
        runs_alm_val.to_csv(f'{result_dir}/runs_alm_max_val.csv')
        grid_alm.to_csv(f'{result_dir}/grid_alm_max.csv')
        del alm_history_train, alm_history_val, runs_alm_train, runs_alm_val, grid_alm

    if 'ssg' in task_cfg.algorithms:
        seed = seed
        torch.manual_seed(seed)
        _, ssg_history_train, ssg_history_val = run_grid(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=torch.optim.Adam,
            param_grid=ssg_grid,
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            dataloader=dataloader_train,
            data_train=(features_train, sens_train, labels_train),
            data_val=data_val,
            mode='sw',
            verbose=False,
            constraints_to_eq=False,
            use_slack=False,
            fuse_loss_constraint=fuse_loss_constraint,
            model_gen=create_model_fn,
            model_kwargs=model_kwargs,
            device=device,
            criterion=criterion)

        best_ssg_params = extract_best_params(ssg_history_val, ssg_grid, constraint_bound*1.1, filter='upper')
        save_best_params(best_ssg_params,'ssg',result_dir)

        print('\n------------\n')
        print('SSG')
        print(best_ssg_params[0], best_ssg_params[1])
        print(f'loss: {best_ssg_params[2]}')
        print(f'max c: {best_ssg_params[3]}')
        print('\n------------\n')
        grid_ssg = pd.DataFrame(ssg_grid)
        runs_ssg_train = runs_to_df(ssg_history_train)
        runs_ssg_train.to_csv(f'{result_dir}/runs_ssg_train.csv')
        runs_ssg_val = runs_to_df(ssg_history_val)
        runs_ssg_val.to_csv(f'{result_dir}/runs_ssg_val.csv')
        grid_ssg.to_csv(f'{result_dir}/grid_ssg.csv')
        del ssg_history_train, ssg_history_val, runs_ssg_train, runs_ssg_val, grid_ssg


@hydra.main(version_base=None, config_path="conf", config_name="benchmark")
def hydra_main(cfg: DictConfig):
    """Run grid search with Hydra config."""
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    task_cfg = cfg.task
    n_epochs = cfg.n_epochs
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
    torch.set_default_device(device)
    constraint_cfg = OmegaConf.to_container(task_cfg.constraint, resolve=True) if 'constraint' in task_cfg else None
    seed = task_cfg.seed
    
    main(data_cfg, task_cfg, n_epochs, constraint_cfg, device, seed)



if __name__ == "__main__":
    hydra_main()
