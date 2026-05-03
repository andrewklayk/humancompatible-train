from utils import *
from utils import create_model, create_conv_model, create_resnet
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



def runs_to_df(runs):
    
    return pd.concat([pd.DataFrame(h).set_index('epoch') for h in runs], keys=range(len(runs)))

### GRID SEARCH
def extract_best_params(runs, param_grid, val_c_tolerance, filter='upper'):
    # concat into one dataframe with kwarg_idx and epoch as indices
    # runs = runs_to_df(runs)
    
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


def main(data_cfg, task_cfg, n_epochs, constraint_cfg, device, seeds):
    # torch.manual_seed(seed)
    
    dataset = data_cfg['name']
    task = task_cfg.task
    seeds = list(seeds)
    print(type(seeds))
    result_dir = dataset + '_' + task + '_' + ''.join([str(seed) for seed in seeds])
    
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
        data_source = load_data_norm

    batch_size = task_cfg.batch_size
    if task == 'cifar10':
        dataloader_train, dataloader_val, dataloader_test, classes, class_ind = load_data_cifar10(device=device)
        features_train, sens_train, labels_train = next(iter(dataloader_train))
        create_model_fn = create_conv_model
        model_kwargs = {}
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    elif task == 'cifar100':
        dataloader_train, dataloader_val, dataloader_test, classes, class_ind = load_data_cifar100(device=device)
        features_train, sens_train, labels_train = next(iter(dataloader_train))
        create_model_fn = create_resnet
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
            "dual__penalty_mult": p_m,
            "dual__penalty_update": p_update,
            "dual__pbf": pb_func,
            "dual__penalty_range": [0.1, 1.],
            "dual__gamma": dual_gamma,
            "dual__delta": 1.,
            "moreau__mu": moreau_mu
        }
        for (
                lr,
                p_m,
                p_update,
                pb_func,
                dual_gamma,
                moreau_mu
            ) in product (
            [0.0005, 0.001, 0.005, 0.01],
            [0.9, 0.99],
            ["dimin_adapt"],
            ["quadratic_logarithmic", "quadratic_reciprocal"],
            [0.9, 0.99],
            [0., 0.2, 0.5, 1., 2.]
            )
    ]

    pbm_grid_dimin = [
        {
            "primal__lr": lr,
            "dual__penalty_mult": p_m,
            "dual__penalty_update": p_update,
            "dual__pbf": pb_func,
            "dual__penalty_range": [0.1, 1.],
            "dual__gamma": dual_gamma,
            "dual__delta": 1.,
            "moreau__mu": moreau_mu
        }
        for (
                lr,
                p_m,
                p_update,
                pb_func,
                dual_gamma,
                moreau_mu
            ) in product (
            [0.0005, 0.001, 0.005, 0.01],
            [1., 0.999, 0.99],
            ["dimin"],
            ["quadratic_logarithmic", "quadratic_reciprocal"],
            [0.9, 0.99],
            [0., 0.2, 0.5, 1., 2.]
            )
    ]

    alm_grid = [
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
            [0.001, 0.002, 0.005, 0.01],
            [0.001, 0.002, 0.005, 0.01],
            [0., 1., 2.],
            # [0., 0.1, 0.2, 0.5],
            [0., 0.5, 1., 2., 4.]
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

    alm_max_grid = alm_grid
    alm_slack_grid = alm_grid

    adam_grid = [{"lr": lr} for lr in [0.001, 0.005, 0.01, 0.05]]

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
        train_hist = []
        val_hist = []
        for seed in seeds:
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
            
            train_hist.append(adam_history_train)
            val_hist.append(adam_history_val)
        
        train_hist = pd.concat(train_hist)
        adam_history_train = train_hist.groupby(train_hist.index).mean()
        val_hist = pd.concat(val_hist)
        adam_history_val = val_hist.groupby(val_hist.index).mean()

        runs_adam_train = runs_to_df(adam_history_train)
        runs_adam_val = runs_to_df(adam_history_val)
        best_adam_params = extract_best_params(runs_adam_val, adam_grid, None, filter='none')

        print('\n------------\n')
        print('adam')
        print(best_adam_params[0], best_adam_params[1])
        print(f'loss: {best_adam_params[2]}')
        print(f'max c: {best_adam_params[3]}')
        print('\n------------\n')
        grid_adam = pd.DataFrame(adam_grid)
        runs_adam_train.to_csv(f'{result_dir}/runs_adam_train.csv')
        runs_adam_val.to_csv(f'{result_dir}/runs_adam_val.csv')
        grid_adam.to_csv(f'{result_dir}/grid_adam.csv')
        del adam_history_train, adam_history_val, runs_adam_train, runs_adam_val, grid_adam

    #################################################################
    print(task_cfg.alg_version)
    if 'pbm' in task_cfg.algorithms:
        train_hist = []
        val_hist = []
        for seed in seeds:
            torch.manual_seed(seed)
            if task_cfg.alg_version == 'adapt':
                grid = pbm_grid
            elif task_cfg.alg_version == 'dimin':
                grid = pbm_grid_dimin
                print(grid)
            else:
                raise ValueError(task_cfg)
            _, pbm_history_train, pbm_history_val = run_grid(
                m=m,
                primal_opt=torch.optim.Adam,
                dual_opt=PBM,
                param_grid=grid,
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
            
            train_hist.append(pbm_history_train)
            val_hist.append(pbm_history_val)
        
        train_hist = pd.concat(train_hist)
        pbm_history_train = train_hist.groupby(train_hist.index).mean()
        val_hist = pd.concat(val_hist)
        pbm_history_val = val_hist.groupby(val_hist.index).mean()

        runs_pbm_train = runs_to_df(pbm_history_train)
        runs_pbm_val = runs_to_df(pbm_history_val)
        best_pbm_params = extract_best_params(runs_pbm_val, grid, constraint_bound*1.1, filter='both')

        print('\n------------\n')
        name = 'pbm' + '_' + task_cfg.alg_version
        print(name)
        print(best_pbm_params[0], best_pbm_params[1])
        print(f'loss: {best_pbm_params[2]}')
        print(f'max c: {best_pbm_params[3]}')
        print('\n------------\n')
        grid_pbm = pd.DataFrame(grid)
        runs_pbm_train.to_csv(f'{result_dir}/runs_{name}_train.csv')
        runs_pbm_val.to_csv(f'{result_dir}/runs_{name}_val.csv')
        grid_pbm.to_csv(f'{result_dir}/grid_{name}.csv')
        del pbm_history_train, pbm_history_val, runs_pbm_train, runs_pbm_val, grid_pbm

    if 'alm_slack' in task_cfg.algorithms:
        train_hist = []
        val_hist = []
        for seed in seeds:
            torch.manual_seed(seed)
            _, alm_history_train, alm_history_val = run_grid(
                m=m,
                primal_opt=torch.optim.Adam,
                dual_opt=ALM,
                param_grid=alm_grid,
                n_epochs=n_epochs,
                constraint_fn=constraint_fn,
                constraint_bound=constraint_bound,
                dataloader=dataloader_train,
                data_train=(features_train, sens_train, labels_train),
                data_val=data_val,
                mode='hc',
                verbose=False,
                constraints_to_eq=True,
                use_slack=True,
                fuse_loss_constraint=fuse_loss_constraint,
                model_gen=create_model_fn,
                model_kwargs=model_kwargs,
                device=device,
                criterion = criterion)
            
            train_hist.append(alm_history_train)
            val_hist.append(alm_history_val)
        
        train_hist = pd.concat(train_hist)
        alm_history_train = train_hist.groupby(train_hist.index).mean()
        val_hist = pd.concat(val_hist)
        alm_history_val = val_hist.groupby(val_hist.index).mean()

        runs_alm_train = runs_to_df(alm_history_train)
        runs_alm_val = runs_to_df(alm_history_val)
        best_alm_params = extract_best_params(runs_alm_val, alm_grid, constraint_bound*1.1, filter='both')

        print('\n------------\n')
        print('ALM')
        print(best_alm_params[0], best_alm_params[1])
        print(f'loss: {best_alm_params[2]}')
        print(f'max c: {best_alm_params[3]}')
        print('\n------------\n')
        grid_alm = pd.DataFrame(alm_grid)
        runs_alm_train.to_csv(f'{result_dir}/runs_alm_slack_train.csv')
        runs_alm_val.to_csv(f'{result_dir}/runs_alm_slack_val.csv')
        grid_alm.to_csv(f'{result_dir}/grid_alm_slack.csv')
        del alm_history_train, alm_history_val, runs_alm_train, runs_alm_val, grid_alm

    if 'alm_max' in task_cfg.algorithms:
        train_hist = []
        val_hist = []
        for seed in seeds:
            torch.manual_seed(seed)
            _, alm_history_train, alm_history_val = run_grid(
                m=m,
                primal_opt=torch.optim.Adam,
                dual_opt=ALM,
                param_grid=alm_grid,
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
            
            train_hist.append(alm_history_train)
            val_hist.append(alm_history_val)
        
        train_hist = pd.concat(train_hist)
        alm_history_train = train_hist.groupby(train_hist.index).mean()
        val_hist = pd.concat(val_hist)
        alm_history_val = val_hist.groupby(val_hist.index).mean()

        runs_alm_train = runs_to_df(alm_history_train)
        runs_alm_val = runs_to_df(alm_history_val)
        best_alm_params = extract_best_params(runs_alm_val, alm_grid, constraint_bound*1.1, filter='both')

        print('\n------------\n')
        print('ALM')
        print(best_alm_params[0], best_alm_params[1])
        print(f'loss: {best_alm_params[2]}')
        print(f'max c: {best_alm_params[3]}')
        print('\n------------\n')
        grid_alm = pd.DataFrame(alm_grid)
        runs_alm_train = runs_to_df(alm_history_train)
        runs_alm_train.to_csv(f'{result_dir}/runs_alm_max_train.csv')
        runs_alm_val = runs_to_df(alm_history_val)
        runs_alm_val.to_csv(f'{result_dir}/runs_alm_max_val.csv')
        grid_alm.to_csv(f'{result_dir}/grid_alm_max.csv')
        del alm_history_train, alm_history_val, runs_alm_train, runs_alm_val, grid_alm

    if 'ssg' in task_cfg.algorithms:
        train_hist = []
        val_hist = []
        for seed in seeds:
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
            
            train_hist.append(ssg_history_train)
            val_hist.append(ssg_history_val)
        
        train_hist = pd.concat(train_hist)
        ssg_history_train = train_hist.groupby(train_hist.index).mean()
        val_hist = pd.concat(val_hist)
        ssg_history_val = val_hist.groupby(val_hist.index).mean()

        runs_ssg_train = runs_to_df(ssg_history_train)
        runs_ssg_val = runs_to_df(ssg_history_val)
        best_ssg_params = extract_best_params(runs_ssg_val, ssg_grid, constraint_bound*1.1, filter='both')

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
    constraint_cfg = OmegaConf.to_container(task_cfg.constraint, resolve=True)
    seeds = task_cfg.seeds
    
    main(data_cfg, task_cfg, n_epochs, constraint_cfg, device, seeds)


if __name__ == "__main__":
    hydra_main()
