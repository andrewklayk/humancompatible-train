from benchmark_utils import *
from humancompatible.train.fairness.utils import BalancedBatchSampler
from itertools import product
import torch
from _data_sources import load_data_FT_vec, load_data_FT_prod, load_data_DUTCH, load_data_norm
import pandas as pd
import argparse
import os

from fairret.statistic import PositiveRate
from fairret.loss import NormLoss

from torch.nn import functional as F

from humancompatible.train.dual_optim import ALM, PBM, MoreauEnvelope

from constraints import loss_per_group, posrate_per_group, weight_constraint, posrate_fairret_constraint



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


def main(dataset, task, n_epochs):
    seed = 0
    torch.manual_seed(seed)
    result_dir = dataset + '_' + task
    
    os.makedirs(result_dir, exist_ok=True)

    if dataset == 'folktables':
        if task == 'eqop':
            data_source = load_data_FT_prod
            batch_size = 30
        elif task == 'vec':
            data_source = lambda batch_size: load_data_FT_vec(batch_size, attr = 'SEX')
            batch_size = 64
        elif task == 'weight_norm':
            data_source = load_data_norm
            batch_size = 64
        elif task == 'loss':
            data_source = lambda batch_size: load_data_FT_vec(batch_size, attr = 'MAR')
            batch_size = 60
    elif dataset == 'dutch':
        batch_size = 72
        data_source = load_data_DUTCH
    
    (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val) = data_source(batch_size)

    # Define hyperparameter grids

    pbm_grid = [
        {
            "primal__lr": lr,
            "dual__lr": dual_lr, "dual__mu": dual_mu, "dual__penalty_update": p_update, "dual__pbf": pb_func, "dual__momentum": dual_momentum,
            "moreau__mu": moreau_mu
        }
        for (
                lr,
                dual_lr, dual_mu, p_update, pb_func, dual_momentum,
                moreau_mu
            ) in product (
            [0.001, 0.01, 0.05],
            [0.9, 0.99, 1.], [0.1, 0.3], ["dimin"], ["quadratic_logarithmic"], [0., 0.3, 0.9],
            [0., 2., 4.]
            )
    ]

    alm_grid = [
        {
            "primal__lr": lr, 
            "dual__lr": dual_lr, "dual__penalty": penalty, "dual__momentum": dual_momentum,
            "moreau__mu": moreau_mu
        }
        for (
                lr,
                dual_lr, penalty, dual_momentum,
                moreau_mu
            ) in product (
            [0.001, 0.01, 0.05],
            [0.001, 0.01, 0.05], [0., 1.], [0., 0.3, 0.9],
            [0., 2., 4.]
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
            [0]
            )
    ]

    alm_max_grid = alm_grid
    alm_slack_grid = alm_grid

    adam_grid = [{"lr": lr} for lr in [0.01]]

    if task == 'eqop':
        constraint_fn = posrate_per_group
    elif task == 'vec':
        constraint_fn = posrate_fairret_constraint
    elif task == 'weight_norm':
        constraint_fn = weight_constraint
    elif task == 'loss':
        constraint_fn = loss_per_group
    else:
        raise ValueError(f'Unknown task: {task}')

    if task == 'eqop':
        if dataset == 'dutch':
            m = 306
        elif dataset == 'folktables':
            m = 30
    elif task == 'vec':
        m = 1
    elif task == 'loss':
        if dataset == 'dutch':
            m == 18
        elif dataset == 'folktables':
            m = 5
    else:
        m = 6

    if task == 'vec':
        constraint_bound = 0.2
    elif task == 'eqop':
        constraint_bound = 0.1
    elif task == 'loss':
        constraint_bound = 0.05
    else:
        constraint_bound = 2.0


    # Run experiments

    #################################################################

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
        data_val=(features_val, sens_val, labels_val),
        mode = 'hc',
        verbose=False,
        constraints_to_eq = False,
        use_slack = False)

    best_pbm_params = extract_best_params(pbm_history_val, pbm_grid, constraint_bound*1.1, filter='both')

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
        data_val=(features_val, sens_val, labels_val),
        mode = 'hc',
        verbose=False,
        constraints_to_eq = True,
        use_slack = True)

    best_alm_params = extract_best_params(alm_history_val, alm_grid, constraint_bound*1.1, filter='both')

    print('\n------------\n')
    print('ALM')
    print(best_alm_params[0], best_alm_params[1])
    print(f'loss: {best_alm_params[2]}')
    print(f'max c: {best_alm_params[3]}')
    print('\n------------\n')
    grid_alm = pd.DataFrame(alm_grid)
    runs_alm_train = runs_to_df(alm_history_train)
    runs_alm_train.to_csv(f'{result_dir}/runs_alm_train.csv')
    runs_alm_val = runs_to_df(alm_history_val)
    runs_alm_val.to_csv(f'{result_dir}/runs_alm_val.csv')
    grid_alm.to_csv(f'{result_dir}/grid_alm.csv')
    del alm_history_train, alm_history_val, runs_alm_train, runs_alm_val, grid_alm

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
        data_val=(features_val, sens_val, labels_val),
        mode = 'sw',
        verbose=False,
        constraints_to_eq = False,
        use_slack = False)

    best_ssg_params = extract_best_params(ssg_history_val, ssg_grid, constraint_bound*1.1, filter='both')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--task", type=str, required=True, help="Name of the constraint to use.")
    parser.add_argument("--n_epochs", type=int, required=True, help="number of epochs to run.")

    args = parser.parse_args()
    main(args.dataset, args.task, args.n_epochs)
