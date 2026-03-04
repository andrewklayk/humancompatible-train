from benchmark_utils import *
from itertools import product
import torch
from _data_sources import load_data_FT_vec, load_data_FT, load_data_DUTCH, load_data_norm
import pandas as pd
import argparse
import os
import torch.nn.functional as F

from fairret.statistic import PositiveRate
from fairret.loss import NormLoss

from plotting import plot_losses_and_constraints_stochastic
from humancompatible.train.dual_optim import ALM, PBM

from constraints import loss_per_group, posrate_per_group, weight_constraint, posrate_fairret_constraint

def runs_to_df(runs):
    
    return pd.concat([pd.DataFrame(h).set_index('epoch') for h in runs], keys=range(len(runs)))


def main(dataset, task, n_runs, n_epochs):
    seed = 0
    torch.manual_seed(seed)
    result_dir = "results/" + dataset + '_' + task
    
    os.makedirs(result_dir, exist_ok=True)

    if dataset == 'folktables':
        # drop small groups
        group_defs = [
            {'SEX': 2, 'MAR': 3},
            {'SEX': 2, 'MAR': 1},
            {'SEX': 2, 'MAR': 5},
            {'SEX': 1, 'MAR': 3},
            {'SEX': 1, 'MAR': 1},
            {'SEX': 1, 'MAR': 5},
        ]
        if task == 'eqop':
            data_source = lambda batch_size: load_data_FT(batch_size, sens_attrs = ['MAR', 'SEX'], states=['VA'], sens_groups=group_defs)
            batch_size = 30
        elif task == 'vec':
            data_source = lambda batch_size: load_data_FT(batch_size, sens_attrs = ['SEX'], states=['VA'])
            batch_size = 64
        elif task == 'weight_norm':
            data_source = load_data_norm
            batch_size = 64
        elif task == 'loss':
            data_source = lambda batch_size: load_data_FT(batch_size, sens_attrs = ['MAR'], states=['VA'])
            batch_size = 80
    elif dataset == 'dutch':
        batch_size = 72
        data_source = load_data_DUTCH
    
    (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val) = data_source(batch_size)

    pbm_params = {
            "primal__lr": 0.01,
            "dual__lr": 0.99, "dual__mu": 0.3, "dual__penalty_update": "dimin", "dual__pbf": "quadratic_logarithmic", "dual__momentum": 0.9,
            "moreau__mu": 4.0
        }
    
    alm_params = {
            "primal__lr": 0.001, 
            "dual__lr": 0.05, "dual__penalty": 1.0, "dual__momentum": 0.3,
            "moreau__mu": 2.0
        }
    
    ssg_params = {
            "primal__lr": 0.01, 
            "dual__lr": 0.005,
            "moreau__mu": 0.
        }

    adam_params = {
        "lr": 0.01
    }

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
    elif task == 'weight_norm':
        m = 6
    else:
        raise ValueError("Unknown task")

    if task == 'vec':
        constraint_bound = 0.2
    elif task == 'eqop':
        constraint_bound = 0.1
    elif task == 'loss':
        constraint_bound = 0.05
    elif task == "weight_norm":
        constraint_bound = 2.0
    else:
        raise ValueError("Unknown task")

    #################################################################
    adam_history_train = []
    adam_history_val = []
    models = []
    for _ in range(n_runs):
        model, h_train, h_val = run_train(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=None,
            param_set=adam_params,
            data_train=(features_train, sens_train, labels_train),
            dataloader=dataloader_train,
            data_val=(features_val, sens_val, labels_val),
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            mode='unconstrained',
            verbose=m < 30,
            constraints_to_eq=False,
            use_slack=False
        )

        models.append(model)
        adam_history_train.append(h_train)
        adam_history_val.append(h_val)

    print('\n------------\n')
    print('ADAM done')
    print('\n------------\n')
    os.makedirs(result_dir + '/adam/models', exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'{result_dir}/adam/models/{i}.pt')
        del model
    grid_adam = pd.DataFrame.from_dict([adam_params])
    runs_adam_train = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in adam_history_train], keys=range(n_runs))
    runs_adam_train.to_csv(f'{result_dir}/runs_adam_train.csv')
    runs_adam_val = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in adam_history_val], keys=range(n_runs))
    runs_adam_val.to_csv(f'{result_dir}/runs_adam_val.csv')
    grid_adam.to_csv(f'{result_dir}/grid_adam.csv')
    del adam_history_train, adam_history_val, runs_adam_train, runs_adam_val, grid_adam

    ##################################################################
    models, pbm_history_train, pbm_history_val = [], [], []
    for _ in range(n_runs):
        model, h_train, h_val = run_train(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=PBM,
            param_set=pbm_params,
            data_train=(features_train, sens_train, labels_train),
            dataloader=dataloader_train,
            data_val=(features_val, sens_val, labels_val),
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            mode='hc',
            verbose=m < 30,
            constraints_to_eq=False,
            use_slack=False
        )
        models.append(model)
        pbm_history_train.append(h_train)
        pbm_history_val.append(h_val)

    print('\n------------\n')
    print('PBM done')
    print('\n------------\n')
    os.makedirs(result_dir + '/pbm/models', exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'{result_dir}/pbm/models/{i}.pt')
        del model
    grid_pbm = pd.DataFrame.from_dict([pbm_params])
    runs_pbm_train = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in pbm_history_train], keys=range(n_runs))
    runs_pbm_train.to_csv(f'{result_dir}/runs_pbm_train.csv')
    runs_pbm_val = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in pbm_history_val], keys=range(n_runs))
    runs_pbm_val.to_csv(f'{result_dir}/runs_pbm_val.csv')
    grid_pbm.to_csv(f'{result_dir}/grid_pbm.csv')
    del pbm_history_train, pbm_history_val, runs_pbm_train, runs_pbm_val, grid_pbm

    ##################################################################
    models, alm_history_train, alm_history_val = [], [], []
    for _ in range(n_runs):
        model, h_train, h_val = run_train(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=ALM,
            param_set=alm_params,
            data_train=(features_train, sens_train, labels_train),
            dataloader=dataloader_train,
            data_val=(features_val, sens_val, labels_val),
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            mode='hc',
            verbose=m < 30,
            constraints_to_eq=True,
            use_slack=True
        )
        models.append(model)
        alm_history_train.append(h_train)
        alm_history_val.append(h_val)

    print('\n------------\n')
    print('ALM done')
    print('\n------------\n')
    os.makedirs(result_dir + '/alm/models', exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'{result_dir}/alm/models/{i}.pt')
        del model
    grid_alm = pd.DataFrame.from_dict([alm_params])
    runs_alm_train = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in alm_history_train], keys=range(n_runs))
    runs_alm_train.to_csv(f'{result_dir}/runs_alm_train.csv')
    runs_alm_val = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in alm_history_val], keys=range(n_runs))
    runs_alm_val.to_csv(f'{result_dir}/runs_alm_val.csv')
    grid_alm.to_csv(f'{result_dir}/grid_alm.csv')
    del alm_history_train, alm_history_val, runs_alm_train, runs_alm_val, grid_alm

    ##################################################################
    models, ssg_history_train, ssg_history_val = [], [], []
    for _ in range(n_runs):
        model, h_train, h_val = run_train(
            m=m,
            primal_opt=torch.optim.Adam,
            dual_opt=torch.optim.Adam,
            param_set=ssg_params,
            data_train=(features_train, sens_train, labels_train),
            dataloader=dataloader_train,
            data_val=(features_val, sens_val, labels_val),
            n_epochs=n_epochs,
            constraint_fn=constraint_fn,
            constraint_bound=constraint_bound,
            mode='sw',
            verbose=m < 30,
            constraints_to_eq=False,
            use_slack=False
        )
        models.append(model)
        ssg_history_train.append(h_train)
        ssg_history_val.append(h_val)

    print('\n------------\n')
    print('SSG done')
    print('\n------------\n')
    os.makedirs(result_dir + '/ssg/models', exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'{result_dir}/ssg/models/{i}.pt')
        del model
    grid_ssg = pd.DataFrame.from_dict([ssg_params])
    runs_ssg_train = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in ssg_history_train], keys=range(n_runs))
    runs_ssg_train.to_csv(f'{result_dir}/runs_ssg_train.csv')
    runs_ssg_val = pd.concat([pd.DataFrame.from_dict(run_history) for run_history in ssg_history_val], keys=range(n_runs))
    runs_ssg_val.to_csv(f'{result_dir}/runs_ssg_val.csv')
    grid_ssg.to_csv(f'{result_dir}/grid_ssg.csv')
    del ssg_history_train, ssg_history_val, runs_ssg_train, runs_ssg_val, grid_ssg
    

    ###### PLOT ######

    def read_prepare_data(path: str):
        train = pd.read_csv(path)
        c_cols = [c for c in train.columns if c.startswith('c_')]
        mean = train.groupby(by='epoch').mean()
        std = train.groupby(by=['Unnamed: 0', 'epoch']).mean().groupby(by='epoch').std()
        loss_mean = mean['loss'].to_numpy()
        loss_std = std['loss'].to_numpy()
        cs_mean = mean[c_cols].to_numpy()
        cs_std = std[c_cols].to_numpy()

        return mean, loss_mean, loss_std, cs_mean, cs_std
    
    alg_names = []

    _, adam_loss_mean_train, adam_loss_std_train, adam_cs_mean_train, adam_cs_std_train = read_prepare_data(f'{result_dir}/runs_adam_train.csv')
    _, adam_loss_mean_val, adam_loss_std_val, adam_cs_mean_val, adam_cs_std_val = read_prepare_data(f'{result_dir}/runs_adam_val.csv')
    alg_names.append('Adam')

    _, pbm_loss_mean_train, pbm_loss_std_train, pbm_cs_mean_train, pbm_cs_std_train = read_prepare_data(f'{result_dir}/runs_pbm_train.csv')
    _, pbm_loss_mean_val, pbm_loss_std_val, pbm_cs_mean_val, pbm_cs_std_val = read_prepare_data(f'{result_dir}/runs_pbm_val.csv')
    alg_names.append('PBM')

    _, alm_loss_mean_train, alm_loss_std_train, alm_cs_mean_train, alm_cs_std_train = read_prepare_data(f'{result_dir}/runs_alm_train.csv')
    _, alm_loss_mean_val, alm_loss_std_val, alm_cs_mean_val, alm_cs_std_val = read_prepare_data(f'{result_dir}/runs_alm_val.csv')
    alg_names.append('ALM')

    _, ssg_loss_mean_train, ssg_loss_std_train, ssg_cs_mean_train, ssg_cs_std_train = read_prepare_data(f'{result_dir}/runs_ssg_train.csv')
    _, ssg_loss_mean_val, ssg_loss_std_val, ssg_cs_mean_val, ssg_cs_std_val = read_prepare_data(f'{result_dir}/runs_ssg_val.csv')
    alg_names.append('SSG')


    plot_losses_and_constraints_stochastic(
        [adam_loss_mean_train, pbm_loss_mean_train, alm_loss_mean_train, ssg_loss_mean_train],
        [adam_loss_std_train, pbm_loss_std_train, alm_loss_std_train, ssg_loss_std_train],
        [adam_cs_mean_train.T, pbm_cs_mean_train.T, alm_cs_mean_train.T, ssg_cs_mean_train.T],
        [adam_cs_std_train.T, pbm_cs_std_train.T, alm_cs_std_train.T, ssg_cs_std_train.T],
        constraint_bound,
        [adam_loss_mean_val, pbm_loss_mean_val, alm_loss_mean_val, ssg_loss_mean_val],
        [adam_loss_std_val, pbm_loss_std_val, alm_loss_std_val, ssg_loss_std_val],
        [adam_cs_mean_val.T, pbm_cs_mean_val.T, alm_cs_mean_val.T, ssg_cs_mean_val.T],
        [adam_cs_std_val.T, pbm_cs_std_val.T, alm_cs_std_val.T, ssg_cs_std_val.T],
        titles=['Adam', 'PBM', 'ALM', 'SSg'],
        plot_max_constraint = m > 5,
        save_path=result_dir + '/plot.png'
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--task", type=str, required=True, help="Name of the constraint to use.")
    parser.add_argument("--n_runs", type=int, required=True, help="number of runs to perform.")
    parser.add_argument("--n_epochs", type=int, required=True, help="number of epochs to run.")

    args = parser.parse_args()
    main(args.dataset, args.task, args.n_runs, args.n_epochs)
