import importlib
import os

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from _data_sources import load_data_FT, load_data_DUTCH, load_data_norm
from benchmark_utils import *
from plotting import plot_losses_and_constraints_stochastic
from humancompatible.train.dual_optim import ALM, PBM
from constraints import weight_constraint

def run_benchmark(data_cfg, task, n_runs, n_epochs, constraint_cfg, pbm_params, alm_params, ssg_params, adam_params, cfg):
    seed = 0
    torch.manual_seed(seed)
    dataset = data_cfg['name']
    result_dir = "results/" + dataset + '_' + task

    os.makedirs(result_dir, exist_ok=True)

    ### load data ###

    if dataset == 'folktables':
        data_source = lambda batch_size: load_data_FT(batch_size, sens_attrs=data_cfg['sens_attrs'], states=data_cfg['states'])
    elif dataset == 'dutch':
        data_source = load_data_DUTCH
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    
    if task == 'weight_norm':
        data_source = load_data_norm

    batch_size = cfg.batch_size

    (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val) = data_source(batch_size)
    
    ### construct the constraint ###

    c = importlib.import_module("constraint_meta").__dict__.get(constraint_cfg['name'])
    if constraint_cfg['name'].startswith('Fairret'):
        # if fairret-based, load statistic
        statistic = importlib.import_module("fairret.statistic").__dict__.get(constraint_cfg['statistic'])
        statistic = statistic()
        # load fairret loss if needed
        if constraint_cfg['name'] == 'FairretLoss':
            fair_loss = importlib.import_module("fairret.loss").__dict__.get(constraint_cfg['loss'])
            c = c(loss=fair_loss(statistic), **constraint_cfg.get('constraint_kwargs', {}))
        else:
            c = c(statistic=statistic, **constraint_cfg.get('constraint_kwargs', {}))
    else:
        # not fairret-based, just initialize constraint
        c = c(**constraint_cfg.get('constraint_kwargs', {}))

    m = c.m_fn(sens_train.shape[-1])
    constraint_fn = c.compute_constraints
    constraint_bound = constraint_cfg['bound']

    if task == 'weight_norm':
        constraint_fn = weight_constraint
        constraint_bound = 2.0

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
    grid_adam = pd.DataFrame.from_dict([dict(adam_params)])
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
    grid_pbm = pd.DataFrame.from_dict([dict(pbm_params)])
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
    grid_alm = pd.DataFrame.from_dict([dict(alm_params)])
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
    grid_ssg = pd.DataFrame.from_dict([dict(ssg_params)])
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
        plot_max_constraint=m > 5,
        save_path=result_dir + '/plot.png'
    )


@hydra.main(version_base=None, config_path="conf", config_name="benchmark")
def hydra_main(cfg: DictConfig):
    task_cfg = cfg.task
    # task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    pbm_params = OmegaConf.to_container(task_cfg.pbm_params, resolve=True)
    alm_params = OmegaConf.to_container(task_cfg.alm_params, resolve=True)
    ssg_params = OmegaConf.to_container(task_cfg.ssg_params, resolve=True)
    adam_params = OmegaConf.to_container(task_cfg.adam_params, resolve=True)
    constraint_cfg = OmegaConf.to_container(task_cfg.constraint, resolve=True)

    run_benchmark(
        task_cfg.data,
        task_cfg.task,
        cfg.n_runs,
        cfg.n_epochs,
        constraint_cfg,
        pbm_params,
        alm_params,
        ssg_params,
        adam_params,
        task_cfg
    )


if __name__ == "__main__":
    hydra_main()
