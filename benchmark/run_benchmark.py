import importlib
import os
from time import time
from utils import *
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from utils import create_model, create_conv_model, create_resnet
from _data_sources import load_data_FT_prod, load_data_FT_vec, load_data_FT, load_data_DUTCH, load_data_norm, load_data_cifar10, load_data_cifar100
# from benchmark_utils import *
from plotting import plot_losses_and_constraints_stochastic
from humancompatible.train.dual_optim import ALM, PBM
from constraints import weight_constraint

def run_benchmark(data_cfg, task, n_runs, n_epochs, constraint_cfg, pbm_params, alm_params, ssg_params, adam_params, cfg):
    seed = 0
    torch.manual_seed(seed)
    dataset = data_cfg['name']

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
    torch.set_default_device(device)
    
    result_dir = "results/" + dataset + '_' + task

    os.makedirs(result_dir, exist_ok=True)

    ### load data ###

    alm_use_slack = True
    if dataset == 'folktables':
        model_kwargs = {}
        data_source = lambda batch_size: load_data_FT(batch_size=batch_size, device=device, **data_cfg['kwargs'])
        if data_cfg['kwargs']['sens_attrs'] == ['SEX']:
            alm_use_slack = True
        if data_cfg['kwargs']['sens_attrs'] == ['MAR', 'SEX']:
            alm_use_slack = True
        # if data_cfg['kwargs']['sens_attrs'] == ['MAR', 'SEX']:
        #     data_source = lambda batch_size: load_data_FT_prod(batch_size, device=device)
        # elif data_cfg['kwargs']['sens_attrs'] == ['SEX']:
        #     data_source = lambda batch_size: load_data_FT_vec(batch_size, device=device)
    elif dataset == 'dutch':
        data_source = load_data_DUTCH
        alm_use_slack = True
        model_kwargs = {'latent_size1': 128, 'latent_size2': 64}
    # else:
    #     raise ValueError(f'Unknown dataset: {dataset}')
    
    if task == 'weight_norm':
        data_source = load_data_norm

    batch_size = cfg.batch_size
    if task == 'cifar10':
        criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        dataloader_train, dataloader_val, dataloader_test, classes, class_ind = load_data_cifar10(device=device)
        features_train, sens_train, labels_train = next(iter(dataloader_train))
        model_fn = create_conv_model
        dataloader_val = dataloader_test
        model_kwargs = {}
    elif task == 'cifar100':
        criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        dataloader_train, dataloader_val, dataloader_test, classes, class_ind = load_data_cifar100(device=device)
        features_train, sens_train, labels_train = next(iter(dataloader_train))
        model_fn = create_resnet
        dataloader_val = dataloader_test
        model_kwargs = {}
        alm_use_slack = False
    else:
        criterion = torch.nn.functional.binary_cross_entropy_with_logits
        (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val), (features_test, sens_test, labels_test) = data_source(batch_size)
        features_val = features_test
        sens_val = sens_test
        labels_val = labels_test
        model_fn = create_model
        model_kwargs = {'input_shape': features_train.shape[-1], **model_kwargs}

    data_val = (features_val, sens_val, labels_val) if task not in ['cifar10', 'cifar100'] else dataloader_val
    
    ### construct the constraint ###
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


    # Track which algorithms are actually run
    algorithms_run = []

    #################################################################
    if adam_params is not None:
        # seed = 0
        # torch.manual_seed(seed)
        adam_history_train = []
        adam_history_val = []
        models = []
        for _ in range(n_runs):
            p = adam_params.pop('penalty', None)
            model, h_train, h_val = run_train(
                m=m,
                primal_opt=torch.optim.Adam,
                dual_opt=None,
                param_set=adam_params,
                data_train=(features_train, sens_train, labels_train),
                dataloader=dataloader_train,
                data_val=data_val,
                n_epochs=n_epochs,
                c_fn=constraint_fn,
                constraint_bound=constraint_bound,
                mode='torch',
                verbose=m < 30,
                constraints_to_eq=False,
                use_slack=False,
                fuse_loss_constraint=fuse_loss_constraint,
                reg_penalty=p,
                model_gen = model_fn,
                model_kwargs = model_kwargs,
                criterion=criterion,
                device=device
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
        algorithms_run.append('adam')
    else:
        print('\nADAM skipped (no parameters provided)\n')

    ##################################################################
    if pbm_params is not None:
        seed = 0
        torch.manual_seed(seed)
        models, pbm_history_train, pbm_history_val = [], [], []
        for _ in range(n_runs):
            model, h_train, h_val = run_train(
                m=m,
                primal_opt=torch.optim.Adam,
                dual_opt=PBM,
                param_set=pbm_params,
                data_train=(features_train, sens_train, labels_train),
                dataloader=dataloader_train,
                data_val=data_val,
                n_epochs=n_epochs,
                c_fn=constraint_fn,
                constraint_bound=constraint_bound,
                mode='hc',
                verbose=m < 30,
                constraints_to_eq=False,
                use_slack=False,
                fuse_loss_constraint=fuse_loss_constraint,
                model_gen = model_fn,
                model_kwargs = model_kwargs,
                criterion=criterion,
                device=device
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
        algorithms_run.append('pbm')

    if alm_params is not None:
        seed = 0
        torch.manual_seed(seed)
        models, alm_history_train, alm_history_val = [], [], []
        for _ in range(n_runs):
            model, h_train, h_val = run_train(
                m=m,
                primal_opt=torch.optim.Adam,
                dual_opt=ALM,
                param_set=alm_params,
                data_train=(features_train, sens_train, labels_train),
                dataloader=dataloader_train,
                data_val=data_val,
                n_epochs=n_epochs,
                c_fn=constraint_fn,
                constraint_bound=constraint_bound,
                mode='hc',
                verbose=m < 30,
                constraints_to_eq=True,
                use_slack=alm_use_slack,
                fuse_loss_constraint=fuse_loss_constraint,
                model_gen = model_fn,
                model_kwargs = model_kwargs,
                criterion=criterion,
                device=device
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
        algorithms_run.append('alm')
    else:
        print('\nALM skipped (no parameters provided)\n')

    if ssg_params is not None:
        seed = 0
        torch.manual_seed(seed)
        models, ssg_history_train, ssg_history_val = [], [], []
        for _ in range(n_runs):
            model, h_train, h_val = run_train(
                m=m,
                primal_opt=torch.optim.Adam,
                dual_opt=torch.optim.Adam,
                param_set=ssg_params,
                data_train=(features_train, sens_train, labels_train),
                dataloader=dataloader_train,
                data_val=data_val,
                n_epochs=n_epochs,
                c_fn=constraint_fn,
                constraint_bound=constraint_bound,
                mode='sw',
                verbose=m < 30,
                constraints_to_eq=False,
                use_slack=False,
                fuse_loss_constraint=fuse_loss_constraint,
                model_gen = model_fn,
                model_kwargs = model_kwargs,
                criterion=criterion,
                device=device
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
        algorithms_run.append('ssg')
    else:
        print('\nSSG skipped (no parameters provided)\n')

    ###### PLOT ######

    # algorithms_run = ['pbm_dimin', 'alm_max', 'adam']
    if len(algorithms_run) == 0:
        print('\nNo algorithms were run. Skipping plotting.\n')
        return

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

    # Build plotting data dynamically based on algorithms_run
    loss_means_train = []
    loss_stds_train = []
    cs_means_train = []
    cs_stds_train = []
    loss_means_val = []
    loss_stds_val = []
    cs_means_val = []
    cs_stds_val = []
    alg_names = []
    alg_display_names = {
        'adam': 'Adam',
        'pbm': 'SPBM',
        'alm': 'ALM',
        'ssg': 'SSG'
    }

    algorthithms_run = ['alm', 'pbm_dimin', 'adam']
    for alg in algorithms_run:
        print(f'plotting {alg}')
        train_file = f'{result_dir}/runs_{alg}_train.csv'
        val_file = f'{result_dir}/runs_{alg}_val.csv'
        
        if os.path.exists(train_file) and os.path.exists(val_file):
            _, loss_mean_train, loss_std_train, cs_mean_train, cs_std_train = read_prepare_data(train_file)
            _, loss_mean_val, loss_std_val, cs_mean_val, cs_std_val = read_prepare_data(val_file)
            
            loss_means_train.append(loss_mean_train)
            loss_stds_train.append(loss_std_train)
            cs_means_train.append(cs_mean_train.T)
            cs_stds_train.append(cs_std_train.T)
            loss_means_val.append(loss_mean_val)
            loss_stds_val.append(loss_std_val)
            cs_means_val.append(cs_mean_val.T)
            cs_stds_val.append(cs_std_val.T)
            alg_names.append(alg_display_names[alg])

    if len(alg_names) > 0:
        plot_losses_and_constraints_stochastic(
            loss_means_train,
            loss_stds_train,
            cs_means_train,
            cs_stds_train,
            constraint_bound,
            loss_means_val,
            loss_stds_val,
            cs_means_val if not task == 'weight_norm' else None,
            cs_stds_val if not task == 'weight_norm' else None,
            titles=alg_names,
            mode='train_test' if not task == 'weight_norm' else 'train',
            # plot_max_constraint=m > 5,
            save_path=result_dir + '/plot.png',
            # combine_algos = True
        )


@hydra.main(version_base=None, config_path="conf", config_name="benchmark")
def hydra_main(cfg: DictConfig):
    task_cfg = cfg.task
    data_cfg = cfg.data
    # task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    pbm_params = OmegaConf.to_container(task_cfg.get('pbm_params', None), resolve=True) if 'pbm_params' in task_cfg else None
    alm_params = OmegaConf.to_container(task_cfg.get('alm_params', None), resolve=True) if 'alm_params' in task_cfg else None
    ssg_params = OmegaConf.to_container(task_cfg.get('ssg_params', None), resolve=True) if 'ssg_params' in task_cfg else None
    adam_params = OmegaConf.to_container(task_cfg.get('adam_params', None), resolve=True) if 'adam_params' in task_cfg else None
    constraint_cfg = OmegaConf.to_container(task_cfg.constraint, resolve=True) if 'constraint' in task_cfg else None

    run_benchmark(
        data_cfg,
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

