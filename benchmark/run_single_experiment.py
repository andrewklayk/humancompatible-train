"""
Single experiment runner for Hydra multirun sweeps.

This script runs a single hyperparameter combination and saves results as separate CSVs
for efficient parallel sweep execution via Hydra's multirun mode.

Usage (sequential):
    python run_single_experiment.py algorithm=alm_max seed=0

Usage (parallel multirun):
    python run_single_experiment.py --multirun \
        seed=0,1,2 \
        params.primal_lr=0.001,0.005,0.01,0.05 \
        params.dual_lr=0.001,0.005,0.01,0.05 \
        params.moreau_mu=0,0.5,1,2,4 \
        hydra.launcher=joblib hydra.launcher.n_jobs=-1
"""

import logging
from utils import *
from utils import create_model, create_conv_model, create_resnet
from humancompatible.train.fairness.utils import BalancedBatchSampler
import torch
from _data_sources import load_data_FT, load_data_DUTCH, load_data_norm, load_data_cifar10, load_data_cifar100
import pandas as pd
import os
import importlib
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from torch.nn import functional as F

from humancompatible.train.dual_optim import ALM, PBM, MoreauEnvelope
from constraints import loss_per_group, posrate_per_group, weight_constraint, posrate_fairret_constraint


def runs_to_df(runs):
    """Convert run histories to dataframe."""
    return pd.concat([pd.DataFrame(h).set_index('epoch') for h in runs], keys=range(len(runs)))


def save_results(
    run_id: str,
    algorithm: str,
    seed: int,
    param_set: dict,
    train_hist: list,
    val_hist: list,
    output_dir: str
):
    """Save individual run results to CSV files."""
    
    # Convert histories to dataframes
    train_df = runs_to_df(train_hist)
    val_df = runs_to_df(val_hist)
    
    # Save with unique run_id
    prefix = os.path.join(output_dir, f"{run_id}")
    train_df.to_csv(f"{prefix}_train.csv")
    val_df.to_csv(f"{prefix}_val.csv")
    
    # Save parameter configuration as JSON for later reference
    # param_set['seed'] = seed
    with open(f"{prefix}_config.json", 'w') as f:
        json.dump(dict(param_set), f, indent=2)
    
    print(f"Saved results to {output_dir}/")
    
    return {
        'train_file': f"{prefix}_train.csv",
        'val_file': f"{prefix}_val.csv",
        'config_file': f"{prefix}_config.json"
    }


def run_single_config(
    data_cfg: dict,
    task_cfg: DictConfig,
    algorithm: str,
    seed: int,
    param_set: dict,
    n_epochs: int,
    constraint_cfg: dict,
    device: str
):
    """Run a single hyperparameter configuration."""
    
    dataset = data_cfg['name']
    task = task_cfg.task
    
    # Set random seed
    torch.manual_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm} | Seed: {seed}")
    print(f"Params: {param_set}")
    print(f"{'='*60}\n")
    
    ### Load data ###
    if dataset == 'folktables':
        data_source = lambda batch_size: load_data_FT(batch_size, device, **data_cfg['kwargs'])
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

    # Determine constraint function and parameters based on task
    fuse_loss_constraint = False
    if constraint_cfg is not None:
        c = importlib.import_module("constraint_meta").__dict__.get(constraint_cfg['name'])
        
        if constraint_cfg['name'].startswith('Fairret'):
            statistic = importlib.import_module("fairret.statistic").__dict__.get(constraint_cfg['statistic'])
            statistic = statistic()
            if constraint_cfg['name'] == 'FairretAgg':
                fair_loss = importlib.import_module("fairret.loss").__dict__.get(constraint_cfg['loss'])
                c = c(loss=fair_loss(statistic), **constraint_cfg.get('kwargs', {}))
            else:
                c = c(statistic=statistic, **constraint_cfg.get('kwargs', {}))
        
        elif constraint_cfg['name'].startswith('Loss'):
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

    # Run single training with param_set
    if algorithm == 'adam':
        primal_opt = torch.optim.Adam
        dual_opt = None
        mode = 'torch'
        constraints_to_eq = False
        use_slack = False
        
    elif algorithm == 'pbm_adapt':
        primal_opt = torch.optim.Adam
        dual_opt = PBM
        mode = 'hc'
        constraints_to_eq = False
        use_slack = False
        
    elif algorithm == 'pbm_dimin':
        primal_opt = torch.optim.Adam
        dual_opt = PBM
        mode = 'hc'
        constraints_to_eq = False
        use_slack = False
        
    elif algorithm == 'alm_max':
        primal_opt = torch.optim.Adam
        dual_opt = ALM
        mode = 'hc'
        constraints_to_eq = True
        use_slack = False
        
    elif algorithm == 'alm_slack':
        primal_opt = torch.optim.Adam
        dual_opt = ALM
        mode = 'hc'
        constraints_to_eq = True
        use_slack = True
        
    elif algorithm == 'ssg':
        primal_opt = torch.optim.Adam
        dual_opt = torch.optim.Adam
        mode = 'sw'
        constraints_to_eq = False
        use_slack = False
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model, train_history, val_history = run_train(
        m=m,
        primal_opt=primal_opt,
        dual_opt=dual_opt,
        param_set=param_set,
        data_train=(features_train, sens_train, labels_train),
        dataloader=dataloader_train,
        data_val=data_val,
        n_epochs=n_epochs,
        c_fn=constraint_fn,
        constraint_bound=constraint_bound,
        mode=mode,
        verbose=False,
        constraints_to_eq=constraints_to_eq,
        use_slack=use_slack,
        constraint_tol=0.,
        fuse_loss_constraint=fuse_loss_constraint,
        model_gen=create_model_fn,
        model_kwargs=model_kwargs,
        device=device,
        criterion=criterion
    )
    
    return [train_history], [val_history]


@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def main(cfg: DictConfig):
    """Run single experiment and save results."""
    
    # Get Hydra's output directory
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except (RuntimeError, AttributeError):
        # Fallback if HydraConfig not available
        output_dir = 'multirun_results'
    
    # Create results directory
    print(hydra_cfg)
    results_dir = output_dir #os.path.join(output_dir, 'results')
    #os.makedirs(results_dir, exist_ok=True)
    
    # Extract configuration
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    task_cfg = cfg.task
    opt_cfg = cfg.algorithm
    algorithm = opt_cfg['algorithm']
    seed = cfg.seed
    n_epochs = cfg.n_epochs
    constraint_cfg = OmegaConf.to_container(task_cfg.constraint, resolve=True) if 'constraint' in task_cfg else None
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    
    
    # Run single configuration
    train_hist, val_hist = run_single_config(
        data_cfg=data_cfg,
        task_cfg=task_cfg,
        algorithm=algorithm,
        seed=seed,
        param_set=opt_cfg,
        n_epochs=n_epochs,
        constraint_cfg=constraint_cfg,
        device=device
    )
    
    # Generate run ID from hyperparameters (hash of params for uniqueness)
    import hashlib
    param_hash = hashlib.md5(json.dumps(dict(opt_cfg), sort_keys=True).encode()).hexdigest()[:12]
    # logging.info(type(opt_cfg))
    # logging.info(type(dict(opt_cfg)))
    # print(type(opt_cfg))
    # print(type(dict(opt_cfg)))
    run_id = f"seed{seed}_{param_hash}"
    
    # Save results
    result_info = save_results(
        run_id=run_id,
        algorithm=algorithm,
        seed=seed,
        param_set=opt_cfg,
        train_hist=train_hist,
        val_hist=val_hist,
        output_dir=results_dir
    )
    
    # Return minimal loss for Hydra logging
    val_df = runs_to_df(val_hist)
    min_loss = val_df['loss'].min()
    
    print(f"\nRun ID: {run_id}")
    print(f"Min validation loss: {min_loss}")
    
    return min_loss


if __name__ == "__main__":
    main()
