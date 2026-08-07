from tqdm import tqdm
import pickle as pkl
import numpy as np
import copy
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from humancompatible.train.dual_optim import ALM, MoreauEnvelope, PBM
import hydra
from omegaconf import DictConfig
import pandas as pd
from hydra.core.hydra_config import HydraConfig
import json
import os

from pde_networks import set_model, u_Net_shallow_wide, u_Net_shallow_wide_resnet, u_Net_deep_narrow, u_Net_deep_narrow_resnet


def save_results(
    run_id: str,
    algorithm: str,
    seed: int,
    param_set: dict,
    total_loss: list,
    val_errs: list,
    test_errs: list,
    constraints: list,
    output_dir: str
):
    """Save individual run results to CSV files."""
    
    # Convert histories to dataframes
    loss_df = pd.DataFrame(total_loss)
    val_errs_df = pd.DataFrame(val_errs)
    test_errs_df = pd.DataFrame(test_errs)
    cs_df = pd.DataFrame(constraints)
    
    # Save with unique run_id
    prefix = os.path.join(output_dir, f"{run_id}")
    loss_df.to_csv(f"{prefix}_loss.csv")
    val_errs_df.to_csv(f"{prefix}_val_errs.csv")
    test_errs_df.to_csv(f"{prefix}_test_errs.csv")
    cs_df.to_csv(f"{prefix}_constraints.csv")

    
    # Save parameter configuration as JSON for later reference
    # param_set['seed'] = seed
    with open(f"{prefix}_config.json", 'w') as f:
        json.dump(dict(param_set), f, indent=2)
    
    print(f"Saved results to {output_dir}/")
    
    return {
        'files': f"{prefix}_X.csv",
    }



# Equation parameter
k, a1, a2 = 1, 1, 4

def q(data) :
    x, y = data[:,0].view(-1,1), data[:,1].view(-1,1)
    return -((a1*np.pi)**2)*torch.sin(a1*np.pi*x)*torch.sin(a2*np.pi*y) \
           -((a2*np.pi)**2)*torch.sin(a1*np.pi*x)*torch.sin(a2*np.pi*y) \
           +(k**2)*torch.sin(a1*np.pi*x)*torch.sin(a2*np.pi*y)

def analytic(data) :
    x, y = data[:,0].view(-1,1), data[:,1].view(-1,1)
    return torch.sin(a1*np.pi*x)*torch.sin(a2*np.pi*y)
    
def calculate_derivative(y, x, device='cuda') :
    return torch.autograd.grad(y, x, create_graph=True,\
                        grad_outputs=torch.ones(y.size()).to(device))[0]


def calculate_all_partial(u, x) :
    del_u = calculate_derivative(u, x)
    u_x, u_y = del_u[:,0], del_u[:,1]
    u_xx = calculate_derivative(u_x, x)[:,0]
    u_yy = calculate_derivative(u_y, x)[:,1]
    return u_xx.view(-1,1), u_yy.view(-1,1)


def train(u_model, beta, trainloader, bdry_data, val_test, optimizer, loss_f, dual_opt=None, device='cuda'):
    loss_list, loss_list1, loss_list2, val_list, test_list = [], [], [], [], []
    X_bdry, u_bdry = bdry_data
    X_val, y_val, X_test, y_test = val_test

    for i, (data,) in enumerate(trainloader) :
        u_model.train()
        optimizer.zero_grad()
        X_v = Variable(data, requires_grad=True).to(device)
        output = u_model(X_v)  
        output_bdry = u_model(X_bdry)
        
        u_xx, u_yy = calculate_all_partial(output, X_v)
        loss1 = loss_f(u_xx + u_yy + (k**2)*output - q(X_v), torch.zeros_like(output))
        constraint = loss_f(output_bdry, torch.zeros_like(output_bdry))
        
        # adam optimizer
        if dual_opt is None :
            loss = loss1 + beta*constraint
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        elif dual_opt is not None:
            threshold = 0.001 if isinstance(dual_opt, PBM) else 0
            constraint = constraint - threshold

            # compute the lagrangian value
            lagrangian = dual_opt.forward_update(loss1, constraint.unsqueeze(0))
            lagrangian.backward()
            optimizer.step()
            optimizer.zero_grad()

        
        u_model.eval()
        val_err = torch.linalg.norm((u_model(X_val) - y_val),2).item() / torch.linalg.norm(y_val,2).item()
        test_err = torch.linalg.norm((u_model(X_test) - y_test),2).item() / torch.linalg.norm(y_test,2).item()

        loss_list.append((loss1+constraint).item())
        loss_list1.append(loss1.item())
        loss_list2.append(constraint.item())
        val_list.append(val_err)
        test_list.append(test_err)
        
    return np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2), np.mean(val_list), np.mean(test_list)


def main_function(cfg: DictConfig, device):
    
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    
    model_name = cfg.model
    param_set = cfg.algorithm
    
    primal_params = {k.removeprefix('primal__'): v for k, v in param_set.items() if k.startswith('primal__')}
    dual_params = {k.removeprefix('dual__'): v for k, v in param_set.items() if k.startswith('dual__')}
    moreau_params = {k.removeprefix('moreau__'): v for k, v in param_set.items() if k.startswith('moreau__')}
    
    beta = param_set['beta']

    seed = cfg['seed']
    torch.manual_seed(seed)

    EPOCH = cfg['n_epochs']
    
    # Dataset Creation
    xmin, xmax = -1,1
    ymin, ymax = -1,1
    Nx, Ny = 51, 51
    X_train = torch.FloatTensor(np.mgrid[xmin:xmax:51j, ymin:ymax:51j].reshape(2, -1).T).to(device)

    # Boundary Conditions
    X_bdry = X_train[(X_train[:,0]==xmin) + (X_train[:,0]==xmax) + (X_train[:,1]==ymin) + (X_train[:,1]==ymax)]
    u_bdry = torch.zeros_like(X_bdry[:,0]).to(device).view(-1,1)
    
    X_test, y_test, X_val, y_val = torch.load('./Helmholtz_test', map_location=device)
    
    # take 1000 samples from the validation set
    idx = np.random.choice(X_val.shape[0], 1000, replace=False)
    X_val = X_val[idx]
    y_val = y_val[idx]
    
    # Make dataloader
    data_train = TensorDataset(X_train)
    train_loader = DataLoader(data_train, batch_size=1000, shuffle=False)
    
    total_loss, test_errs, val_errs, constraints = [], [], [], []
    u_model = set_model(model_name, device)
    
    if param_set.algorithm == 'adam':
        optimizer = torch.optim.Adam(params=u_model.parameters(), **primal_params)
    else:
        optimizer = MoreauEnvelope(torch.optim.Adam(params=u_model.parameters(), **primal_params), **moreau_params)

    ################################################################################
    if param_set.algorithm == 'adam':
        # best_model = copy.deepcopy(u_model)
        for t in tqdm(range(0, EPOCH)) :

            loss, loss1, loss2, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                        bdry_data=[X_bdry, u_bdry],\
                                                        val_test = [X_val, y_val, X_test, y_test],\
                                                        optimizer=optimizer, loss_f=nn.MSELoss())
            
            val_errs.append(val_err)
            test_errs.append(test_err)
            total_loss.append(loss)
            constraints.append(loss2)
            
            #Print Log
            if t%100 == 0 :
                print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                    (t, EPOCH, loss, loss1, loss2, val_err, test_err))

            # if np.argmin(val_errs) == t :
            #     best_model = copy.deepcopy(u_model)

    ################################################################################
    elif param_set.algorithm == 'pbm':
        
        dual = PBM(
            m=1,
            dual_range=(0.01, 10000.),
            **dual_params,
            device=device
        )

        for t in tqdm(range(0, EPOCH)) :

            loss, loss1, loss2, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                        bdry_data=[X_bdry, u_bdry],\
                                                        val_test = [X_val, y_val, X_test, y_test],\
                                                        optimizer=optimizer, loss_f=nn.MSELoss(),
                                                        dual_opt=dual)
            
            val_errs.append(val_err)
            test_errs.append(test_err)
            total_loss.append(loss)
            constraints.append(loss2)

            #Print Log
            if t%100 == 0 :
                print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                    (t, EPOCH, loss, loss1, loss2, val_err, test_err))

            # if np.argmin(val_errs_spbm) == t :
            #     best_model = copy.deepcopy(u_model)
    
    ################################################################################
    elif param_set.algorithm == 'alm':
        dual = ALM(
            m=1,
            dual_range=(0.01, 10000.),
            **dual_params,
            device=device   
        ) 

        for t in tqdm(range(0, EPOCH)) :

            loss, loss1, loss2, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                        bdry_data=[X_bdry, u_bdry],\
                                                        val_test = [X_val, y_val, X_test, y_test],\
                                                        optimizer=optimizer, loss_f=nn.MSELoss(),
                                                        dual_opt=dual)
            
            val_errs.append(val_err)
            test_errs.append(test_err)
            total_loss.append(loss)
            constraints.append(loss2)
            
            #Print Log
            if t%100 == 0 :
                print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                    (t, EPOCH, loss, loss1, loss2, val_err, test_err))

            # if np.argmin(val_errs_spbm) == t :
            #     best_model = copy.deepcopy(u_model)



    import hashlib
    param_hash = hashlib.md5(json.dumps(dict(param_set), sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"seed{seed}_{param_hash}"
    
    # Save results
    result_info = save_results(
        run_id=run_id,
        algorithm=param_set.algorithm,
        seed=seed,
        param_set=param_set,
        total_loss=total_loss,
        val_errs=val_errs,
        test_errs=test_errs,
        constraints=constraints,
        output_dir=output_dir
    )
    print(result_info)


@hydra.main(version_base=None, config_path="conf", config_name="helmholtz")
def main(cfg: DictConfig):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_function(cfg, device)
    
if __name__ == "__main__":
    main()
    