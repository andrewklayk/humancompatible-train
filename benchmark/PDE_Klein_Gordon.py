from tqdm import tqdm
import pickle as pkl
import numpy as np
import copy
import sys
sys.path.append("..")

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

# Equation parameter
k=3
alpha, delta, gamma =  -1, 0, 1


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


def analytic(bdry) :
    t, x = bdry[:,0].view(-1,1), bdry[:,1].view(-1,1)
    return x*torch.cos(5*np.pi*t) + ((x*t)**3)

def u_tt(data) :
    t, x = data[:,0].view(-1,1), data[:,1].view(-1,1)
    return -((5*np.pi)**2)*x*torch.cos(5*np.pi*t) + 6*(x**3)*t

def u_xx(data) :
    t, x = data[:,0].view(-1,1), data[:,1].view(-1,1)
    return 6*x*(t**3)

def u3(data) :
    t, x = data[:,0].view(-1,1), data[:,1].view(-1,1)
    return (x*torch.cos(5*np.pi*t) + ((x*t)**3))**3

def u(data) :
    t, x = data[:,0].view(-1,1), data[:,1].view(-1,1)
    return x*torch.cos(5*np.pi*t) + ((x*t)**3)

def f(data) :
    return u_tt(data) + alpha*u_xx(data) + delta*u(data) + gamma*u3(data)

def calculate_derivative(y, x, device='cuda') :
    return torch.autograd.grad(y, x, create_graph=True,\
                        grad_outputs=torch.ones(y.size()).to(device))[0]

def calculate_all_partial(u, x, device='cuda') :
    del_u = calculate_derivative(u, x, device=device)
    u_t, u_x = del_u[:,0], del_u[:,1]
    u_tt = calculate_derivative(u_t, x, device=device)[:,0]
    u_xx = calculate_derivative(u_x, x, device=device)[:,1]
    return u_tt.view(-1,1), u_xx.view(-1,1)


def train(u_model, beta, trainloader, ini_bdry_data, val_test, optimizer, loss_f, dual_opt=None, device='cuda') :
    loss_list, loss_list1, loss_list2, loss_list3, loss_list4, val_list, test_list = [], [], [], [], [], [], []
    X_ini, u_ini, u_ini_t, X_bdry, u_bdry = ini_bdry_data
    X_val, y_val, X_test, y_test = val_test

    for i, (data,) in enumerate(trainloader) :
        u_model.train()
        optimizer.zero_grad()
        X_v = Variable(data, requires_grad=True).to(device)
        output = u_model(X_v)  
        output_ini = u_model(X_ini)
        output_ini_t = calculate_derivative(output_ini, X_ini, device=device)[:,0].view(-1,1)
        output_bdry = u_model(X_bdry)
        
        u_tt, u_xx = calculate_all_partial(output, X_v, device=device)
        loss1 = loss_f(u_tt + alpha*u_xx + delta*output + gamma*(output**k) - f(X_v), torch.zeros_like(output))
        loss2 = loss_f(output_ini, u_ini) 
        loss3 = loss_f(output_ini_t, u_ini_t)
        loss4 = loss_f(output_bdry, u_bdry)
        
        if dual_opt is None:
            loss = loss1 + beta*loss2 + beta*loss3 + beta*loss4
            loss.backward()
            optimizer.step()
        elif dual_opt is not None:
            threshold = 0.1 if isinstance(dual_opt, PBM) else 0
            constraints = torch.stack([loss2, loss3, loss4], dim=0)
            constraints = constraints - threshold

            # compute the lagrangian value
            lagrangian = dual_opt.forward_update(loss1, constraints)
            lagrangian.backward()
            optimizer.step()
            optimizer.zero_grad()

        u_model.eval()
        val_err = torch.linalg.norm((u_model(X_val) - y_val),2).item() / torch.linalg.norm(y_val,2).item()
        test_err = torch.linalg.norm((u_model(X_test) - y_test),2).item() / torch.linalg.norm(y_test,2).item()

        loss_list.append((loss1+loss2+loss3+loss4).item())
        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())
        loss_list3.append(loss3.item())
        loss_list4.append(loss4.item())
        val_list.append(val_err)
        test_list.append(test_err)
        
    return np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2),\
           np.mean(loss_list3), np.mean(loss_list4), np.mean(val_list), np.mean(test_list)


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
    tmin, tmax = 0, 1
    xmin, xmax = 0, 1
    Nt, Nx = 51, 51
    X_train = torch.FloatTensor(np.mgrid[tmin:tmax:51j, xmin:xmax:51j].reshape(2, -1).T).to(device)

    # Initial Conditions
    X_ini = X_train[X_train[:,0]==tmin].to(device)
    u_ini = X_ini[:,1].view(-1,1)
    u_ini_t = torch.zeros_like(u_ini)
                                
    # Boundary Conditions
    X_bdry = X_train[(X_train[:,1]==xmin) + (X_train[:,1]==xmax)]
    u_bdry = analytic(X_bdry)
    
    # Validation & Test Set
    X_test, y_test, X_val, y_val = torch.load('./Klein_Gordon_test', map_location=device)
    
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
        for t in tqdm(range(0, EPOCH)) :

            loss, loss1, loss2, loss3, loss4, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                        ini_bdry_data=[X_ini, u_ini, u_ini_t, X_bdry, u_bdry],\
                                                        val_test = [X_val, y_val, X_test, y_test],\
                                                        optimizer=optimizer, loss_f=nn.MSELoss(),
                                                        dual_opt=None, device=device)
            
            val_errs.append(val_err)
            test_errs.append(test_err)
            total_loss.append(loss)
            constraints.append([loss2, loss3, loss4])

            #Print Log
            if t%100 == 0 :
                print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                    (t, EPOCH, loss, loss1, loss2+loss3+loss4, val_err, test_err))

    ################################################################################
    elif param_set.algorithm == 'pbm':
        
        dual = PBM(
            m=3,
            dual_range=(0.01, 100.),
            **dual_params,
            device=device
        )

        for t in tqdm(range(0, EPOCH)) :

            loss, loss1, loss2, loss3, loss4, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                        ini_bdry_data=[X_ini, u_ini, u_ini_t, X_bdry, u_bdry],\
                                                        val_test = [X_val, y_val, X_test, y_test],\
                                                        optimizer=optimizer, loss_f=nn.MSELoss(),
                                                        dual_opt=dual, device=device)
            
            val_errs.append(val_err)
            test_errs.append(test_err)
            total_loss.append(loss)
            constraints.append([loss2, loss3, loss4])

            #Print Log
            if t%100 == 0 :
                print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                    (t, EPOCH, loss, loss1, loss2+loss3+loss4, val_err, test_err))

    ################################################################################
    elif param_set.algorithm == 'alm':
        dual = ALM(
            m=3,
            dual_range=(0.01, 100.),
            **dual_params,
            device=device 
        ) 

        for t in tqdm(range(0, EPOCH)) :

            loss, loss1, loss2, loss3, loss4, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                        ini_bdry_data=[X_ini, u_ini, u_ini_t, X_bdry, u_bdry],\
                                                        val_test = [X_val, y_val, X_test, y_test],\
                                                        optimizer=optimizer, loss_f=nn.MSELoss(),
                                                        dual_opt=dual, device=device)
            
            val_errs.append(val_err)
            test_errs.append(test_err)
            total_loss.append(loss)
            constraints.append([loss2, loss3, loss4])

            #Print Log
            if t%100 == 0 :
                print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                    (t, EPOCH, loss, loss1, loss2+loss3+loss4, val_err, test_err))

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


@hydra.main(version_base=None, config_path="conf", config_name="klein_gordon")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_function(cfg, device)
    
if __name__ == "__main__":
    main()
    