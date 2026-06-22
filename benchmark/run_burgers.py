from tqdm import tqdm
import pickle as pkl
import numpy as np
import copy
import argparse
import sys
import os                                                          # ADDED
import json                                                        # ADDED
import time as _time                                               # ADDED
import pandas as pd                                                # ADDED
from itertools import product                                      # ADDED
sys.path.append("..")

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from humancompatible.train.dual_optim import ALM, MoreauEnvelope, PBM

from networks import set_model, u_Net_shallow_wide, u_Net_shallow_wide_resnet, u_Net_deep_narrow, u_Net_deep_narrow_resnet

# Equation parameter

nu = 0.01/np.pi
THRESHOLD = 1e-4                                                   # ADDED (was inline in train)

# ── HP GRIDS (ADDED) ─────────────────────────────────────────────────────────
pbm_grid = [
    {"primal__lr": lr, "dual__penalty_mult": pm, "dual__penalty_update": pu,
     "dual__pbf": pbf, "dual__penalty_range": pr, "dual__gamma": g,
     "dual__delta": 1., "moreau__mu": mu}
    for (lr, pm, pu, pbf, pr, g, mu) in product(
        [0.001, 0.005, 0.01, 0.05], [0., 0.1, 0.2, 0.5], ["dimin_adapt"],
        ["quadratic_logarithmic"], [[1e-1, 1.], [1e-2, 1.]], [0.9], [2.])
]
alm_proj_grid = [
    {"primal__lr": lr, "dual__lr": dlr, "dual__penalty": pen, "moreau__mu": mu}
    for (lr, dlr, pen, mu) in product(
        [0.001, 0.005, 0.01, 0.05], [0.001, 0.005, 0.01, 0.05], [0., 1.], [2.])
]
alm_max_grid = [dict(d) for d in alm_proj_grid]                    # same grid; clamp differs
adam_grid = [{"primal__lr": lr, "beta": beta}
             for (lr, beta) in product([0.001, 0.005, 0.01, 0.05], [0.1, 1., 2., 5.])]


def calculate_derivative(y, x) :
    return torch.autograd.grad(y, x, create_graph=True,\
                        grad_outputs=torch.ones(y.size()).to(device))[0]

def calculate_all_partial(u, x) :
    del_u = calculate_derivative(u, x)
    u_t, u_x = del_u[:,0], del_u[:,1]
    u_xx = calculate_derivative(u_x, x)[:,1]
    return u_t.view(-1,1), u_x.view(-1,1), u_xx.view(-1,1)


def train(u_model, beta, trainloader, ini_bdry_data, val_test, optimizer, loss_f, dual_opt=None, clamp=False) :  # +clamp
    loss_list, loss_list1, loss_list2, loss_list3, val_list, test_list = [], [], [], [], [], []
    X_ini, u_ini, X_bdry, u_bdry = ini_bdry_data
    X_val, y_val, X_test, y_test = val_test

    for i, (data,) in enumerate(trainloader) :
        u_model.train()
        optimizer.zero_grad()
        X_v = Variable(data, requires_grad=True).to(device)
        output = u_model(X_v)  
        output_ini = u_model(X_ini)
        output_bdry = u_model(X_bdry)
        
        u_t, u_x, u_xx = calculate_all_partial(output, X_v)
        loss1 = loss_f(u_t + output*u_x - nu*u_xx, torch.zeros_like(u_t))
        loss2 = loss_f(output_ini-u_ini, torch.zeros_like(output_ini))
        loss3 = loss_f(output_bdry, torch.zeros_like(output_bdry))

        # unconstrained Adam
        if dual_opt is None :
            loss = loss1 + beta*loss2 + beta*loss3
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        elif dual_opt is not None:
            threshold = THRESHOLD                                  # CHANGED: was 1e-4 literal
            constraints = torch.stack([loss2, loss3], dim=0)
            constraints = constraints - threshold
            if clamp:                                              # ADDED: alm_max clamp
                constraints = torch.max(constraints, torch.zeros_like(constraints))

            # compute the lagrangian value
            lagrangian = dual_opt.forward_update(loss1, constraints)
            lagrangian.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        u_model.eval()
        val_err = torch.linalg.norm((u_model(X_val) - y_val),2).item() / torch.linalg.norm(y_val,2).item()
        test_err = torch.linalg.norm((u_model(X_test) - y_test),2).item() / torch.linalg.norm(y_test,2).item()

        loss_list.append((loss1+loss2+loss3).item())
        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())
        loss_list3.append(loss3.item())
        val_list.append(val_err)
        test_list.append(test_err)
        
    return np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2),\
           np.mean(loss_list3), np.mean(val_list), np.mean(test_list)


# ── saving helpers (ADDED) ───────────────────────────────────────────────────
def save_method(result_dir, method, histories, grid):
    """Stack per-config histories into runs_{method}.csv (config,epoch,time,loss,
    c_0,c_1,val,test) + grid_{method}.csv + best_{method}.json (val-selected)."""
    df = pd.concat([pd.DataFrame(h) for h in histories],
                   keys=range(len(histories)), names=["config", "row"]
                   ).reset_index(level="config").reset_index(drop=True)
    df.to_csv(f"{result_dir}/runs_{method}.csv", index=False)
    pd.DataFrame(grid).to_csv(f"{result_dir}/grid_{method}.csv")

    rows = []
    for i, h in enumerate(histories):
        last = h[-1]
        max_c = max(last[k] for k in last if k.startswith("c_"))
        rows.append((i, last["val"], max_c, max_c <= THRESHOLD))
    feas = [r for r in rows if r[3]]
    pool = feas if feas else rows
    best = min(pool, key=lambda r: r[1])
    json.dump({"method": method, "best_idx": int(best[0]),
               "val_loss": float(best[1]), "val_max_c": float(best[2]),
               "n_configs": len(rows), "n_feasible": len(feas)},
              open(f"{result_dir}/best_{method}.json", "w"), indent=2)


def main_function(model_name, beta, lr, EPOCH, device, seed) :     # +seed
    
    result_dir = f"results/burgers_pinn{seed}"                     # ADDED
    os.makedirs(result_dir, exist_ok=True)                         # ADDED

    # Dataset Creation
    tmin, tmax = 0, 1
    xmin, xmax = -1,1
    Ns, Nx = 51, 51
    X_train = torch.FloatTensor(np.mgrid[tmin:tmax:51j, xmin:xmax:51j].reshape(2, -1).T).to(device)

    # Initial Conditions
    X_ini = X_train[X_train[:,0]==tmin]
    u_ini = -torch.sin(np.pi*X_ini[:,1].view(-1,1))
                                
    # Boundary Conditions
    X_bdry = X_train[(X_train[:,1]==xmin) + (X_train[:,1]==xmax)]
    u_bdry = torch.zeros_like(X_bdry[:,0]).to(device).view(-1,1)
    
    # Validation & Test Set
    X_test, y_test, X_val, y_val= torch.load('./PDEs/Viscous_Burgers/Burgers_test', map_location=device)

    # take 1000 samples from the validation set
    idx = np.random.choice(X_val.shape[0], 1000, replace=False)
    X_val = X_val[idx]
    y_val = y_val[idx]

    # Make dataloader
    data_train = TensorDataset(X_train)
    train_loader = DataLoader(data_train, batch_size=10000, shuffle=False)

    ini_bdry = [X_ini, u_ini, X_bdry, u_bdry]                      # ADDED (convenience)
    val_test = [X_val, y_val, X_test, y_test]                      # ADDED

    # helper: run ONE config's full training, return history list (ADDED)
    def run_config(params, dual_ctor, clamp=False):
        primal = {k.removeprefix("primal__"): v for k, v in params.items() if k.startswith("primal__")}
        moreau = {k.removeprefix("moreau__"): v for k, v in params.items() if k.startswith("moreau__")}
        b = params.get("beta", beta)
        torch.manual_seed(0)                                       # same init per config (as old code reseeded per block)
        u_model = set_model(model_name, device)
        if dual_ctor is None:
            optimizer = torch.optim.Adam([{'params': u_model.parameters()}], **primal)
            dual = None
        else:
            optimizer = MoreauEnvelope(torch.optim.Adam([{'params': u_model.parameters()}], **primal), **moreau)
            dual = dual_ctor(params)
        history = []
        t0 = _time.time()
        for t in range(0, EPOCH):
            loss, loss1, loss2, loss3, val_err, test_err = train(
                u_model, b, trainloader=train_loader, ini_bdry_data=ini_bdry,
                val_test=val_test, optimizer=optimizer, loss_f=nn.MSELoss(),
                dual_opt=dual, clamp=clamp)
            history.append({"epoch": t, "time": _time.time() - t0, "loss": loss1,
                            "c_0": loss2, "c_1": loss3, "val": val_err, "test": test_err})
            if t % 100 == 0:
                print("%s/%s | loss: %06.6f | c: %06.6f | val: %06.6f | test: %06.6f " %
                      (t, EPOCH, loss1, loss2 + loss3, val_err, test_err))
        return history

    # dual constructors built FROM params (same classes/args as the old hardcoded blocks)
    def make_pbm(params):
        dp = {k.removeprefix("dual__"): v for k, v in params.items() if k.startswith("dual__")}
        return PBM(m=2, dual_range=(0.01, 100.), **dp, device=device)
    def make_alm(params):
        dp = {k.removeprefix("dual__"): v for k, v in params.items() if k.startswith("dual__")}
        return ALM(m=2, **dp, device=device)

    # ===== ADAM =====
    histories = [run_config(p, None) for p in tqdm(adam_grid, desc="adam")]
    save_method(result_dir, "adam", histories, adam_grid)

    # ===== SPBM (PBM) =====
    histories = [run_config(p, make_pbm) for p in tqdm(pbm_grid, desc="pbm")]
    save_method(result_dir, "pbm", histories, pbm_grid)

    # ===== ALM projection (raw signed constraints) =====
    histories = [run_config(p, make_alm, clamp=False) for p in tqdm(alm_proj_grid, desc="alm_proj")]
    save_method(result_dir, "alm_proj", histories, alm_proj_grid)

    # ===== ALM max (clamped constraints) =====
    histories = [run_config(p, make_alm, clamp=True) for p in tqdm(alm_max_grid, desc="alm_max")]
    save_method(result_dir, "alm_max", histories, alm_max_grid)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deep_narrow', help='Specify the model. Choose one of [deep_narrow, shallow_wide, deep_narrow_resent, shallow_wide_resnet].')
    parser.add_argument('--beta', default=1, type=float, help='Penalty parameter beta')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--EPOCH', default=1000, type=int, help='Number of training EPOCH')
    parser.add_argument('--ordinal', default=0, type=int, help='Specify the cuda device ordinal.')
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.ordinal) if torch.cuda.is_available() else "cpu")

    seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))           # ADDED: seed from Slurm array
    torch.manual_seed(seed)                                        # ADDED
    
    main_function(args.model, args.beta, args.lr, args.EPOCH, device, seed)