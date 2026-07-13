from tqdm import tqdm
import pickle as pkl
import numpy as np
import copy
import sys
import os                                                          # ADDED
import json                                                        # ADDED
import time as _time                                               # ADDED
import pandas as pd                                                # ADDED
from itertools import product                                      # ADDED
import hydra                                                       # ADDED (Hydra entry)
from omegaconf import DictConfig                                   # ADDED
sys.path.append("..")
 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from humancompatible.train.dual_optim import ALM, MoreauEnvelope, PBM
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
 
from networks import set_model, u_Net_shallow_wide, u_Net_shallow_wide_resnet, u_Net_deep_narrow, u_Net_deep_narrow_resnet
# Equation parameter

nu = 0.01/np.pi
THRESHOLD = 1e-4                                                   # ADDED (was inline in train)
device='cuda'


# ── HP GRIDS (ADDED) ─────────────────────────────────────────────────────────
pbm_grid = [
    {"primal__lr": lr, "dual__penalty_mult": pm, "dual__penalty_update": pu,
     "dual__pbf": pbf, "dual__penalty_range": pr, "dual__gamma": g,
     "dual__delta": 1., "moreau__mu": mu,
    "dual__primal_update_process_length": primal_update_process_length,
    "dual__gamma_annealing": gamma_annealing, "dual__penalty_annealing": penalty_annealing,
    "dual__logscaled_dual_update": logscaled_dual_update, "dual__logscaled_dual_step_size": logscaled_dual_step_size}
    for (lr, pm, pu, pbf, pr, g, mu, primal_update_process_length, gamma_annealing, penalty_annealing, logscaled_dual_update, logscaled_dual_step_size) 
    in product(
        [0.001, 0.005, 0.01, 0.02, 0.05], [0.8, 0.9, 0.99, 0.999, 1.0], ["dimin_adapt"],
        ["quadratic_logarithmic"], [[1e-2, 1.]], [0.9, 0.99], [0., 1., 2.], 
        [1], [True], [True], [False], [None])
]

# ensure the primal update process length is the same for both moreau and dual
for arr_dict in pbm_grid:
    arr_dict["moreau__primal_update_process_length"] = arr_dict["dual__primal_update_process_length"]
    
pbm_logascaled_grid = [
    {"primal__lr": lr, "dual__penalty_mult": pm, "dual__penalty_update": pu,
     "dual__pbf": pbf, "dual__penalty_range": pr, "dual__gamma": g,
     "dual__delta": 1., "moreau__mu": mu,
    "dual__primal_update_process_length": primal_update_process_length,
    "dual__gamma_annealing": gamma_annealing, "dual__penalty_annealing": penalty_annealing,
    "dual__logscaled_dual_update": logscaled_dual_update, "dual__logscaled_dual_step_size": logscaled_dual_step_size}
    for (lr, pm, pu, pbf, pr, g, mu, primal_update_process_length, gamma_annealing, penalty_annealing, logscaled_dual_update, logscaled_dual_step_size) 
    in product(
        [0.001, 0.005, 0.01, 0.02, 0.05], [0., 0.1, 0.5, 0.9, 1.0], ["dimin_adapt"],
        ["quadratic_logarithmic"], [[1e-1, 1.], [1e-2, 1.]], [None], [0., 1., 2.], 
        [1], [None], [True], [True], [0.1, 0.01, 0.5])
]

alm_proj_grid = [
    {"primal__lr": lr, "dual__lr": dlr, "dual__penalty": pen, "moreau__mu": mu, 
            "dual__is_ineq": True}
    for (lr, dlr, pen, mu) in product(
        [0.001, 0.005, 0.01, 0.02, 0.05], [0.001, 0.005, 0.01, 0.02, 0.05], [0., 1.], [0., 1., 2.])
]
alm_max_grid = [
    {"primal__lr": lr, "dual__lr": dlr, "dual__penalty": pen, "moreau__mu": mu, 
            "dual__is_ineq": False}
    for (lr, dlr, pen, mu) in product(
        [0.001, 0.005, 0.01, 0.02, 0.05], [0.001, 0.005, 0.01, 0.02, 0.05], [0., 1.], [0., 1., 2.])
]
ssg_grid = [{"primal__lr": lr, "dual__lr": dlr, "moreau__mu": mu}  # ADDED: SSw grid (matches fairness)
            for (lr, dlr, mu) in product(
                [0.001, 0.005, 0.01, 0.02, 0.05], [0.001, 0.005, 0.01, 0.02, 0.05], [0., 1., 2.])]
adam_grid = [{"primal__lr": lr, "beta": beta}
             for (lr, beta) in product([0.001, 0.005, 0.01, 0.02, 0.05], [0.5, 1., 2., 5., 10.])]


def calculate_derivative(y, x) :
    return torch.autograd.grad(y, x, create_graph=True,\
                        grad_outputs=torch.ones(y.size()).to(device))[0]

def calculate_all_partial(u, x) :
    del_u = calculate_derivative(u, x)
    u_t, u_x = del_u[:,0], del_u[:,1]
    u_xx = calculate_derivative(u_x, x)[:,1]
    return u_t.view(-1,1), u_x.view(-1,1), u_xx.view(-1,1)


def train(u_model, beta, trainloader, ini_bdry_data, val_test, optimizer, loss_f, dual_opt=None, clamp=False, mode='lagrangian',
           sw_dual=None, constraint_tol=1.) :  # +clamp, +mode, +sw_dual, +constraint_tol
    loss_list, loss_list1, loss_list2, loss_list3, val_list, test_list = [], [], [], [], [], []
    kkt_list = []  # ADDED: per-batch KKT dicts (full-batch loader -> whole train set)
    X_ini, u_ini, X_bdry, u_bdry = ini_bdry_data
    X_val, y_val, X_test, y_test = val_test
    
    n_epochs = len(trainloader)  # ADDED: for linear decay of constraint_tol in SSw

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
        if dual_opt is None and mode != 'sw':
            loss = loss1 + beta*loss2 + beta*loss3
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        elif mode == 'sw':                                         # ADDED: SSw switching subgradient (matches train_loop_sw)
            # Switching subgradient (Huang & Lin): if the most-violated constraint
            # exceeds constraint_tol, take a step with the DUAL optimizer on that
            # constraint; otherwise take a step with the PRIMAL optimizer on the
            # objective. Two separate optimizers (separate LRs), as in utils.py.
            cons = torch.stack([loss2, loss3], dim=0) - THRESHOLD
            max_c = cons.max()
            optimizer.zero_grad()
            sw_dual.zero_grad()
            if max_c > constraint_tol:                             # infeasible -> dual step on constraint
                max_c.backward()
                sw_dual.step()
            else:                                                  # feasible -> primal step on objective
                loss1.backward()
                optimizer.step()
            optimizer.zero_grad()
            sw_dual.zero_grad()
            constraint_tol = constraint_tol * (1 - i / n_epochs)  # linear decay to 0

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
        # train_L = torch.linalg.norm((output - u_model(X_v)),2).item()

        # ── KKT on the entire train set (post-step, current weights) ──── ADDED
        # min f=loss1  s.t.  g=[loss2,loss3]-THRESHOLD <= 0 ; L = f + λᵀg.
        X_k = Variable(data, requires_grad=True).to(device)
        u_k = u_model(X_k)
        ut_k, ux_k, uxx_k = calculate_all_partial(u_k, X_k)
        f = loss_f(ut_k + u_k*ux_k - nu*uxx_k, torch.zeros_like(ut_k))
        out_ini_k, out_bdry_k = u_model(X_ini), u_model(X_bdry)
        g = torch.stack([loss_f(out_ini_k - u_ini, torch.zeros_like(out_ini_k)),
                         loss_f(out_bdry_k, torch.zeros_like(out_bdry_k))]) - THRESHOLD
        lam = dual_opt.duals.detach().reshape(-1) if dual_opt is not None \
              else beta * torch.ones(2, device=device)
        L = f + lam @ g
        params = [p for p in u_model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(L, params, allow_unused=True)
        grad_norm = torch.sqrt(sum((gr**2).sum() for gr in grads if gr is not None)).item()
        max_viol = g.max().item()
        compl = (lam * g).abs().sum().item()
        kkt_list.append({"kkt_r": grad_norm + max(0., max_viol) + compl,
                         "kkt_grad": grad_norm, "kkt_viol": max_viol, "kkt_compl": compl,
                         "lambda_0": lam[0].item(), "lambda_1": lam[1].item()})


        loss_list.append((loss1+loss2+loss3).item())
        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())
        loss_list3.append(loss3.item())
        val_list.append(val_err)
        test_list.append(test_err)
        
    kkt = {k: float(np.mean([d[k] for d in kkt_list])) for k in kkt_list[0]}  # ADDED
    return np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2),\
           np.mean(loss_list3), np.mean(val_list), np.mean(test_list), kkt


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


def main_function(model_name, beta, lr, EPOCH, device, seed, cfg) :     # +seed
    
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
    X_test, y_test, X_val, y_val= torch.load('../PDEs/Viscous_Burgers/Burgers_test', map_location=device)

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
    def run_config(params, dual_ctor, clamp=False, mode='lagrangian'):
        primal = {"weight_decay": 0.01, **{k.removeprefix("primal__"): v for k, v in params.items() if k.startswith("primal__")}}
        dual_p = {"weight_decay": 0.01, **{k.removeprefix("dual__"): v for k, v in params.items() if k.startswith("dual__")}}
        moreau = {k.removeprefix("moreau__"): v for k, v in params.items() if k.startswith("moreau__")}
        b = params.get("beta", beta)
        torch.manual_seed(seed)                                       # same init per config (as old code reseeded per block)
        u_model = set_model(model_name, device)
        sw_dual = None
        if mode == 'sw':                                           # SSw: both primal and dual are Moreau-wrapped Adam on model params (separate LRs)
            optimizer = MoreauEnvelope(torch.optim.Adam([{'params': u_model.parameters()}], **primal), **moreau)
            sw_dual = MoreauEnvelope(torch.optim.Adam([{'params': u_model.parameters()}], **dual_p), **moreau)   # matches train_loop_sw
            dual = None
        elif dual_ctor is None:
            optimizer = torch.optim.Adam([{'params': u_model.parameters()}], **primal)
            dual = None
        else:
            optimizer = MoreauEnvelope(torch.optim.Adam([{'params': u_model.parameters()}], **primal), **moreau)
            dual = dual_ctor(params)
        history = []
        t0 = _time.time()

        # setup scheduler
        total_steps  = EPOCH
        warmup_steps = int(0.05 * total_steps)

        sched = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
            ],
            milestones=[warmup_steps],
        )

        for t in range(0, EPOCH):
            loss, loss1, loss2, loss3, val_err, test_err, kkt = train(
                u_model, b, trainloader=train_loader, ini_bdry_data=ini_bdry,
                val_test=val_test, optimizer=optimizer, loss_f=nn.MSELoss(),
                dual_opt=dual, clamp=clamp, mode=mode, sw_dual=sw_dual)
            history.append({"epoch": t, "time": _time.time() - t0, "loss": loss1,
                            "c_0": loss2, "c_1": loss3, "val": val_err, "test": test_err,
                            **kkt})

            # step the lr scheduler
            sched.step()

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
    if 'adam' in cfg.algorithms:
        histories = [run_config(p, None) for p in tqdm(adam_grid, desc="adam")]
        save_method(result_dir, "adam", histories, adam_grid)

    # ===== SPBM (PBM) Log  =====
    # ensure the pbm has the size of the epoch (for gamma annealing)
    if 'pbm_logscaled' in cfg.algorithms:
        for arr_dict in pbm_logascaled_grid:   
            arr_dict["dual__epoch_length"] = len(train_loader)

        histories = [run_config(p, make_pbm) for p in tqdm(pbm_logascaled_grid, desc="pbm")]
        save_method(result_dir, "pbm_logscaled", histories, pbm_logascaled_grid)

    # ===== SPBM (PBM) =====
    # ensure the pbm has the size of the epoch (for gamma annealing)
    if 'pbm' in cfg.algorithms:
        for arr_dict in pbm_grid:   
            arr_dict["dual__epoch_length"] = 60

        histories = [run_config(p, make_pbm) for p in tqdm(pbm_grid, desc="pbm")]
        save_method(result_dir, "pbm", histories, pbm_grid)

    # ===== ALM projection (raw signed constraints) =====
    if 'alm_proj' in cfg.algorithms:
        histories = [run_config(p, make_alm, clamp=False) for p in tqdm(alm_proj_grid, desc="alm_proj")]
        save_method(result_dir, "alm_proj", histories, alm_proj_grid)

    # ===== ALM max (clamped constraints) =====
    if 'alm_max' in cfg.algorithms:
        histories = [run_config(p, make_alm, clamp=True) for p in tqdm(alm_max_grid, desc="alm_max")]
        save_method(result_dir, "alm_max", histories, alm_max_grid)

    # ===== SSw (switching subgradient) =====
    if 'ssg' in cfg.algorithms:
        histories = [run_config(p, None, mode='sw') for p in tqdm(ssg_grid, desc="ssg")]
        save_method(result_dir, "ssg", histories, ssg_grid)


@hydra.main(version_base=None, config_path="conf/task/", config_name="burgers")   # CHANGED: argparse -> Hydra
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    main_function(cfg.get("model", "deep_narrow"), cfg.get("beta", 1.0),
                  cfg.get("lr", 1e-3), cfg.get("n_epochs", 100), device, seed, cfg)
 
 
if __name__ == "__main__":
    main()
 