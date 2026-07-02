from tqdm import tqdm
import numpy as np
import sys
import os
import json
import time as _time
import pandas as pd
from itertools import product
import hydra
from omegaconf import DictConfig
sys.path.append("..")

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from humancompatible.train.dual_optim import ALM, MoreauEnvelope, PBM

from networks import set_model, u_Net_shallow_wide, u_Net_shallow_wide_resnet, u_Net_deep_narrow, u_Net_deep_narrow_resnet

# Equation parameters
k, a1, a2 = 1, 1, 4

THRESHOLD = 1e-4
device = 'cuda'


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
        [0.001, 0.005, 0.01, 0.02, 0.05], [0., 0.1, 0.5, 0.9, 1.0], ["dimin_adapt"],
        ["quadratic_logarithmic"], [[1e-1, 1.], [1e-2, 1.]], [0.9], [0., 1., 2.], 
        [1], [True], [True, False], [False], [None])
]
# ensure the primal update process length is the same for both moreau and dual
for arr_dict in pbm_grid:
    arr_dict["moreau__primal_update_process_length"] = arr_dict["dual__primal_update_process_length"]

    if arr_dict["dual__gamma_annealing"] == True:
        arr_dict["dual__gamma"] = 1 / 10 # in the case of dual anneling, gamma needs to be small at first

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
        [1], [None], [True, False], [True], [0.1, 0.01, 0.5])
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


# ── PDE helpers ───────────────────────────────────────────────────────────────
def q(data):
    x, y = data[:, 0].view(-1, 1), data[:, 1].view(-1, 1)
    return (-((a1 * np.pi) ** 2) * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y)
            - ((a2 * np.pi) ** 2) * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y)
            + (k ** 2) * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y))

def calculate_derivative(y, x):
    return torch.autograd.grad(y, x, create_graph=True,
                               grad_outputs=torch.ones(y.size()).to(device))[0]

def calculate_all_partial(u_out, x):
    del_u = calculate_derivative(u_out, x)
    u_x, u_y = del_u[:, 0], del_u[:, 1]
    u_xx = calculate_derivative(u_x, x)[:, 0]
    u_yy = calculate_derivative(u_y, x)[:, 1]
    return u_xx.view(-1, 1), u_yy.view(-1, 1)


def train(u_model, beta, trainloader, bdry_data, val_test, optimizer, loss_f,
          dual_opt=None, clamp=False, mode='lagrangian', sw_dual=None, constraint_tol=0.):
    loss_list, loss_list1, loss_list2 = [], [], []
    val_list, test_list = [], []
    kkt_list = []  # ADDED: per-batch KKT dicts (full-batch loader -> whole train set)
    X_bdry, u_bdry = bdry_data
    X_val, y_val, X_test, y_test = val_test

    for i, (data,) in enumerate(trainloader):
        u_model.train()
        optimizer.zero_grad()
        X_v = Variable(data, requires_grad=True).to(device)
        output = u_model(X_v)
        output_bdry = u_model(X_bdry)

        u_xx, u_yy = calculate_all_partial(output, X_v)
        loss1 = loss_f(u_xx + u_yy + (k ** 2) * output - q(X_v), torch.zeros_like(output))
        loss2 = loss_f(output_bdry, torch.zeros_like(output_bdry))

        if dual_opt is None and mode != 'sw':
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        elif mode == 'sw':
            cons = torch.stack([loss2], dim=0) - THRESHOLD
            max_c = cons.max()
            optimizer.zero_grad()
            sw_dual.zero_grad()
            if max_c > constraint_tol:
                max_c.backward()
                sw_dual.step()
            else:
                loss1.backward()
                optimizer.step()
            optimizer.zero_grad()
            sw_dual.zero_grad()

        elif dual_opt is not None:
            constraint = loss2 - THRESHOLD
            if clamp:
                constraint = torch.max(constraint, torch.zeros_like(constraint))
            lagrangian = dual_opt.forward_update(loss1, constraint.unsqueeze(0))
            lagrangian.backward()
            optimizer.step()
            optimizer.zero_grad()

        u_model.eval()
        val_err = torch.linalg.norm((u_model(X_val) - y_val), 2).item() / torch.linalg.norm(y_val, 2).item()
        test_err = torch.linalg.norm((u_model(X_test) - y_test), 2).item() / torch.linalg.norm(y_test, 2).item()


        # ── KKT on the entire train set (post-step, current weights) ──── ADDED
        # min f=loss1  s.t.  g=[loss2,loss3]-THRESHOLD <= 0 ; L = f + λᵀg.
        X_v = Variable(data, requires_grad=True).to(device)
        output = u_model(X_v)
        output_bdry = u_model(X_bdry)

        u_xx, u_yy = calculate_all_partial(output, X_v)
        f = loss_f(u_xx + u_yy + (k ** 2) * output - q(X_v), torch.zeros_like(output))
        g = loss_f(output_bdry, torch.zeros_like(output_bdry)).reshape(-1)

        lam = dual_opt.duals.detach() if dual_opt is not None \
              else torch.zeros(1, device=device)
        L = f + lam @ g
        
        params = [p for p in u_model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(L, params, allow_unused=True)
        grad_norm = torch.sqrt(sum((gr**2).sum() for gr in grads if gr is not None)).item()
        max_viol = g.max().item()
        compl = (lam * g).abs().sum().item()
        kkt_list.append({"kkt_r": grad_norm + max(0., max_viol) + compl,
                         "kkt_grad": grad_norm, "kkt_viol": max_viol, "kkt_compl": compl,
                         "lambda_0": lam[0].item()})

        loss_list.append((loss1 + loss2).item())
        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())
        val_list.append(val_err)
        test_list.append(test_err)


    kkt = {k: float(np.mean([d[k] for d in kkt_list])) for k in kkt_list[0]}  # ADDED
    return (np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2),
            np.mean(val_list), np.mean(test_list), kkt)


# ── saving helpers ────────────────────────────────────────────────────────────
def save_method(result_dir, method, histories, grid):
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


def main_function(model_name, beta, lr, EPOCH, device, seed):

    result_dir = f"results/helmholtz_pinn{seed}"
    os.makedirs(result_dir, exist_ok=True)

    # Dataset Creation
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    X_train = torch.FloatTensor(np.mgrid[xmin:xmax:51j, ymin:ymax:51j].reshape(2, -1).T).to(device)

    # Boundary Conditions
    X_bdry = X_train[(X_train[:, 0] == xmin) + (X_train[:, 0] == xmax)
                     + (X_train[:, 1] == ymin) + (X_train[:, 1] == ymax)]
    u_bdry = torch.zeros_like(X_bdry[:, 0]).to(device).view(-1, 1)

    # Validation & Test Set
    X_test, y_test, X_val, y_val = torch.load('../PDEs/Helmholtz/Helmholtz_test', map_location=device)

    idx = np.random.choice(X_val.shape[0], 1000, replace=False)
    X_val = X_val[idx]
    y_val = y_val[idx]

    # Make dataloader
    data_train = TensorDataset(X_train)
    train_loader = DataLoader(data_train, batch_size=10000, shuffle=False)

    bdry = [X_bdry, u_bdry]
    val_test = [X_val, y_val, X_test, y_test]

    def run_config(params, dual_ctor, clamp=False, mode='lagrangian'):
        primal = {k.removeprefix("primal__"): v for k, v in params.items() if k.startswith("primal__")}
        dual_p = {k.removeprefix("dual__"): v for k, v in params.items() if k.startswith("dual__")}
        moreau = {k.removeprefix("moreau__"): v for k, v in params.items() if k.startswith("moreau__")}
        b = params.get("beta", beta)
        torch.manual_seed(0)
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
        for t in range(0, EPOCH):
            loss, loss1, loss2, val_err, test_err, kkt = train(
                u_model, b, trainloader=train_loader, bdry_data=bdry,
                val_test=val_test, optimizer=optimizer, loss_f=nn.MSELoss(),
                dual_opt=dual, clamp=clamp, mode=mode, sw_dual=sw_dual)
            history.append({"epoch": t, "time": _time.time() - t0, "loss": loss1,
                            "c_0": loss2, "val": val_err, "test": test_err, **kkt})
            if t % 100 == 0:
                print("%s/%s | loss: %06.6f | c: %06.6f | val: %06.6f | test: %06.6f " %
                      (t, EPOCH, loss1, loss2, val_err, test_err))
        return history

    def make_pbm(params):
        dp = {k.removeprefix("dual__"): v for k, v in params.items() if k.startswith("dual__")}
        return PBM(m=1, dual_range=(0.01, 100.), **dp, device=device)

    def make_alm(params):
        dp = {k.removeprefix("dual__"): v for k, v in params.items() if k.startswith("dual__")}
        return ALM(m=1, **dp, device=device)

    # ===== ADAM =====
    # histories = [run_config(p, None) for p in tqdm(adam_grid, desc="adam")]
    # save_method(result_dir, "adam", histories, adam_grid)

    # ===== SPBM (PBM) Log  =====
    # ensure the pbm has the size of the epoch (for penalty annealing)
    for arr_dict in pbm_logascaled_grid:   
        arr_dict["dual__epoch_length"] = len(train_loader)

    histories = [run_config(p, make_pbm) for p in tqdm(pbm_logascaled_grid, desc="pbm")]
    save_method(result_dir, "pbm", histories, pbm_logascaled_grid)

    # ===== SPBM (PBM) =====
    for arr_dict in pbm_grid:   
        arr_dict["dual__epoch_length"] = len(train_loader)
    histories = [run_config(p, make_pbm) for p in tqdm(pbm_grid, desc="pbm")]
    save_method(result_dir, "pbm", histories, pbm_grid)

    # ===== ALM projection (raw signed constraints) =====
    histories = [run_config(p, make_alm, clamp=False) for p in tqdm(alm_proj_grid, desc="alm_proj")]
    save_method(result_dir, "alm_proj", histories, alm_proj_grid)

    # ===== ALM max (clamped constraints) =====
    histories = [run_config(p, make_alm, clamp=True) for p in tqdm(alm_max_grid, desc="alm_max")]
    save_method(result_dir, "alm_max", histories, alm_max_grid)

    # ===== SSw (switching subgradient) =====
    histories = [run_config(p, None, mode='sw') for p in tqdm(ssg_grid, desc="ssg")]
    save_method(result_dir, "ssg", histories, ssg_grid)


@hydra.main(version_base=None, config_path="conf/task/", config_name="helmholtz")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    main_function(cfg.get("model", "deep_narrow"), cfg.get("beta", 1.0),
                  cfg.get("lr", 1e-3), cfg.get("n_epochs", 1000), device, seed)


if __name__ == "__main__":
    main()
