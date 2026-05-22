from tqdm import tqdm
import pickle as pkl
import numpy as np
import copy
import argparse
import sys
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

def calculate_derivative(y, x) :
    return torch.autograd.grad(y, x, create_graph=True,\
                        grad_outputs=torch.ones(y.size()).to(device))[0]

def calculate_all_partial(u, x) :
    del_u = calculate_derivative(u, x)
    u_t, u_x = del_u[:,0], del_u[:,1]
    u_xx = calculate_derivative(u_x, x)[:,1]
    return u_t.view(-1,1), u_x.view(-1,1), u_xx.view(-1,1)


def train(u_model, beta, trainloader, ini_bdry_data, val_test, optimizer, loss_f, dual_opt=None) :
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
            threshold = 1e-4
            constraints = torch.stack([loss2, loss3], dim=0)
            constraints = constraints - threshold

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


def main_function(model_name, beta, lr, EPOCH, device) :
    
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
    
    # train
    torch.manual_seed(0)
    total_loss, test_errs, val_errs, constraints = [], [], [], []
    u_model = set_model(model_name, device)
    optimizer=torch.optim.Adam([{'params': u_model.parameters()}], lr=lr)
    best_model = copy.deepcopy(u_model)

    for t in tqdm(range(0, EPOCH)) :

        loss, loss1, loss2, loss3, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                      ini_bdry_data=[X_ini, u_ini, X_bdry, u_bdry],\
                                                      val_test = [X_val, y_val, X_test, y_test],\
                                                      optimizer=optimizer, loss_f=nn.MSELoss(),
                                                      dual_opt=None)
        
        # val_errs.append(torch.linalg.norm((u_model(X_val) - y_val),2).item() / torch.linalg.norm(y_val,2).item())    
        # test_errs.append(torch.linalg.norm((u_model(X_test) - y_test),2).item() / torch.linalg.norm(y_test,2).item())    
        # total_loss.append(loss)

        val_errs.append(val_err)
        test_errs.append(test_err)
        total_loss.append(loss)
        constraints.append([loss2, loss3])    # append both costraint

        # Print Log
        if t%100 == 0 :
            print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                  (t, EPOCH, loss, loss1, loss2+loss3, val_err, test_err))

        if np.argmin(val_errs) == t :
            best_model = copy.deepcopy(u_model)



    # SPBM
    torch.manual_seed(0)
    total_loss_spbm, test_errs_spbm, val_errs_spbm, constraints_spbm  = [], [], [], []
    u_model = set_model(model_name, device)
    
    # Define data and optimizers
    optimizer = MoreauEnvelope(torch.optim.Adam([{'params': u_model.parameters()}], lr=0.005), mu=2.0)
    
    dual = PBM(
        m=2,
        # penalty_update='dimin',
        # penalty_update='dimin_adapt',
        penalty_update='const',
        pbf = 'quadratic_logarithmic',
        gamma=0.9,
        init_duals=0.01,
        init_penalties=1.,
        penalty_range=(0.5, 1.),
        penalty_mult=0.99,
        dual_range=(0.01, 100.),
        delta=1.0,
        device=device
    )

    for t in tqdm(range(0, EPOCH)) :

        loss, loss1, loss2, loss3, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                      ini_bdry_data=[X_ini, u_ini, X_bdry, u_bdry],\
                                                      val_test = [X_val, y_val, X_test, y_test],\
                                                      optimizer=optimizer, loss_f=nn.MSELoss(),
                                                      dual_opt=dual)
        
        val_errs_spbm.append(val_err)
        test_errs_spbm.append(test_err)
        total_loss_spbm.append(loss)
        constraints_spbm.append([loss2, loss3])

        #Print Log
        if t%100 == 0 :
            print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                  (t, EPOCH, loss, loss1, loss2+loss3, val_err, test_err))

        if np.argmin(val_errs_spbm) == t :
            best_model = copy.deepcopy(u_model)

    # ALM
    torch.manual_seed(0)
    total_loss_alm, test_errs_alm, val_errs_alm, constraints_alm = [], [], [], []
    u_model = set_model(model_name, device)
    
    # Define data and optimizers
    optimizer = MoreauEnvelope(torch.optim.Adam([{'params': u_model.parameters()}], lr=0.005), mu=2.0)
    
    dual = ALM(
        m=2,
        lr=0.1,
        momentum=0.5,
        device=device   
    )  

    for t in tqdm(range(0, EPOCH)) :

        loss, loss1, loss2, loss3, val_err, test_err = train(u_model, beta, trainloader=train_loader,\
                                                      ini_bdry_data=[X_ini, u_ini, X_bdry, u_bdry],\
                                                      val_test = [X_val, y_val, X_test, y_test],\
                                                      optimizer=optimizer, loss_f=nn.MSELoss(),
                                                      dual_opt=dual)
        
        val_errs_alm.append(val_err)
        test_errs_alm.append(test_err)
        total_loss_alm.append(loss)
        constraints_alm.append([loss2, loss3])

        #Print Log
        if t%100 == 0 :
            print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
                  (t, EPOCH, loss, loss1, loss2+loss3, val_err, test_err))

        if np.argmin(val_errs_alm) == t :
            best_model = copy.deepcopy(u_model)


    # plot the resultsimport matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # wider figure

    axes[0].plot(total_loss, label='Adam')
    axes[0].plot(total_loss_spbm, label='SPBM')
    axes[0].plot(total_loss_alm, label='ALM')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Total Loss')
    axes[0].legend()

    axes[1].plot(test_errs, label='Adam')
    axes[1].plot(test_errs_spbm, label='SPBM')
    axes[1].plot(test_errs_alm, label='ALM')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Error')
    axes[1].legend()

    # plot both constraints + methods shuold have the same color but dashed vs solid
    axes[2].plot([c[0] for c in constraints], label='Adam - Initial Condition', linestyle='--', color='blue')
    axes[2].plot([c[1] for c in constraints], label='Adam - Boundary Condition', linestyle='-', color='blue')
    axes[2].plot([c[0] for c in constraints_spbm], label='SPBM - Initial Condition', linestyle='--', color='orange')
    axes[2].plot([c[1] for c in constraints_spbm], label='SPBM - Boundary Condition', linestyle='-', color='orange')   
    axes[2].plot([c[0] for c in constraints_alm], label='ALM - Initial Condition', linestyle='--', color='green')
    axes[2].plot([c[1] for c in constraints_alm], label='ALM - Boundary Condition', linestyle='-', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Constraint Violation')
    axes[2].legend()

    plt.tight_layout(pad=2.0)  # extra padding between subplots
    plt.savefig('./PDEs/Viscous_Burgers/results.png', bbox_inches='tight')
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deep_narrow', help='Specify the model. Choose one of [deep_narrow, shallow_wide, deep_narrow_resent, shallow_wide_resnet].')
    parser.add_argument('--beta', default=1, type=float, help='Penalty parameter beta')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--EPOCH', default=1000, type=int, help='Number of training EPOCH')
    parser.add_argument('--ordinal', default=0, type=int, help='Specify the cuda device ordinal.')
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.ordinal) if torch.cuda.is_available() else "cpu")
    
    main_function(args.model, args.beta, args.lr, args.EPOCH, device)
    