import torch
import torch.nn as nn
import torch.optim as optim
from fairret.statistic import Accuracy
from fairret.loss import NormLoss
from humancompatible.train.fairness.utils import BalancedBatchSampler
import copy
import numpy as np
from humancompatible.train.optim.ssw import SSG
from humancompatible.train.optim.PBM import PBM
from humancompatible.train.optim.ssl_alm_adam import SSLALM_Adam

"""
Helper functions for benchmarking notebook
"""

def cifar_train(network_achitecture, n_epochs, seed_n, trainloader, loss_per_class_f, test_network_f, device, classes_arr, fair_crit_bound, print_n, method='unconstrained',
                        model_params=None, init_weights=None):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # create the network
    net = network_achitecture(init_weights, num_classes=len(classes_arr))
    net.to(device)

    # define the loss function and the 
    criterion = nn.CrossEntropyLoss()

    # define number of constraints of the demographic parity
    num_constraints = len(classes_arr) * (len(classes_arr)-1)

    # set the model parameters based on the optimizer
    if method == "unconstrained":
        lr = model_params['lr']
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif method == 'ssw':
        lr = model_params['lr']
        dual_lr = model_params['dual_lr']
        optimizer = SSG(params=net.parameters(), m=1, lr=lr, dual_lr=dual_lr)
    elif method == 'ssl-alm':
        lr = model_params['lr']
        dual_lr = model_params['dual_lr']
        mu = model_params['mu']
        rho = model_params['rho']
        optimizer = SSLALM_Adam(
            params=net.parameters(),
            m=num_constraints,  # number of constraints - one in our case
            lr=lr,  # primal variable lr
            dual_lr=dual_lr,  # lr of a dual ALM variable
            dual_bound=5,
            rho=rho,  # rho penalty in ALM parameter
            mu=mu,  # smoothing parameter
            device=device,
        )
    elif method == "pbm":
        lr = model_params['lr']
        dual_beta = model_params['dual_beta']
        mu = model_params['mu']
        penalty = model_params['penalty']
        init_dual = model_params['init_dual']
        warm_start = model_params['warm_start']
        penalty_update_m = model_params['p_update']
        optimizer = PBM(params=net.parameters(), m=num_constraints, lr=lr, dual_beta=dual_beta, mu=mu, 
                epoch_len=len(trainloader), init_dual=init_dual, penalty_update_m='CONST', p_lb=0.1, warm_start=warm_start,
                barrier=penalty, device=device)
    else: 
        raise ValueError("No such method available!")

    # alloc arrays for plotting
    S_loss_log_plotting = []  # mean
    S_c_log_plotting = []  # mean
    S_loss_std_log_plotting = []  # std
    S_c_std_log_plotting = []  # std

    test_S_loss_log_plotting = []  # mean
    test_S_c_log_plotting = []  # mean
    test_S_loss_std_log_plotting = []  # std
    test_S_c_std_log_plotting = []  # std

    # save the accuracy per class
    accuracy_plotting = []
    accuracy_per_class_plotting = []
    accuracy_plotting_t = []
    accuracy_per_class_plotting_t = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # predictions + predictions per group stats
        total_pred_epoch = {classname: 0 for classname in classes_arr}
        correct_pred_epoch = {classname: 0 for classname in classes_arr}
        total_epoch = 0
        correct_epoch = 0

        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        for i, data in enumerate(trainloader, 0):

            ############################ FORWARD ###########

            # get the inputs; data is a list of [inputs, labels]
            inputs, sens, labels = data[0].to(device), data[1].to(device), data[2].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            
            _, predicted = torch.max(outputs, 1) # compute the classes predictions

            ############################ CONSTRAINTS + STATISTICS ###########

            # compute the accuracy overall
            total_epoch += labels.size(0)
            correct_epoch += (predicted == labels).sum().item()

            # compute the constraints - accuracy among all groups
            for idx, zipped in enumerate(zip(labels, predicted)):
                label, prediction = zipped
                if label == prediction:
                    correct_pred_epoch[classes_arr[label]] += 1
                    
                # save the number of samples per that group in the batch
                total_pred_epoch[classes_arr[label]] += 1   

            ############################ BACKPROPAGATION ###########

            # compute the loss
            loss = criterion(outputs, labels)

            # compute a loss of each class in this batch
            loss_per_class = loss_per_class_f(outputs, labels, net, criterion)

            if method == 'ssw':
                max_c = torch.zeros(1, device=device)

            if method == 'ssl-alm':
                constraints = torch.zeros(num_constraints, device=device)


            if method == 'ssl-alm' or method == 'ssw':
                # compute the demographic parity
                c_log.append([])
                constraint_k = 0
                for group_i in range(0, len(classes_arr)):
                    for group_j in range(0, len(classes_arr)):

                        if group_i != group_j:
                            
                            # demographic parity between i,j
                            g = loss_per_class[group_i] - loss_per_class[group_j]
                            constr = g - fair_crit_bound

                            if method == 'ssw':
                                max_c = torch.max(constr, max_c)
                            
                            if method == 'ssl-alm':
                                constr = torch.max( g - fair_crit_bound, torch.zeros(1, device=device) )[0]
                                optimizer.dual_step(constraint_k, constr)
                                constraints[constraint_k] = constr 

                            c_log[-1].append(g.detach().cpu().numpy())
                            constraint_k += 1

            # for unconstrained and pbm do vectorized version
            elif method == 'pbm':

                # loss_per_class: shape (N,)
                N = loss_per_class.shape[0]

                # pairwise differences: shape (N, N)
                diff = loss_per_class.unsqueeze(1) - loss_per_class.unsqueeze(0)
                
                # remove diagonal (i == j)
                mask = ~torch.eye(N, dtype=torch.bool, device=loss_per_class.device)

                # apply fairness bound and flatten
                constr = (diff - fair_crit_bound)[mask]   # shape: (N*(N-1),)

                optimizer.dual_steps(constr)

                c_log.append(diff[mask].detach().cpu().numpy())

            else: # unsconstrained - just save the constraint 

                # loss_per_class: shape (N,)
                N = loss_per_class.shape[0]

                # pairwise differences: shape (N, N)
                diff = loss_per_class.unsqueeze(1) - loss_per_class.unsqueeze(0)
                
                # remove diagonal (i == j)
                mask = ~torch.eye(N, dtype=torch.bool, device=loss_per_class.device)

                c_log.append(diff[mask].detach().cpu().numpy())

            if method == "unconstrained":

                # backpropagate
                loss.backward()
                optimizer.step()
        
            if method == 'ssw':
                # calculate the Jacobian of the max-violating norm constraint
                max_c.backward(retain_graph=True)

                # save the gradient of the constraint
                optimizer.dual_step(0)
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step(max_c)
                optimizer.zero_grad()

            if method == 'ssl-alm':
                optimizer.step(loss, constraints)
                duals_log.append(optimizer._dual_vars.detach().cpu())

            if method == 'pbm':
                # backpropagate
                optimizer.step(loss)
                duals_log.append(optimizer._dual_vars.detach().cpu())

            # save the logs
            loss_log.append(loss.detach().cpu().numpy())

            ############################ PRINT ###########

            # print the statistics
            if i % print_n == print_n-1:    # print every 2000 mini-batches

                S_loss_log_plotting.append(np.mean(loss_log))
                S_c_log_plotting.append(np.mean(c_log, axis=0))
                S_loss_std_log_plotting.append(np.std(loss_log, axis=0))
                S_c_std_log_plotting.append(np.std(c_log, axis=0))

                # test the network after each epoch
                losses_test, c_test, overall_accuracy_t, acc_pergroup_t = test_network_f(net)

                # compute the accuracy per class    
                for classname, correct_count in correct_pred_epoch.items():
                    accuracy = float(correct_count) / total_pred_epoch[classname]
                    correct_pred_epoch[classname] = accuracy

                test_S_loss_log_plotting.append(np.mean(losses_test))
                test_S_c_log_plotting.append(np.mean(c_test, axis=0))
                test_S_loss_std_log_plotting.append(np.std(losses_test, axis=0))
                test_S_c_std_log_plotting.append(np.std(c_test, axis=0))

                accuracy_plotting += [correct_epoch/total_epoch]
                accuracy_per_class_plotting += [ copy.deepcopy(correct_pred_epoch) ]
                accuracy_plotting_t += [overall_accuracy_t]
                accuracy_per_class_plotting_t += [ copy.deepcopy(acc_pergroup_t) ]

                # print of the dual variables
                if method != 'ssw' and method != 'unconstrained':
                    str_dual_print = f"dual: {np.mean(duals_log, axis=0)}"
                else: 
                    str_dual_print = ""
                
                print(
                    f"Step: {epoch}, {i + 1:5d}, "
                    f"loss ({np.mean(loss_log):.4f}/{np.mean(losses_test):.4f}):"
                    f"constraints (train/test): ({np.max(np.mean(c_log, axis=0))}/{np.max(np.mean(c_test, axis=0))}), "
                    f"Accuracy (train/test): {correct_epoch/total_epoch}/{overall_accuracy_t}\n"
                    f"Accuracy Per group (train): {correct_pred_epoch}\n"
                    f"{str_dual_print}"
                )

                # restart the stats        
                total_pred_epoch = {classname: 0 for classname in classes_arr}
                correct_pred_epoch = {classname: 0 for classname in classes_arr}
                total_epoch = 0
                correct_epoch = 0
                
    print('Finished Training')

    # return the statistics
    return S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting, test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
            accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t