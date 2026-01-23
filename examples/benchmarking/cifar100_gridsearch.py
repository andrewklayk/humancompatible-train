"""
python script with cifar10 benchmarking
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch
from torch.nn import Sequential
from folktables import ACSDataSource, ACSIncome, generate_categories
import numpy as np
import matplotlib.pyplot as plt
import os
from humancompatible.train.optim import SSG
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from humancompatible.train.optim import SSLALM_Adam
import sys, os
from humancompatible.train.optim.PBM import PBM
from fairret.statistic import PositiveRate
from fairret.loss import NormLoss
from humancompatible.train.fairness.utils import BalancedBatchSampler
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
from humancompatible.train.fairness.utils.dataset_loader import get_data_dutch
import torchvision.transforms as transforms
import torchvision
from benchmarking import cifar_train
import copy
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchvision.models import resnet18


def load_data(balanced=False):

    # load the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 400

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

    # Get class names for CIFAR-100
    global classes
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    class_ind = {}

    # index the classes
    for i, classn in enumerate(classes):
        class_ind[classn] = i

    print(classes)
    print(class_ind)

    # load all data and create a balanced sampler
    X = torch.stack([item[0] for item in trainset])
    targets = torch.tensor([item[1] for item in trainset])

    # create onehot vectors
    groups_onehot = torch.eye(100)[targets]

    # create a train dataset
    dataset_train = torch.utils.data.TensorDataset(X, groups_onehot, targets)

    # create the balanced dataloader
    sampler = BalancedBatchSampler(
        group_onehot=groups_onehot, batch_size=batch_size, drop_last=True
    )
    if balanced:
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler, num_workers=10)
    else: 
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=10)

    # load all data and create a balanced sampler
    X_test = torch.stack([item[0] for item in testset])
    targets_test = torch.tensor([item[1] for item in testset])

    # create onehot vectors
    groups_onehot_test = torch.eye(100)[targets_test]

    # split test / val
    X_test, X_val, targets_test, targets_val, groups_onehot_test, groups_onehot_val = \
                            train_test_split(X_test, targets_test, groups_onehot_test, test_size=0.5)

    # create a train dataset
    dataset_val = torch.utils.data.TensorDataset(X_val, groups_onehot_val, targets_val)

    # create the balanced dataloader
    sampler = BalancedBatchSampler(
        group_onehot=groups_onehot_val, batch_size=batch_size, drop_last=True
    )
    global testloader   
    if balanced:
        testloader = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler, num_workers=10)
    else:
        testloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=10)

    # clean the memory of redundant variables
    del X, targets, groups_onehot
    del X_test, targets_test, groups_onehot_test

    return trainloader, testloader, classes, class_ind

def loss_per_class_f(batch_outputs, batch_targets, network, criterion, num_classes=100):
    """
    Computes the constraint of a demographic parity - that is a loss between all groups
    """

    losses_per_class = torch.zeros(num_classes)

    # for each class compute a loss
    for class_number in range(0, num_classes):

        # get data of that class
        class_args_i = torch.where(batch_targets == class_number)
        batch_outputs_class_i = batch_outputs[class_args_i]
        batch_targets_class_i = batch_targets[class_args_i]
        
        # compute loss for a given class
        batch_loss_class_i = criterion(batch_outputs_class_i, batch_targets_class_i)

        # save the loss
        losses_per_class[class_number] = batch_loss_class_i

    return losses_per_class

def test_network(network):
    
    # prepare count for correct answers
    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    loss_log = []
    c_log = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():

        for data in testloader:
            
            images, _, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            
            # calculate outputs by running images through the network
            outputs = network(images)

            # compute loss per class
            loss_per_class = loss_per_class_f(outputs, labels, network, criterion)

            # compute the loss
            loss = criterion(outputs, labels)   

            # save the logs
            loss_log.append(loss.detach().cpu().numpy())

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            
            # compute the accuracy overall
            correct += (predicted == labels).sum().item()

            # compute per class accuracy for this batch
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1

                total_pred[classes[label]] += 1

            # compute the demographic parity constraints
            c_log.append([])
            constraint_k = 0
            for group_i in range(0, 100):
                for group_j in range(0, 100):

                    if group_i != group_j:
                        
                        # demographic parity between i,j
                        g = loss_per_class[group_i] - loss_per_class[group_j]

                        c_log[-1].append(g.detach().cpu().numpy())
                        constraint_k += 1

    # compute the accuracy per class
    for classname, correct_count in correct_pred.items():
        accuracy = float(correct_count) / total_pred[classname]
        correct_pred[classname] = accuracy

    accuracy_total = correct / total
    accuracy_per_group = correct_pred 

    # returns loss, constraints, total hard accuracy, hard accuracy per group, soft accuracy per group
    return loss_log, c_log, accuracy_total, accuracy_per_group

def print_groups(tensor_per_group_acc, classes, class_ind):
    """Accuracy with class labels
    """

    correct_pred = {classname: 0 for classname in classes}
    for classname, _ in correct_pred.items():
            correct_pred[ classname ] = tensor_per_group_acc[class_ind[classname]].detach().cpu().item()
    
    return correct_pred


import torch.nn as nn
import torch.nn.functional as F

# define the network
class Net(nn.Module):
    def __init__(self, _=None, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def grid_adam(n_epochs_fit, seed_n, fair_crit_bound, print_n):

    lrs = [0.1, 0.01, 0.001, 0.003]

    # best 
    best_fit = np.inf
    best_params = None

    for lr in lrs:
            
            # set the model params
            model_params = {'lr': lr}

            # train the model on cifar dataset, with constraints based on the given parameters and the method
            S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
            test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
            accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
                    cifar_train(network_arch, n_epochs_fit, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='unconstrained', model_params=model_params)

            # best fit update based on the current validation data
            cur_fit = (test_S_loss_log_plotting[-1] + test_S_loss_log_plotting[-2]) / 2

            # print the status
            print("Current FIT: ", cur_fit, model_params)
            print("Best FIT: ", best_fit, best_params)

            if cur_fit < best_fit:
                    best_fit = cur_fit
                    best_params = copy.deepcopy(model_params)

    return best_fit, best_params

def grid_ssw(n_epochs_fit, seed_n, fair_crit_bound, print_n):

    # define the params and the number of epochs 
    lrs = [0.1, 0.01, 0.001, 0.003]
    dual_lrs = [0.1, 0.01, 0.001]

    # best 
    best_fit = np.inf
    best_params = None

    for lr in lrs:
        for dual_lr in dual_lrs:
            
            # set the model params
            model_params = {'lr': lr, 'dual_lr': dual_lr}

            # train the model on cifar dataset, with constraints based on the given parameters and the method
            S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
            test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
            accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
                    cifar_train(network_arch, n_epochs_fit, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='ssw', model_params=model_params)

            # best fit update based on the current train
            c_mean1 = np.max(test_S_c_log_plotting[-1], axis=0)
            c_mean2 = np.max(test_S_c_log_plotting[-2], axis=0)

            cur_fit = ( test_S_loss_log_plotting[-1] + test_S_loss_log_plotting[-2] + c_mean1 + c_mean2 ) / 4

            # print the status
            print("Current FIT: ", cur_fit, model_params)
            print("Best FIT: ", best_fit, best_params)

            if cur_fit < best_fit:
                    best_fit = cur_fit
                    best_params = copy.deepcopy(model_params)


    return best_fit, best_params


def grid_sslalm(n_epochs_fit, seed_n, fair_crit_bound, print_n):

    lrs = [0.1, 0.01, 0.003, 0.001]
    dual_lrs = [0.1, 0.01, 0.001]
    mus = [0.0, 1.0]
    rhos = [0.0, 1.0]

    # best 
    best_fit = np.inf
    best_params = None

    for lr in lrs:
        for dual_lr in dual_lrs:
            for mu in mus:
                for rho in rhos:
            
                    # set the model params
                    model_params = {'lr': lr, 'dual_lr': dual_lr, 'mu': mu, 'rho': rho}

                    # train the model on cifar dataset, with constraints based on the given parameters and the method
                    S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
                    test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
                    accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
                            cifar_train(network_arch, n_epochs_fit, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='ssl-alm', model_params=model_params)

                    # best fit update based on the current train
                    c_mean1 = np.max(test_S_c_log_plotting[-1])
                    c_mean2 = np.max(test_S_c_log_plotting[-2])

                    cur_fit = ( test_S_loss_log_plotting[-1] + test_S_loss_log_plotting[-2] + c_mean1 + c_mean2 ) / 4

                    # print the status
                    print("Current FIT: ", cur_fit, model_params)
                    print("Best FIT: ", best_fit, best_params)

                    if cur_fit < best_fit:
                            best_fit = cur_fit
                            best_params = copy.deepcopy(model_params)


    return best_fit, best_params


def grid_pbm(n_epochs_fit, seed_n, fair_crit_bound, print_n):   

    # define the params and the number of epochs 
    lrs = [0.1, 0.01, 0.003, 0.005, 0.001]
    dual_betas = [0.0, 0.5, 0.9]
    mus = [0.1, 1.0]
    init_duals = [0.0001]
    penalties = ["quadratic_logarithmic", "quadratic_reciprocal"]
    warm_starts = [0]

    # best 
    best_fit = np.inf
    best_params = None

    for lr in lrs:
        for dual_beta in dual_betas:
            for penalty in penalties:
                    for init_dual in init_duals:
                            for mu in mus:
                                    for warm_start in warm_starts:
                            
                                            # set the model params
                                            model_params = {'lr': lr, 'dual_beta': dual_beta, 'mu': mu, 'penalty': penalty, 'init_dual': init_dual, 'warm_start': warm_start}

                                            # train the model on cifar dataset, with constraints based on the given parameters and the method
                                            S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
                                            test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
                                            accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
                                                    cifar_train(network_arch, n_epochs_fit, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='pbm', model_params=model_params)

                                            # best fit update based on the current train
                                            c_mean1 = np.max(test_S_c_log_plotting[-1])
                                            c_mean2 = np.max(test_S_c_log_plotting[-2])

                                            cur_fit = ( test_S_loss_log_plotting[-1] + test_S_loss_log_plotting[-2] + c_mean1 + c_mean2 ) / 4

                                            # print the status
                                            print("Current FIT: ", cur_fit, model_params)
                                            print("Best FIT: ", best_fit, best_params)

                                            if cur_fit < best_fit:
                                                    best_fit = cur_fit
                                                    best_params = copy.deepcopy(model_params)


    return best_fit, best_params

if __name__ == '__main__':

    # define the torch seed here
    n_epochs = 6
    n_constraints = 9990
    threshold = 0.5
    # device = 'cpu'    
    seed_n = 1
    device = 'cuda:0'
    bench_mus = False  # true to benchmark mus on cifar10 pbm

    network_arch = resnet18

    # load data
    trainloader, loader_val, classes, class_ind = load_data(balanced=True)

    # define the criterion
    criterion = nn.CrossEntropyLoss()
    
    print('STARTING ADAM')
    best_fit_adam, best_params_adam = grid_adam(n_epochs, seed_n, threshold, len(trainloader))

    print('STARTING SSW')
    best_fit_ssw, best_params_ssw = grid_ssw(n_epochs, seed_n, threshold, len(trainloader))

    print('STARTING SSLALM')
    best_fit_sslalm, best_params_sslalm = grid_sslalm(n_epochs, seed_n, threshold, len(trainloader))

    print('STARTING PBM')
    best_fit_pbm, best_params_pbm = grid_pbm(n_epochs, seed_n, threshold, len(trainloader))

    # print the best params found for the current method
    print('Best found params for ADAM: ')
    print(best_fit_adam)
    print(best_params_adam)
    
    # print the best params found for the current method
    print('Best found params for SSW: ')
    print(best_fit_ssw)
    print(best_params_ssw)


    # print the best params found for the current method
    print('Best found params for SSLALM: ')
    print(best_fit_sslalm)
    print(best_params_sslalm)
    
    # print the best params found for the current method
    print('Best found params for PBM: ')
    print(best_fit_pbm)
    print(best_params_pbm)