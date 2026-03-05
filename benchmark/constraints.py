from benchmark_utils import *
from itertools import product
import torch
from _data_sources import load_data_FT_vec, load_data_FT_prod, load_data_DUTCH, load_data_norm
import pandas as pd
import argparse
import os
import torch.nn.functional as F

from fairret.statistic import PositiveRate
from fairret.loss import NormLoss

from plotting import plot_losses_and_constraints_stochastic
from humancompatible.train.dual_optim import ALM, PBM

# loss per each group - mean loss

def _bce_loss_per_group(batch_logits, batch_sens, batch_labels):
    # Ensure shapes are compatible
    batch_logits = batch_logits.view(-1)
    batch_labels = batch_labels.view(-1)
    
    # Compute BCE loss for each sample
    loss = F.binary_cross_entropy_with_logits(
        batch_logits, batch_labels, reduction='none'
    )

    # Weight loss by group membership and sum per group
    group_loss = torch.matmul(batch_sens.T, loss)

    # Divide by the number of samples in each group
    group_counts = batch_sens.sum(dim=0)
    group_loss /= group_counts

    # Compute mean-reduced loss
    mean_loss = loss.mean()

    return torch.abs(group_loss - mean_loss)

def loss_per_group(model, out, batch_sens, batch_labels):
    return _bce_loss_per_group(out, batch_sens, batch_labels)



def positive_rate_per_group(out_batch, batch_sens, prob_f=torch.nn.functional.sigmoid):
    """
    Calculates the positive rate vector based on the given outputs of the model for the given groups. 
    
    """
    if prob_f is None: 
        preds = out_batch
    else: 
        preds = prob_f( out_batch )
    pr = PositiveRate()
    probs_per_group = pr(preds, batch_sens)

    return probs_per_group

def posrate_per_group(model, out, batch_sens, batch_labels):
    pos_rate_pergroup = positive_rate_per_group(out, batch_sens)
    constraints = ((pos_rate_pergroup.unsqueeze(1) - pos_rate_pergroup.unsqueeze(0)).to(torch.float))
    mask = ~torch.eye(batch_sens.shape[-1], dtype=torch.bool)
    constraints = constraints[mask]

    return constraints

def posrate_fairret_constraint(model, out, batch_sens, batch_labels):
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    
    return fair_criterion(out, batch_sens).unsqueeze(0)

def weight_constraint(model, out, batch_sens, batch_labels):
    norms = []
    for param in model.parameters():
        norm = torch.linalg.norm(param)
        norms.append(norm.unsqueeze(0))
    
    return torch.concat(norms)
