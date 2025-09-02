import torch
from torch import Tensor
from torch.nn import Module

def loss_equality(preds: Tensor, sens: Tensor, labels: Tensor, criterion: Module = None, diff_to_overall: bool = False):
    """
    A constraint that penalizes the sum of difference between the loss of each group and the overall loss if`diff_to_overall`is`True`,
    and the absolute difference in loss between the two groups if`diff_to_overall`is`False`.
    
    Args:
            logits (torch.Tensor): Predictions of shape :math:`(N)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): One-hot encoding of group membership of shape`(N, S)`with`S`the number of sensitive features.
                `S`must be 2 if`diff_to_overall`is`True`.
            labels (torch.Tensor): Predictions of shape`(N)`.
            loss: (torch.nn.Module): The loss function to calculate.
            diff_to_overall: (bool): Determines whether to penalize the sum of the absolute difference between
                each group's loss and the overall loss if`True`, or the absolute difference in losses of two groups otherwise.
    """
    if not diff_to_overall and sens.shape[-1] != 2:
        raise ValueError(f"If`diff_to_overall` is`False`, expected`sens.shape[-1]` to be 2, got {sens.shape[-1]}")
    
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    sens_t = sens.T
    group_losses = torch.empty(sens.shape[-1])
    for group in range(sens.shape[-1]):
        group_preds, group_labels = preds[sens_t[group] == 1], labels[sens_t[group] == 1]
        group_losses[group] = criterion(group_preds.squeeze(), group_labels)
        
    if not diff_to_overall:
        return torch.abs(group_losses[0] - group_losses[1])
    
    overall_loss = criterion(preds.squeeze(), labels)
    return torch.sum(torch.abs(overall_loss-group_losses))