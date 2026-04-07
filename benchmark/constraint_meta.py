from typing import Callable, Any, Dict
from attr import dataclass
import torch
from constraints import positive_rate_per_group
import fairret



@dataclass
class ConstraintMetadata:
    """This class is a wrapper for fairness constraints;
    it contains the function that computes the constraint
    and a function that computes the number of constraints given the number of protected groups
    (for example, if the constraint calculates a metric for each pair of groups,`m`would be`n_groups`* (`n_groups` - 1))."""
    fn: Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    m_fn: Callable[[int], int]




### FAIRRET CONSTRAINTS ###

class FairretPairwise(ConstraintMetadata):
    """Wrapper class for a pairwise fairness constraint based on a given statistic (e.g., positive rate, false positive rate, etc.).
    The constraint is computed as the difference between the statistic for each pair of groups."""
    def __init__(self, statistic: Callable, uses_labels: bool, abs_diff: bool = False, as_logits: bool = False):
        """Initializes the FairretPairwise constraint.
        Args:
            statistic (Callable): An initialized fairret.statistic object.
            uses_labels (bool): Whether the statistic function requires the labels as input.
        """
        super().__init__(
            fn=self.compute_constraints,
            m_fn=lambda n_groups: n_groups * (n_groups - 1) if not self.abs_diff else n_groups * (n_groups - 1) // 2
        )
        self.abs_diff = abs_diff
        self.as_logits = as_logits
        if self.abs_diff:
            raise NotImplementedError("abs_diff=True is not implemented yet.")
        self.statistic = statistic
        self.uses_labels = uses_labels

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels):
        if not self.as_logits:
            batch_out = torch.sigmoid(batch_out)
        if self.uses_labels:
            stat_pergroup = self.statistic(batch_out, batch_sens, batch_labels)
        else:
            stat_pergroup = self.statistic(batch_out, batch_sens)
        constraints = ((stat_pergroup.unsqueeze(1) - stat_pergroup.unsqueeze(0)).to(torch.float))
        mask = ~torch.eye(batch_sens.shape[-1], dtype=torch.bool)
        constraints = constraints[mask]

        return constraints



class FairretMean(ConstraintMetadata):
    """Wrapper class for a pairwise fairness constraint based on a given statistic (e.g., positive rate, false positive rate, etc.).
    The constraint is computed as the difference between the statistic for each pair of groups."""
    def __init__(self, statistic: Callable, uses_labels: bool, as_logits: bool = False):
        """Initializes the FairretPairwise constraint.
        Args:
            statistic (Callable): An initialized fairret.statistic object.
            uses_labels (bool): Whether the statistic function requires the labels as input.
        """
        super().__init__(
            fn=self.compute_constraints,
            m_fn=lambda n_groups: n_groups
        )
        self.as_logits = as_logits
        self.statistic = statistic
        self.uses_labels = uses_labels

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels):
        if not self.as_logits:
            batch_out = torch.sigmoid(batch_out)
        if self.uses_labels:
            stat_pergroup = self.statistic(batch_out, batch_sens, batch_labels)
            mean_stat = self.statistic(batch_out, batch_labels)
        else:
            stat_pergroup = self.statistic(batch_out, batch_sens)
            mean_stat = self.statistic(batch_out)

        constraints = torch.abs(stat_pergroup - mean_stat)

        return constraints



class FairretLoss(ConstraintMetadata):
    """Wrapper class for a vector fairness constraint based on a given statistic (e.g., positive rate, false positive rate, etc.).
    The constraint is computed as the difference between the statistic for each group and the mean statistic across all groups."""
    def __init__(self, loss: Callable, uses_labels: bool, as_logits: bool = False):
        super().__init__(
            fn=self.compute_constraints,
            m_fn=lambda n_groups: 1
        )
        self.loss = loss
        self.as_logits = as_logits
        if self.as_logits:
            raise ValueError("`as_logits=True`is not supported for the fairret loss constraint, since the loss should already be computed on the logits.")
        self.uses_labels = uses_labels

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels):
        if self.uses_labels:
            loss = self.loss(batch_out, batch_sens, batch_labels)
        else:
            loss = self.loss(batch_out, batch_sens)

        return loss.unsqueeze(0)




### (OBJECTIVE) LOSS CONSTRAINTS ###

class LossPairwise(ConstraintMetadata):
    """Wrapper class for a fairness constraint that enforces equal loss across groups.
    The constraint is computed as the pairwise difference between the losses for each group."""
    def __init__(self, loss: Callable = None, abs_diff: bool = False):
        """
        Args:
            loss (Callable): A function that computes the loss for each sample in the batch; must be **unaggregated** (i.e., reduction='none')
            If not provided, the constraint will expect the loss to be precomputed and passed as an argument to the compute_constraints function.
        """
        super().__init__(
            fn=self.compute_constraints,
            m_fn=lambda n_groups: n_groups * (n_groups - 1) if not self.abs_diff else n_groups * (n_groups - 1) // 2
        )
        self.abs_diff = abs_diff
        if self.abs_diff:
            raise NotImplementedError("abs_diff=True is not implemented yet.")
        self.loss = loss

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels, loss = None):
        if loss is None:
            loss = self.loss(batch_out, batch_labels)
        per_group_losses = _get_normalized_per_group_losses(loss, batch_sens)
        constraints = ((per_group_losses.unsqueeze(1) - per_group_losses.unsqueeze(0)).to(torch.float))    
        mask = ~torch.eye(batch_sens.shape[-1], dtype=torch.bool)
        constraints = constraints[mask]

        return constraints
    


class LossMean(ConstraintMetadata):
    """Wrapper class for a fairness constraint that enforces equal loss across groups.
    The constraint is computed as the difference between the loss for each group and the overall loss."""
    def __init__(self, loss: Callable = None):
        """
        Args:
            loss (Callable): A function that computes the loss for each sample in the batch; must be **unaggregated** (i.e., reduction='none')
            If not provided, the constraint will expect the loss to be precomputed and passed as an argument to the compute_constraints function.
        """
        super().__init__(
            fn=self.compute_constraints,
            m_fn=lambda n_groups: n_groups
        )
        self.loss = loss

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels, loss = None):
        if loss is None:
            loss = self.loss(batch_out, batch_labels)
        per_group_losses = _get_normalized_per_group_losses(loss, batch_sens)
        mean_loss = loss.mean()
        return torch.abs(per_group_losses - mean_loss).squeeze()


def _get_normalized_per_group_losses(loss, sens_onehot):
    return loss.T @ sens_onehot / sens_onehot.sum(dim=0)


def weight_constraint(model, out, batch_sens, batch_labels):
    norms = []
    for param in model.parameters():
        norm = torch.linalg.norm(param)
        norms.append(norm.unsqueeze(0))
    
    return torch.concat(norms)