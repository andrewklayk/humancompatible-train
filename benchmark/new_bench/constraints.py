"""Constraint definitions for the benchmark.

Merges the original ``constraint_meta.py`` (the ``ConstraintMetadata`` wrapper
classes) and the ``weight_constraint`` helper from ``constraints.py``. The math
is unchanged; only dead imports/functions were dropped.

A constraint object exposes two things, consumed by ``tasks.py``:
  * ``compute_constraints(model, out, sens, labels[, loss])`` -> tensor of raw
    constraint values (one entry per scalar constraint).
  * ``m_fn(n_groups) -> int`` -> number of scalar constraints for ``n_groups``.
"""
from typing import Callable, Any
from dataclasses import dataclass

import torch


@dataclass
class ConstraintMetadata:
    """Wrapper for a constraint: the function that computes it and a function
    that returns the number of scalar constraints given the number of groups."""
    fn: Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    m_fn: Callable[[int], int]


### FAIRRET (statistic-based) CONSTRAINTS ###

class FairretPairwise(ConstraintMetadata):
    """Pairwise difference of a fairret statistic between every ordered pair of groups."""
    def __init__(self, statistic: Callable, uses_labels: bool, abs_diff: bool = False, as_logits: bool = True):
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
        if self.as_logits:
            batch_out = torch.sigmoid(batch_out)
        if self.uses_labels:
            stat_pergroup = self.statistic(batch_out, batch_sens, batch_labels)
        else:
            stat_pergroup = self.statistic(batch_out, batch_sens)
        constraints = (stat_pergroup.unsqueeze(1) - stat_pergroup.unsqueeze(0))
        mask = ~torch.eye(batch_sens.shape[-1], dtype=torch.bool)
        return constraints[mask]


class FairretMean(ConstraintMetadata):
    """Absolute difference between each group's statistic and the overall statistic."""
    def __init__(self, statistic: Callable, uses_labels: bool, as_logits: bool = True):
        super().__init__(fn=self.compute_constraints, m_fn=lambda n_groups: n_groups)
        self.as_logits = as_logits
        self.statistic = statistic
        self.uses_labels = uses_labels

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels):
        if self.as_logits:
            batch_out = torch.sigmoid(batch_out)
        if self.uses_labels:
            stat_pergroup = self.statistic(batch_out, batch_sens, batch_labels)
            mean_stat = self.statistic(batch_out, sens=None, labels=batch_labels)
        else:
            stat_pergroup = self.statistic(batch_out, batch_sens)
            mean_stat = self.statistic(batch_out, sens=None)
        return torch.abs(stat_pergroup - mean_stat)


class FairretAgg(ConstraintMetadata):
    """Single aggregated constraint from a fairret loss (e.g. NormLoss)."""
    def __init__(self, loss: Callable, uses_labels: bool, as_logits: bool = True):
        super().__init__(fn=self.compute_constraints, m_fn=lambda n_groups: 1)
        self.loss = loss
        self.as_logits = as_logits
        if not self.as_logits:
            raise ValueError("`as_logits=False` is not supported for the fairret loss constraint, "
                             "since the loss should already be computed on the logits.")
        self.uses_labels = uses_labels

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels):
        if self.uses_labels:
            loss = self.loss(batch_out, batch_sens, batch_labels)
        else:
            loss = self.loss(batch_out, batch_sens)
        return loss.unsqueeze(0)


### (OBJECTIVE) LOSS CONSTRAINTS ###

class LossPairwise(ConstraintMetadata):
    """Pairwise difference of per-group mean loss between every ordered pair of groups."""
    def __init__(self, loss: Callable = None, abs_diff: bool = False):
        super().__init__(
            fn=self.compute_constraints,
            m_fn=lambda n_groups: n_groups * (n_groups - 1) if not self.abs_diff else n_groups * (n_groups - 1) // 2
        )
        self.abs_diff = abs_diff
        if self.abs_diff:
            raise NotImplementedError("abs_diff=True is not implemented yet.")
        self.loss = loss

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels, loss=None):
        if loss is None:
            loss = self.loss(batch_out, batch_labels)
        per_group_losses = _get_normalized_per_group_losses(loss, batch_sens).squeeze()
        constraints = (per_group_losses.unsqueeze(1) - per_group_losses.unsqueeze(0))
        mask = ~torch.eye(batch_sens.shape[-1], dtype=torch.bool)
        return constraints[mask]


class LossMean(ConstraintMetadata):
    """Absolute difference between each group's mean loss and the overall loss."""
    def __init__(self, loss: Callable = None):
        super().__init__(fn=self.compute_constraints, m_fn=lambda n_groups: n_groups)
        self.loss = loss

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels, loss=None):
        if loss is None:
            loss = self.loss(batch_out, batch_labels)
        per_group_losses = _get_normalized_per_group_losses(loss, batch_sens)
        mean_loss = loss.mean()
        return torch.abs(per_group_losses - mean_loss).squeeze()


def _get_normalized_per_group_losses(loss, sens_onehot):
    return loss.unsqueeze(0) @ sens_onehot / sens_onehot.sum(dim=0)


### WEIGHT-NORM CONSTRAINT ###

def weight_constraint(model, out, batch_sens, batch_labels):
    norms = []
    for param in model.parameters():
        norms.append(torch.linalg.norm(param).unsqueeze(0))
    return torch.concat(norms)
