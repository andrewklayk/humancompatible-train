"""Task layer: turn ``cfg.task`` + a ``DataBundle`` into everything the training
loop needs that is *task*-specific (not algorithm-specific):

    Task(model_factory, loss_fn, constraint_fn, m, bound, fuse_loss_constraint)

``constraints_to_eq`` and ``select_filter`` are deliberately NOT here -- they are
properties of the *algorithm*, see ``algorithms.py``.
"""
import importlib
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F

import models
from constraints import weight_constraint


@dataclass
class Task:
    model_factory: Callable[[], torch.nn.Module]
    loss_fn: Callable
    constraint_fn: Callable
    m: int
    bound: float
    fuse_loss_constraint: bool


def _build_model_factory(model_kind: str, bundle) -> Callable[[], torch.nn.Module]:
    if model_kind == "mlp":
        return partial(models.create_mlp, input_shape=bundle.input_shape,
                       latent_size1=64, latent_size2=32)
    if model_kind == "conv":
        return partial(models.create_conv, num_classes=bundle.n_groups)
    raise ValueError(f"Unknown model kind '{model_kind}' (expected 'mlp' or 'conv').")


def _build_loss(loss_kind: str) -> Callable:
    if loss_kind == "bce":
        return F.binary_cross_entropy_with_logits
    if loss_kind == "ce":
        return torch.nn.CrossEntropyLoss(reduction="none")
    raise ValueError(f"Unknown loss kind '{loss_kind}' (expected 'bce' or 'ce').")


def _build_constraint(constraint_cfg, n_groups, model_factory):
    """Returns (constraint_fn, m, bound, fuse_loss_constraint).

    Mirrors the original dispatch in run_gridsearch.main exactly.
    """
    kind = constraint_cfg["kind"]
    bound = constraint_cfg["bound"]
    kwargs = dict(constraint_cfg.get("kwargs", {}) or {})

    if kind == "weight_norm":
        # m = number of parameter tensors (MLP -> 6); computed instead of hardcoded.
        m = len(list(model_factory().parameters()))
        return weight_constraint, m, bound, False

    cls = importlib.import_module("constraints").__dict__[kind]

    if kind.startswith("Fairret"):
        statistic = importlib.import_module("fairret.statistic").__dict__[constraint_cfg["statistic"]]()
        if kind == "FairretAgg":
            fair_loss = importlib.import_module("fairret.loss").__dict__[constraint_cfg["loss"]]
            c = cls(loss=fair_loss(statistic), **kwargs)
        else:
            c = cls(statistic=statistic, **kwargs)
        fuse = False
    elif kind.startswith("Loss"):
        loss = importlib.import_module("torch.nn").__dict__[constraint_cfg["loss"]](reduction="none")
        c = cls(loss=loss, **kwargs)
        fuse = True
    else:
        raise ValueError(f"Unknown constraint kind '{kind}'.")

    m = c.m_fn(n_groups)
    return c.compute_constraints, m, bound, fuse


def build_task(cfg_task, bundle) -> Task:
    model_factory = _build_model_factory(cfg_task["model"], bundle)
    loss_fn = _build_loss(cfg_task["loss"])
    constraint_fn, m, bound, fuse = _build_constraint(
        cfg_task["constraint"], bundle.n_groups, model_factory
    )
    return Task(
        model_factory=model_factory,
        loss_fn=loss_fn,
        constraint_fn=constraint_fn,
        m=m,
        bound=bound,
        fuse_loss_constraint=fuse,
    )
