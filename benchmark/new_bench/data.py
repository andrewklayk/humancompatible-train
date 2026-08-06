"""Uniform data layer.

``build_data(cfg_data, batch_size, device, cv_seed, n_folds, fold, init_seed)``
turns a ``cfg.data`` node into a uniform bundle, regardless of dataset. The
per-dataset quirks live behind the registry; everything downstream (task, train
loop) only sees a ``DataBundle``. Cross-validation (fixed stratified test +
K-fold dev) and the split/init seed separation live in ``_loaders``.
"""
from dataclasses import dataclass
from typing import Any, Optional

import torch

import _loaders


@dataclass
class DataBundle:
    train_loader: torch.utils.data.DataLoader
    # Validation / test are EITHER a (features, sens, labels) tensor tuple
    # (tabular: evaluated in one shot) OR a DataLoader (image: evaluated in batches).
    # Both are None when approach='opt' (full dataset, train only).
    val: Optional[Any]
    test: Optional[Any]
    train_full: tuple            # (features_train, sens_train, labels_train)
    n_groups: int                # size of the sensitive/class one-hot
    input_shape: Optional[int]   # feature dimension for tabular models; None for images
    is_image: bool
    # Fixed sample for the end-of-epoch full-batch optimality eval (approach='opt'):
    # the whole train set (tabular) or a fixed class-balanced subsample (images);
    # None when not applicable (ml mode). See train.evaluate_optimality.
    opt_eval: Optional[Any] = None


def _build_tabular(loader_out, approach):
    (train_loader, _val_loader, _test_loader), train_full, val_full, test_full = loader_out
    sens_train = train_full[1]
    features_train = train_full[0]
    return DataBundle(
        train_loader=train_loader,
        val=val_full,
        test=test_full,
        train_full=train_full,
        n_groups=sens_train.shape[-1],
        input_shape=features_train.shape[1],
        is_image=False,
        # opt mode: train_full IS the whole training set -> exact full-batch eval.
        opt_eval=train_full if approach == "opt" else None,
    )


def _loader_cv(cv):
    """The split/seed kwargs shared by the loaders (drops the opt-eval knob)."""
    return {k: cv[k] for k in ("cv_seed", "n_folds", "fold", "init_seed", "approach")}


def _folktables(cfg, batch_size, device, cv):
    kwargs = {k: v for k, v in cfg.get("kwargs", {}).items()}
    return _build_tabular(_loaders.load_data_FT(
        batch_size, device, test_size=cfg.get("test_size", 0.2), **kwargs, **_loader_cv(cv)),
        cv["approach"])


def _dutch(cfg, batch_size, device, cv):
    return _build_tabular(_loaders.load_data_DUTCH(
        batch_size=batch_size, device=device, test_size=cfg.get("test_size", 0.4), **_loader_cv(cv)),
        cv["approach"])


def _folktables_norm(cfg, batch_size, device, cv):
    # weight-norm task: same folktables features, SEX dropped.
    return _build_tabular(_loaders.load_data_norm(
        batch_size, device, test_size=cfg.get("test_size", 0.2), **_loader_cv(cv)),
        cv["approach"])


def _cifar(num_classes):
    def _build(cfg, batch_size, device, cv):
        # cifar K-folds the train set (test_size unused -> canonical test is held out).
        train_loader, val_loader, test_loader, opt_eval = _loaders.load_data_cifar(
            num_classes=num_classes, device=device,
            cv_seed=cv["cv_seed"], n_folds=cv["n_folds"], fold=cv["fold"],
            init_seed=cv["init_seed"], approach=cv["approach"],
            opt_eval_size=cv["opt_eval_size"],
            # Balanced batches guarantee every class is present each step, so the
            # per-class-loss constraint is always well-defined. Without this, a
            # shuffled batch can miss a class -> 0/0 per-group loss -> NaN that
            # poisons the Lagrangian and, via Adam, every weight permanently.
            balanced=True,
        )
        # A single batch stands in for ``train_full`` (feature/group dims only).
        features, sens, labels = next(iter(train_loader))
        return DataBundle(
            train_loader=train_loader,
            val=val_loader,
            test=test_loader,        # canonical test split, held out before K-folding
            train_full=(features, sens, labels),
            n_groups=sens.shape[-1],
            input_shape=None,
            is_image=True,
            opt_eval=opt_eval,       # fixed class-balanced subsample (opt mode) or None
        )
    return _build


# Registry: cfg.data.name -> builder(cfg_data, batch_size, device, cv) -> DataBundle
DATA_BUILDERS = {
    "folktables": _folktables,
    "dutch": _dutch,
    "folktables_norm": _folktables_norm,
    "cifar10": _cifar(10),
    "cifar100": _cifar(100),
}


def build_data(cfg_data, batch_size, device, cv_seed, n_folds, fold, init_seed,
               approach="ml", opt_eval_size=10000) -> DataBundle:
    """``cv_seed``/``n_folds``/``fold`` fix the stratified test + K-fold partition;
    ``init_seed`` drives batch order. ``approach='opt'`` uses the full dataset as
    train with no val/test split. ``opt_eval_size`` caps the fixed class-balanced
    optimality-eval subsample for image tasks (ignored by tabular). See ``_loaders``."""
    name = cfg_data["name"]
    if name not in DATA_BUILDERS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(DATA_BUILDERS)}")
    cv = {"cv_seed": cv_seed, "n_folds": n_folds, "fold": fold, "init_seed": init_seed,
          "approach": approach, "opt_eval_size": opt_eval_size}
    return DATA_BUILDERS[name](cfg_data, batch_size, device, cv)
