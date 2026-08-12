"""Fairness-constrained learning problems for E2.

Two datasets, both already exercised by ``benchmark/new_bench`` so the
preprocessing is not novel here:

* ``income``  -- folktables ACSIncome, sensitive attribute = the *cross product*
  of two ACS columns (default marital status x sex, 6 groups).
* ``dutch``   -- Dutch census 2001, sensitive attribute = sex x age (18 groups).

and two constraint shapes, both from ``new_bench/constraints.py`` (imported by
path rather than reimplemented):

* ``pairwise``  -- the positive-rate gap between every *ordered* pair of groups,
  ``m = G(G-1)``. The demanding variant: every pair must hold.
* ``agg``       -- one aggregated fairret norm-loss constraint, ``m = 1``.

Why this file exists rather than calling ``_loaders.load_data_FT``: that function
resolves the ACS download root relative to the *current working directory* (it is
meant to run from ``benchmark/new_bench/``) and bundles a K-fold split E2 does
not want. What is reusable -- ``comb_cat_dummies``, ``get_data_dutch``, and the
constraint classes -- is reused.

The constraint convention is the package's: ``problem.constraints(out, sens)``
returns a flat tensor that should be ``<= 0``, i.e. the raw statistic gap minus
the declared bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from humancompatible.train.fairness.utils import BalancedBatchSampler
from paper._harness import REPO_ROOT, load_benchmark_module

# The ACS PUMS files live where new_bench downloaded them. Only the states whose
# ``psam_p*.csv`` is present can be loaded offline; see ``available_states()``.
ACS_ROOT = REPO_ROOT / "benchmark" / "new_bench" / "data"

# ACS numeric state FIPS -> postal code, for the states this repo has on disk.
_FIPS = {"12": "FL", "51": "VA"}

TEST_SIZE = 0.2
# Samples per group in a training minibatch. BalancedBatchSampler requires
# ``batch_size % n_groups == 0`` and puts ``batch_size // n_groups`` of each group
# in every batch, so the batch size is *derived* from the group count rather than
# fixed: a per-group constraint estimated from 2 samples per group is noise, and
# a fixed batch size silently becomes that as soon as the group count grows.
PER_GROUP = 8


@dataclass
class FairnessProblem:
    """A constrained ERM problem: ``min E[BCE] s.t. E[gap_j] - bound <= 0``.

    :param raw_constraints: ``(logits, sens_onehot) -> tensor`` of raw statistic
        gaps, *before* the bound is subtracted. Kept separate from
        :meth:`constraints` so the reported quantity ("the positive-rate gap is
        0.07") is the interpretable one and the bound stays a declared knob.
    :param train: ``(X, A, y)`` on the training split, full-batch, for evaluating
        KKT metrics at a frozen iterate.
    :param test: the same on the held-out split. The gap between train and test
        violation is E2's headline and the thing a synthetic benchmark cannot
        show.
    """

    name: str
    m: int
    bound: float
    n_groups: int
    n_features: int
    raw_constraints: Callable[[Tensor, Tensor], Tensor]
    train: tuple[Tensor, Tensor, Tensor]
    test: tuple[Tensor, Tensor, Tensor]
    loader: DataLoader
    generator: torch.Generator
    notes: str = ""

    def reseed(self, seed: int) -> None:
        """Reset the batch-order generator.

        Required before every run: the sampler's generator is created once, with
        the problem, so without this the *n*-th run over the same problem starts
        from wherever the (n-1)-th left it. Two methods would then see different
        data orders and a "seed" would control only the model init -- which
        silently breaks any comparison, including a supposedly
        gradient-identical control.
        """
        self.generator.manual_seed(seed)

    def make_model(self) -> nn.Module:
        """The 64-32-1 MLP of ``new_bench/models.py``.

        Inlined rather than imported: it is five lines, and a path import for
        five lines costs more than it saves.
        """
        return nn.Sequential(
            nn.Linear(self.n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    @staticmethod
    def objective(logits: Tensor, labels: Tensor) -> Tensor:
        """Mean binary cross-entropy on logits."""
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def constraints(self, logits: Tensor, sens: Tensor) -> Tensor:
        """``c <= 0`` form: the raw gaps minus the declared bound."""
        return self.raw_constraints(logits, sens).reshape(-1) - self.bound

    def violation(self, logits: Tensor, sens: Tensor) -> float:
        """``max_j c_j`` -- positive iff some constraint is violated."""
        return float(self.constraints(logits, sens).detach().max())

    @property
    def batch_size(self) -> int:
        return PER_GROUP * self.n_groups


# --------------------------------------------------------------------------- #
# constraint shapes
# --------------------------------------------------------------------------- #


def _build_raw_constraints(shape: str, statistic: str = "PositiveRate"):
    """``(logits, sens) -> raw gaps``, plus ``m`` as a function of group count.

    The classes come from ``new_bench/constraints.py`` so the pairwise-mask and
    fairret-loss algebra has exactly one implementation in the repo.
    """
    constraints_mod = load_benchmark_module("constraints")
    import fairret.loss
    import fairret.statistic

    stat = getattr(fairret.statistic, statistic)()

    if shape == "pairwise":
        meta = constraints_mod.FairretPairwise(statistic=stat, uses_labels=False)
    elif shape == "agg":
        meta = constraints_mod.FairretAgg(
            loss=fairret.loss.NormLoss(stat), uses_labels=False
        )
    else:
        raise ValueError(f"unknown constraint shape {shape!r} (pairwise|agg)")

    # ``compute_constraints`` takes (model, out, sens, labels); neither the model
    # nor the labels are used by these two shapes.
    def raw(logits: Tensor, sens: Tensor) -> Tensor:
        return meta.compute_constraints(None, logits, sens, None)

    return raw, meta.m_fn


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


def available_states() -> list[str]:
    """States whose ACS PUMS file is on disk, so can be loaded offline."""
    year_dir = ACS_ROOT / "2018" / "1-Year"
    found = []
    for path in sorted(year_dir.glob("psam_p*.csv")):
        code = _FIPS.get(path.stem.removeprefix("psam_p"))
        if code:
            found.append(code)
    return found


def _load_income(states, sens_attrs):
    """ACSIncome features, crossed-one-hot sensitive groups, binary labels."""
    import pandas as pd
    from folktables import ACSDataSource, ACSIncome, BasicProblem, generate_categories

    source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=str(ACS_ROOT)
    )
    # download=False keeps this runnable on a compute node with no network; the
    # files are the ones new_bench already fetched.
    acs = source.get_data(states=list(states), download=False)
    definition = source.get_definitions(download=False)

    problem = BasicProblem(
        features=ACSIncome.features,
        target=ACSIncome.target,
        target_transform=ACSIncome.target_transform,
        group=list(sens_attrs),
        group_transform=lambda x: pd.get_dummies(x, columns=list(sens_attrs)),
        preprocess=ACSIncome._preprocess,
        postprocess=ACSIncome._postprocess,
    )
    categories = generate_categories(
        features=problem.features, definition_df=definition
    )
    features_df, labels_df, sens_df = problem.df_to_pandas(
        acs, categories=categories, dummies=True
    )

    if "MAR" in sens_attrs:
        # Merge the three small marital statuses (separated / widowed / divorced)
        # into one, as new_bench does: on a single state the tails are a few
        # thousand rows and a per-group rate estimated from them is noise.
        sens_df["MAR_2"] = sens_df["MAR_2"] + sens_df["MAR_4"] + sens_df["MAR_5"]
        sens_df = sens_df.drop(columns=["MAR_4", "MAR_5"])

    if len(sens_attrs) > 1:
        loaders = load_benchmark_module("_loaders")
        sens_df = loaders.comb_cat_dummies(sens_df)

    # Drop the sensitive columns from the features: the constraint is about them,
    # so leaving them in makes the model's job trivially different.
    drop = [c for c in features_df.columns if c.startswith(tuple(sens_attrs))]
    features = features_df.drop(columns=drop).to_numpy(dtype="float32")
    groups = sens_df.to_numpy(dtype="float32")
    labels = labels_df.to_numpy(dtype="float32")
    return features, groups, labels


def _load_dutch():
    """Dutch census, loaded from the parquet the repo already has.

    ``fairml_datasets`` hardcodes its cache to a *cwd-relative* ``Path("cache")``
    with no environment override, and its ``dataset`` module binds that path by
    from-import -- so patching ``file_handling`` has no effect, and running from
    anywhere but ``benchmark/`` silently re-downloads the ARFF over the network.
    Point the two names it actually reads at the committed cache instead, so E2
    runs offline and always on the same bytes.
    """
    from fairml_datasets import dataset as fairml_dataset

    cache = REPO_ROOT / "benchmark" / "cache"
    fairml_dataset.DATASET_CACHE_DIR = cache / "datasets"
    fairml_dataset.DOWNLOAD_CACHE_DIR = cache / "downloads"

    loaders = load_benchmark_module("_loaders")
    features, groups, labels, _ = loaders.get_data_dutch(
        drop_small_groups=True, print_stats=False
    )
    return (
        features.astype("float32"),
        groups.astype("float32"),
        labels.astype("float32"),
    )


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def _split_and_scale(features, groups, labels, *, seed, device):
    """Stratified train/test split, scaler fit on train only."""
    strat = groups.argmax(1)
    idx_train, idx_test = train_test_split(
        np.arange(len(features)), test_size=TEST_SIZE, random_state=seed,
        stratify=strat,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[idx_train])
    x_test = scaler.transform(features[idx_test])

    def to_tensor(array):
        return torch.as_tensor(np.ascontiguousarray(array),
                               dtype=torch.get_default_dtype(), device=device)

    train = (to_tensor(x_train), to_tensor(groups[idx_train]),
             to_tensor(labels[idx_train]))
    test = (to_tensor(x_test), to_tensor(groups[idx_test]),
            to_tensor(labels[idx_test]))
    return train, test


def _drop_small_groups(groups, minimum):
    """Keep only groups with at least ``minimum`` members; return a row mask and
    the reduced one-hot.

    Necessary for any crossed attribute with a long tail -- SEX x RAC1P on one
    state has groups of 5 rows, and BalancedBatchSampler would happily put one
    of them in every batch, making that constraint's estimate pure noise.
    """
    sizes = groups.sum(0)
    keep = sizes >= minimum
    if not keep.any():
        raise ValueError(f"no group has {minimum} members; sizes={sizes.tolist()}")
    rows = groups[:, keep].any(1) if groups.dtype == bool else groups[:, keep].sum(1) > 0
    return rows, keep


def build(dataset: str = "income", shape: str = "pairwise", *, bound: float = 0.05,
          states=("FL",), sens_attrs=("MAR", "SEX"), min_group: int = 500,
          split_seed: int = 0, batch_seed: int = 0, device="cpu",
          balanced: bool = True) -> FairnessProblem:
    """Assemble one E2 problem.

    :param bound: the fairness bound. 0.05 for ``pairwise`` (a 5-point
        positive-rate gap); the aggregate shape is a norm over groups and needs a
        looser one, so pass it explicitly.
    :param min_group: groups smaller than this are dropped, along with their rows.
    :param split_seed: fixes the train/test partition. Kept separate from
        ``batch_seed`` so the optimization-side variance (model init, batch order)
        can be measured against a *fixed* test set -- otherwise a method's test
        violation moves for two unrelated reasons at once.
    """
    if dataset == "income":
        features, groups, labels = _load_income(states, sens_attrs)
        notes = f"ACSIncome {'+'.join(states)}, sens={'x'.join(sens_attrs)}"
    elif dataset == "dutch":
        features, groups, labels = _load_dutch()
        notes = "Dutch census 2001, sens=sex x age"
    else:
        raise ValueError(f"unknown dataset {dataset!r} (income|dutch)")

    rows, keep = _drop_small_groups(groups, min_group)
    dropped = int((~keep).sum())
    if dropped:
        features, groups, labels = features[rows], groups[rows][:, keep], labels[rows]
        notes += f", dropped {dropped} group(s) with < {min_group} members"

    n_groups = groups.shape[1]
    raw, m_fn = _build_raw_constraints(shape)
    train, test = _split_and_scale(features, groups, labels,
                                   seed=split_seed, device=device)

    x_train, a_train, y_train = train
    batch_size = PER_GROUP * n_groups
    generator = torch.Generator(device="cpu").manual_seed(batch_seed)
    dataset_tensors = TensorDataset(x_train, a_train, y_train)
    if balanced:
        sampler = BalancedBatchSampler(
            group_onehot=a_train, batch_size=batch_size, drop_last=True,
            generator=generator,
        )
        loader = DataLoader(dataset_tensors, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset_tensors, batch_size=batch_size, shuffle=True,
                            drop_last=True, generator=generator)

    return FairnessProblem(
        name=f"{dataset}_{shape}",
        m=m_fn(n_groups),
        bound=bound,
        n_groups=n_groups,
        n_features=features.shape[1],
        raw_constraints=raw,
        train=train,
        test=test,
        loader=loader,
        generator=generator,
        notes=notes,
    )
