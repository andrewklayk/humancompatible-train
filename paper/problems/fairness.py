"""Fairness-constrained learning problems for E2.

Two datasets, loaded independently of ``benchmark/new_bench`` (only the
constraint shapes are still reused from there -- see below):

* ``income``  -- folktables ACSIncome, sensitive attribute = the *cross product*
  of two ACS columns (default marital status x sex, 6 groups). A plain,
  configurable ``folktables`` call: ``root_dir`` and ``states`` are parameters,
  not hardcoded paths.
* ``dutch``   -- Dutch census 2001, sensitive attribute = sex x age (18 groups).

and two constraint shapes, both from ``new_bench/constraints.py`` (imported by
path rather than reimplemented, since the pairwise-mask and fairret-loss algebra
should have exactly one implementation in the repo):

* ``pairwise``  -- the positive-rate gap between every *ordered* pair of groups,
  ``m = G(G-1)``. The demanding variant: every pair must hold.
* ``agg``       -- one aggregated fairret norm-loss constraint, ``m = 1``.

The constraint convention is the package's: ``problem.constraints(out, sens)``
returns a flat tensor that should be ``<= 0``, i.e. the raw statistic gap minus
the declared bound.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from humancompatible.train.fairness.utils import BalancedBatchSampler
from paper._harness import REPO_ROOT, load_benchmark_module

# Where the ACS PUMS files live. Only the states whose ``psam_p*.csv`` is
# present can be loaded offline; see ``available_states()``.
ACS_ROOT = Path(__file__).resolve().parent / "data"

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
    per_group: int = PER_GROUP

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
    def objective(logits: Tensor, labels: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        """Mean binary cross-entropy on logits, optionally per-sample weighted."""
        return nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weight)

    def constraints(self, logits: Tensor, sens: Tensor) -> Tensor:
        """``c <= 0`` form: the raw gaps minus the declared bound."""
        return self.raw_constraints(logits, sens).reshape(-1) - self.bound

    def violation(self, logits: Tensor, sens: Tensor) -> float:
        """``max_j c_j`` -- positive iff some constraint is violated."""
        return float(self.constraints(logits, sens).detach().max())

    @property
    def batch_size(self) -> int:
        return self.per_group * self.n_groups


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


def _cross_group_dummies(df):
    """One-hot columns for the *cross product* of several one-hot attribute
    groups (e.g. ``MAR_1, MAR_2, ...`` x ``SEX_1, SEX_2`` -> ``MAR_1&SEX_1``,
    ``MAR_1&SEX_2``, ...): group columns by their ``_``-prefix, then AND every
    combination of one column per group (``.min`` over 0/1 columns is AND)."""
    import pandas as pd

    prefixes: dict[str, list[str]] = {}
    for col in df.columns:
        prefixes.setdefault(col.split("_")[0], []).append(col)

    combos = itertools.product(*prefixes.values())
    return pd.DataFrame(
        {"&".join(cols): df[list(cols)].min(axis=1) for cols in combos},
        index=df.index,
    )


@lru_cache(maxsize=1)  # parsing the ACS CSV dominates a multi-shape run
def _load_income(states, sens_attrs, root_dir=ACS_ROOT):
    """ACSIncome features, crossed-one-hot sensitive groups, binary labels.

    A plain, configurable ``folktables`` call with standard preprocessing:
    ``root_dir`` is wherever the ACS PUMS CSVs live
    (``root_dir/2018/1-Year/psam_p*.csv``, see ``available_states()``), loaded
    with ``download=False`` so this stays runnable on a compute node with no
    network.
    """
    import pandas as pd
    from folktables import ACSDataSource, ACSIncome, BasicProblem, generate_categories

    source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=str(root_dir)
    )
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
        # into one: on a single state the tails are a few thousand rows and a
        # per-group rate estimated from them is noise.
        sens_df["MAR_2"] = sens_df["MAR_2"] + sens_df["MAR_4"] + sens_df["MAR_5"]
        sens_df = sens_df.drop(columns=["MAR_4", "MAR_5"])

    if len(sens_attrs) > 1:
        sens_df = _cross_group_dummies(sens_df)

    # Drop the sensitive columns from the features: the constraint is about them,
    # so leaving them in makes the model's job trivially different.
    drop = [c for c in features_df.columns if c.startswith(tuple(sens_attrs))]
    features = features_df.drop(columns=drop).to_numpy(dtype="float32")
    groups = sens_df.to_numpy(dtype="float32")
    labels = labels_df.to_numpy(dtype="float32")
    return features, groups, labels


def _load_dutch():
    """Dutch census 2001, sensitive attribute sex x age (18 groups).

    ``fairml_datasets`` hardcodes its cache to a *cwd-relative* ``Path("cache")``
    with no environment override, and its ``dataset`` module binds that path by
    from-import -- so patching ``file_handling`` has no effect, and running from
    anywhere but ``benchmark/`` silently re-downloads the ARFF over the network.
    Point the two names it actually reads at the committed cache instead, so E2
    runs offline and always on the same bytes.
    """
    from fairml_datasets import Dataset, dataset as fairml_dataset

    cache = REPO_ROOT / "benchmark" / "cache"
    fairml_dataset.DATASET_CACHE_DIR = cache / "datasets"
    fairml_dataset.DOWNLOAD_CACHE_DIR = cache / "downloads"

    dataset = Dataset.from_id("dutch")
    df = dataset.load()
    # Ages 13-15 have too few members for a per-group rate to mean anything.
    df = df.drop(df[df.age.isin(["13", "14", "15"])].index)
    num_age_groups = 9

    target_column = dataset.get_target_column()
    df_transformed, _ = dataset.transform(df)
    sex_cols = ["sex_1", "sex_2"]

    labels = df_transformed[target_column].to_numpy(dtype="float32").reshape(-1, 1)
    features = df_transformed.drop(columns=[target_column, "age", *sex_cols])
    features = features.to_numpy(dtype="float32")

    sex_idx = df_transformed[sex_cols].to_numpy().argmax(axis=1).astype(int)
    age = df_transformed["age"].to_numpy().astype(int)
    group_idx = sex_idx * num_age_groups + (age - 4)
    groups = np.eye(num_age_groups * 2, dtype="float32")[group_idx]

    return features, groups, labels


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
          balanced: bool = True, per_group: int = PER_GROUP,
          extend_groups: Optional[bool] = True,
          root_dir=ACS_ROOT) -> FairnessProblem:
    """Assemble one E2 problem.

    :param bound: the fairness bound. 0.05 for ``pairwise`` (a 5-point
        positive-rate gap); the aggregate shape is a norm over groups and needs a
        looser one, so pass it explicitly.
    :param min_group: groups smaller than this are dropped, along with their rows.
    :param split_seed: fixes the train/test partition. Kept separate from
        ``batch_seed`` so the optimization-side variance (model init, batch order)
        can be measured against a *fixed* test set -- otherwise a method's test
        violation moves for two unrelated reasons at once.
    :param per_group: samples per group per batch; overrides the module default
        ``PER_GROUP``. Batch size is derived as ``per_group * n_groups``.
    :param extend_groups: forwarded to ``BalancedBatchSampler`` -- ``True`` reuses
        smaller groups' samples so the epoch runs as long as the *largest* group
        allows, instead of being cut short by the smallest one. ``None`` (default):
        epoch length bound by the smallest group.
    :param root_dir: where the ACS PUMS CSVs live (``income`` only); see
        ``available_states()``.
    """
    raw, m_fn = _build_raw_constraints(shape)
    if dataset == "income":
        features, groups, labels = _load_income(states, sens_attrs, root_dir)
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
    train, test = _split_and_scale(features, groups, labels,
                                   seed=split_seed, device=device)

    x_train, a_train, y_train = train
    batch_size = per_group * n_groups
    generator = torch.Generator(device="cpu").manual_seed(batch_seed)
    dataset_tensors = TensorDataset(x_train, a_train, y_train)
    if balanced:
        sampler = BalancedBatchSampler(
            group_onehot=a_train, batch_size=batch_size, drop_last=True,
            generator=generator, extend_groups=extend_groups,
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
        per_group=per_group,
    )
