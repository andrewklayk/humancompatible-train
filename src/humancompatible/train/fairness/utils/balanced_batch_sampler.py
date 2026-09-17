import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterable, Optional


class BalancedBatchSampler(Sampler):
    """
    A Sampler that yields an equal number of samples from each groups specified with either one-hot encoding or indices.
    Specifically, if given`S`groups and batch size of`N`, yields a batch consisting of`N//S`samples of each group, sorted by group, but shuffled within each group.

    Groups listed in`extend_groups`are oversampled: their samples are reused across batches so that the epoch is
    as long as the largest group allows, instead of ending when the smallest group is exhausted. A group that is not
    extended still binds the epoch length. With`n = batch_size // S`samples per group per batch, the number of
    batches is`min_g(target_g) // n`, where`target_g`is the size of the largest group if group`g`is extended and
    the size of group`g`itself otherwise.

    Oversampling reuses a group's samples evenly: over one epoch, each sample of an extended group is drawn either
    `floor`or`ceil`of the per-sample average, and no sample is ever repeated within a single batch.

    :param group_indices: List of indices for each group. Defaults to`None`.
    :type group_indices: Iterable[Iterable[int]]
    :param group_onehot: Tensor of one-hot-encoded groups memberships of shape`(N, S)`, where`S`is the number of groups. Defaults to`None`.
    :type group_onehot: torch.Tensor
    :param batch_size: Number of samples per batch Defaults to 1.
    :type batch_size: int
    :param drop_last: If`True`, drop the last incomplete batch. Supports only`True`for now. Defaults to`True`.
    :type drop_last: bool
    :param extend_groups: Groups which should be extended (oversampled). Either the indices of those groups, or`True`for all of them. Defaults to`None`.
    :type extend_groups: bool | Iterable[int]
    :param generator: Optional ``torch.Generator`` for reproducible sampling. Its device is the one the shuffles are drawn on; a CPU generator is used when none is given.
    :type generator: torch.Generator
    """
    def __init__(
        self,
        group_onehot: Optional[torch.Tensor] = None,
        group_indices: Optional[Iterable[Iterable[int]]] = None,
        batch_size: int = 1,
        drop_last: bool = True,
        extend_groups: Optional[bool | Iterable[int]] = False,
        generator: Optional[torch.Generator]=None
    ):

        if group_indices is None and group_onehot is None:
            raise ValueError(
                f"Exactly one of`group_indices`,`group_onehot`must be`None`"
            )
        if group_indices is not None and group_onehot is not None:
            raise ValueError(
                f"Exactly one of`group_indices`,`group_onehot`must be`None`, got both"
            )

        # convert one-hot group masks (fairret style) to group indices
        if group_onehot is not None:
            group_onehot = group_onehot.cpu().numpy()
            group_indices = [
                np.argwhere(group_onehot[:, gr] == 1).squeeze()
                for gr in range(group_onehot.shape[-1])
            ]

        self._n_groups = len(group_indices)
        # Check if batch_size is divisible by the number of groups
        assert batch_size % self._n_groups == 0, (
            f"Batch size ({batch_size}) must be divisible by the number of groups ({self._n_groups})."
        )
        self.batch_size = batch_size
        self._n_samples_per_group = batch_size // self._n_groups
        assert all(
            [self._n_samples_per_group <= len(group) for group in group_indices]
        ), (
            f"Size of every group must be greater or equal to batch_size / number_of_groups to avoid repeating samples within a batch"
            + f"Got {self._n_samples_per_group} samples per group, {[len(group) for group in group_indices]} group lengths."
        )

        if drop_last is False:
            raise NotImplementedError("drop_last=False not supported yet!")
        self.drop_last = drop_last

        self._group_indices = group_indices
        self._group_sizes = [len(indices) for indices in group_indices]
        # `True` extends every group; an iterable names the groups to extend
        if extend_groups is True:
            self._extend_groups = frozenset(range(self._n_groups))
        elif extend_groups is None or extend_groups is False:
            self._extend_groups = frozenset()
        else:
            self._extend_groups = frozenset(extend_groups)

        self.generator = generator
        # randperm is drawn on the generator's device, otherwise the default one
        self._device = generator.device if generator is not None else None

    def _stream(self, size, length):
        """Stream of`length`group-local indices, in tiles of`_n_samples_per_group`.

        A tile never repeats an index, and each index is used`floor`or`ceil`of
        `length / size`times: every drawn permutation contributes each of its entries
        to the stream exactly once, either as`fill`or later off the deck.
        """
        n = self._n_samples_per_group
        out, deck = [], []
        while len(out) < length:
            tile, deck = deck[:n], deck[n:]
            if len(tile) < n:  # deck ran out mid-tile
                perm = torch.randperm(
                    size, generator=self.generator, device=self._device
                ).tolist()
                held = set(tile)
                fill = [i for i in perm if i not in held][: n - len(tile)]
                tile += fill
                deck = [i for i in perm if i not in fill]
            out.extend(tile)
        return out[:length]

    def __iter__(self):
        n = self._n_samples_per_group
        n_batches = len(self)
        # extended groups are streamed past their size, the rest get a permutation prefix
        streams = [
            self._stream(size, n_batches * n) for size in self._group_sizes
        ]
        for batch_idx in range(n_batches):
            start = batch_idx * n
            end = start + n
            batch = [
                self._group_indices[group_idx][i]
                for group_idx in range(self._n_groups)
                for i in streams[group_idx][start:end]
            ]
            # Yield the global indices for the batch, shuffled within the batch
            shuffled_batch_indices = torch.randperm(
                len(batch), generator=self.generator, device=self._device
            )
            yield [batch[i] for i in shuffled_batch_indices.tolist()]

    def __len__(self):
        # an extended group is streamed up to the largest group; the rest bind as they are
        return min(
            max(self._group_sizes) if group_idx in self._extend_groups
            else self._group_sizes[group_idx]
            for group_idx in range(self._n_groups)
        ) // self._n_samples_per_group

    @property
    def group_weights(self) -> torch.Tensor:
        """Inverse-propensity per-group weight (n_groups * population proportion), to
        pass as `weight=` to a mean-reduced loss and correct for this sampler's equal
        per-batch group representation."""
        sizes = torch.tensor(self._group_sizes, dtype=torch.float32)
        return self._n_groups * sizes / sizes.sum()
