import numpy as np
import torch
from torch.utils.data import Sampler
from math import ceil
from typing import Iterable, Optional


class BalancedBatchSampler(Sampler):
    def __init__(
        self,
        group_onehot: Optional[Iterable[Iterable[int]]] | torch.Tensor = None,
        group_indices: Optional[Iterable[Iterable[int]]] = None,
        batch_size: int = 1,
        drop_last: bool = True,
        extend_groups: Optional[Iterable[int]] = None,
    ):
        """
        A Sampler that yields an equal number of samples from each groups specified with either one-hot encoding or indices.
        Specifically, if given`S`groups and batch size of`N`, yields a batch consisting of`N//S`samples of each group, sorted by group, but shuffled within each group.

        Args:
            group_indices (list of list): List of indices for each group. Defaults to`None`.
            group_onehot (tensor): Tensor of one-hot-encoded groups memberships of shape`(N, S)`, where`S`is the number of groups. Defaults to`None`.
            batch_size (int): Number of samples per batch Defaults to 1.
            drop_last (bool): If`True`, drop the last incomplete batch. Supports only`True`for now. Defaults to`True`.
            extend_groups (list of int): Indices of groups which should be extended (shuffled with replacement). Defaults to`None`.
        """

        if group_indices is None and group_onehot is None:
            raise ValueError(
                f"Exactly one of`group_indices`,`group_onehot`must be`None`"
            )

        # convert one-hot group masks (fairret style) to group indices
        if group_onehot is not None:
            group_onehot = group_onehot.numpy()
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
        self._extend_groups = extend_groups

    def __iter__(self):
        shuffled_group_indices = []
        for group_id, group_indices in enumerate(self._group_indices):
            group_indices_tiled_shuffled = []
            # determine number of tiles
            if not self._extend_groups or group_id not in self._extend_groups:
                num_tiles = 1
            else:
                num_tiles = ceil(max(self._group_sizes) / self._group_sizes[group_id])
            # tile with random reorderings of list of indices of the group
            for _ in range(num_tiles):
                # shuffle
                indices_shuffled = torch.randperm(len(group_indices)).tolist()
                # add new shuffled tile to the indices
                group_indices_tiled_shuffled.extend(indices_shuffled)
            # cutoff at the length of max group
            group_indices_tiled_shuffled = group_indices_tiled_shuffled[
                : max(self._group_sizes)
            ]
            shuffled_group_indices.append(group_indices_tiled_shuffled)

        # Calculate the maximum number of batches per group
        max_batches = min(
            len(indices) // self._n_samples_per_group
            for indices in shuffled_group_indices
        )
        if not self.drop_last and any(
            len(indices) % self._n_samples_per_group != 0
            for indices in self._group_indices
        ):
            max_batches += 1  # Include partial batches if drop_last is False

        # Yield balanced batches
        for batch_idx in range(max_batches):
            batch = []
            for group_idx in range(self._n_groups):
                start = batch_idx * self._n_samples_per_group
                end = start + self._n_samples_per_group
                group_batch_indices = shuffled_group_indices[group_idx][start:end]
                batch.extend(
                    [self._group_indices[group_idx][i] for i in group_batch_indices]
                )
            # Yield the global indices for the batch, shuffled within the batch
            shuffled_batch_indices = torch.randperm(len(batch), dtype=int)
            yield [batch[i] for i in shuffled_batch_indices]

    def __len__(self):
        if self.drop_last:
            return min(
                len(indices) // self._n_samples_per_group
                for indices in self._group_indices
            )
        else:
            return max(
                (len(indices) + self._n_samples_per_group - 1)
                // self._n_samples_per_group
                for indices in self._group_indices
            )
