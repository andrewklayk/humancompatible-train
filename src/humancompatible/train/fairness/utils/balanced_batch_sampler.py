import numpy as np
import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(
            self,
            group_onehot=None,
            group_indices=None,
            batch_size=1,
            drop_last=True,
            extend_groups=None,
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
            raise ValueError(f"Exactly one of`group_indices`,`group_onehot`must be`None`")
        
        # convert one-hot group masks (fairret style) to group indices
        if group_onehot is not None:
            group_onehot = group_onehot.numpy()
            group_indices = [
                np.argwhere(group_onehot[:, gr] == 1).squeeze() for gr in range(group_onehot.shape[-1])
            ]
        
        self.batch_size = batch_size
        if drop_last is False:
            raise NotImplementedError('drop_last=False not supported yet!')
        self._drop_last = drop_last
        self._n_groups = len(group_indices) 
        self._n_samples_per_group = batch_size // self._n_groups

        # Check if batch_size is divisible by the number of groups
        assert batch_size % self._n_groups == 0, (
            f"Batch size ({batch_size}) must be divisible by the number of groups ({self._n_groups})."
        )

        # extend group indices for groups specified in extend_groups
        self._group_indices = group_indices
        self._group_sizes = [len(indices) for indices in group_indices]
        max_group_size = max(self._group_sizes)
        for group in extend_groups:
            # tile and slice to match the size of the largest group
            self._group_indices[group] = np.tile(group_indices[group], int(max_group_size / len(group_indices[group])))[:max_group_size]

    def __iter__(self):
        # Shuffle indices within each group
        shuffled_group_indices = [torch.randperm(len(indices)).tolist() for indices in self._group_indices]

        # Calculate the maximum number of batches per group
        max_batches = min(len(indices) // self._n_samples_per_group for indices in self._group_indices)
        if not self._drop_last and any(len(indices) % self._n_samples_per_group != 0 for indices in self._group_indices):
            max_batches += 1  # Include partial batches if drop_last is False
        # TODO: randomly permute the batch as well
        # Yield balanced batches
        for batch_idx in range(max_batches):
            batch = []
            for group_idx in range(self._n_groups):
                start = batch_idx * self._n_samples_per_group
                end = start + self._n_samples_per_group
                group_batch_indices = shuffled_group_indices[group_idx][start:end]
                batch.extend([self._group_indices[group_idx][i] for i in group_batch_indices])

            # Yield the global indices for the batch, shuffled
            shuffled_batch_indices = torch.randperm(len(batch))
            yield batch[shuffled_batch_indices]

    def __len__(self):
        if self._drop_last:
            return min(len(indices) // self._n_samples_per_group for indices in self._group_indices)
        else:
            return max((len(indices) + self._n_samples_per_group - 1) // self._n_samples_per_group
                       for indices in self._group_indices)