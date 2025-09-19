import numpy as np
import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, subgroup_onehot=None, subgroup_indices=None, batch_size=1, drop_last=True):
        """
        A Sampler that yields an equal number of samples from each subgroups specified with either one-hot encoding or indices.
        Specifically, if given`S`subgroups and batch size of`N`, yields a batch consisting of`N//S`samples of each subgroup, sorted by subgroup, but shuffled within each subgroup.
        
        Args:
            subset_indices (list of list): List of indices for each subset. Defaults to None.
            subgroup_onehot (tensor): Tensor of one-hot-encoded subgroups memberships of shape`(N, S)`, where`S`is the number of subgroups. Defaults to None.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If`True`, drop the last incomplete batch. Supports only`True`for now.
        """
        
        if subgroup_indices is None and subgroup_onehot is None:
            raise ValueError(f"Exactly one of`subgroup_indices`,`subgroup_onehot`must be`None`")
        
        if subgroup_onehot is not None:
            subgroup_onehot = subgroup_onehot.numpy()
            subgroup_indices = [
                np.argwhere(subgroup_onehot[:, gr] == 1).squeeze() for gr in range(subgroup_onehot.shape[-1])
            ]
            
        self.subset_indices = subgroup_indices
        self.batch_size = batch_size
        if drop_last is False:
            raise NotImplementedError('drop_last=False not supported yet!')
        self.drop_last = drop_last
        self.n_subsets = len(subgroup_indices)
        self.subset_sizes = [len(indices) for indices in subgroup_indices]
        self.n_samples_per_subset = batch_size // self.n_subsets
        # Check if batch_size is divisible by the number of subsets
        assert batch_size % self.n_subsets == 0, (
            f"Batch size ({batch_size}) must be divisible by the number of subsets ({self.n_subsets})."
        )

    def __iter__(self):
        # Shuffle indices within each subset
        shuffled_subset_indices = [torch.randperm(len(indices)).tolist() for indices in self.subset_indices]

        # Calculate the maximum number of batches per subset
        max_batches = min(len(indices) // self.n_samples_per_subset for indices in self.subset_indices)
        if not self.drop_last and any(len(indices) % self.n_samples_per_subset != 0 for indices in self.subset_indices):
            max_batches += 1  # Include partial batches if drop_last is False
        # TODO: randomly permute the batch as well
        # Yield balanced batches
        for batch_idx in range(max_batches):
            batch = []
            for subset_idx in range(self.n_subsets):
                start = batch_idx * self.n_samples_per_subset
                end = start + self.n_samples_per_subset
                subset_batch_indices = shuffled_subset_indices[subset_idx][start:end]
                batch.extend([self.subset_indices[subset_idx][i] for i in subset_batch_indices])

            # Yield the global indices for the batch
            yield batch

    def __len__(self):
        if self.drop_last:
            return min(len(indices) // self.n_samples_per_subset for indices in self.subset_indices)
        else:
            return max((len(indices) + self.n_samples_per_subset - 1) // self.n_samples_per_subset
                       for indices in self.subset_indices)