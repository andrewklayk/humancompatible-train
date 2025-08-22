import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, subset_indices, batch_size, drop_last=False):
        """
        Args:
            subset_indices (list of list): List of indices for each subset.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch.
        """
        self.subset_indices = subset_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_subsets = len(subset_indices)
        self.subset_sizes = [len(indices) for indices in subset_indices]
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