import unittest
import torch
from torch.utils.data import TensorDataset, Subset, DataLoader
from humancompatible.train.fairness.utils import BalancedBatchSampler

class TestBalancedBatchSampler(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([[i, i+1] for i in range(10)])
        self.labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        self.dataset = TensorDataset(self.data, self.labels)
        self.subset_indices = [
            [0, 1],  # Class 0
            [2, 3, 4],  # Class 1
            [5, 6, 7, 8, 9],  # Class 2
        ]
        self.subset_onehot = torch.tensor([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        ]).T

    def test_batch_size_divisible(self):
        with self.assertRaises(AssertionError):
            BalancedBatchSampler(subgroup_indices=self.subset_indices, batch_size=4, drop_last=True)

    def test_onehot_init(self):
        sampler = BalancedBatchSampler(subgroup_onehot=self.subset_onehot, batch_size=3)
        self.assertListEqual(
            [i.tolist() for i in sampler.subset_indices],
            self.subset_indices
        )

    def test_iter(self):
        sampler = BalancedBatchSampler(subgroup_indices=self.subset_indices, batch_size=6, drop_last=True)
        batches = list(sampler)
        self.assertEqual(len(batches), 1)  # Only 1 full batch of size 6 (2+2+2)
        self.assertEqual(len(batches[0]), 6)

    def test_len_drop_last_true(self):
        sampler = BalancedBatchSampler(subgroup_indices=self.subset_indices, batch_size=6, drop_last=True)
        self.assertEqual(len(sampler), 1)

    def test_balanced_batches(self):
        sampler = BalancedBatchSampler(subgroup_indices=self.subset_indices, batch_size=6, drop_last=True)
        batch = next(iter(sampler))
        # Check that each subset contributes 2 samples
        self.assertEqual(len([i for i in batch if i in self.subset_indices[0]]), 2)
        self.assertEqual(len([i for i in batch if i in self.subset_indices[1]]), 2)
        self.assertEqual(len([i for i in batch if i in self.subset_indices[2]]), 2)

## TODO: test edge cases!

class TestDataLoaderIntegration(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([[i, i+1] for i in range(10)])
        self.labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        self.dataset = TensorDataset(self.data, self.labels)
        self.subset_indices = [
            [0, 1],  # Class 0
            [2, 3, 4],  # Class 1
            [5, 6, 7, 8, 9],  # Class 2
        ]
        self.subsets = [Subset(self.dataset, indices) for indices in self.subset_indices]

    def test_dataloader(self):
        sampler = BalancedBatchSampler(subgroup_indices=self.subset_indices, batch_size=6, drop_last=True)
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=sampler
        )
        batch_data, batch_labels = next(iter(dataloader))
        self.assertEqual(batch_data.shape, (6, 2))
        self.assertEqual(len(batch_labels), 6)
        # Check balance: 2 samples from each class
        self.assertEqual((batch_labels == 0).sum().item(), 2)
        self.assertEqual((batch_labels == 1).sum().item(), 2)
        self.assertEqual((batch_labels == 2).sum().item(), 2)

if __name__ == "__main__":
    unittest.main()