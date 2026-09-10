import unittest
from collections import Counter

import torch
from torch.utils.data import TensorDataset, Subset, DataLoader

from humancompatible.train.fairness.utils import BalancedBatchSampler


class TestBalancedBatchSampler(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([[i, i + 1] for i in range(10)])
        self.labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        self.dataset = TensorDataset(self.data, self.labels)
        self.subset_indices = [
            [0, 1],  # Class 0
            [2, 3, 4],  # Class 1
            [5, 6, 7, 8, 9],  # Class 2
        ]
        self.subset_onehot = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            ]
        ).T


    def test_extends_without_replacement(self):
        for _ in range(10):  # Run multiple times to check randomness
            sampler = BalancedBatchSampler(
                group_indices=self.subset_indices,
                batch_size=6,
                drop_last=True,
                extend_groups=[0, 1],
            )
            # Check that each extended group is sampled without replacement within a batch
            batches = []
            for i, batch in enumerate(sampler):
                self.assertEqual(len(set(batch)), len(batch))
                batches.extend(batch)


    def test_extend_num_batches(self):
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices,
            batch_size=6,
            drop_last=True,
            extend_groups=[0, 1],
        )
        # check correct number of batches in case of tiling
        i = 0
        for _ in iter(sampler):
            i += 1
        self.assertEqual(i, 2)

    def test_batch_size_divisible(self):
        with self.assertRaises(AssertionError):
            BalancedBatchSampler(
                group_indices=self.subset_indices, batch_size=4, drop_last=True
            )

    def test_onehot_init(self):
        sampler = BalancedBatchSampler(group_onehot=self.subset_onehot, batch_size=3)
        self.assertListEqual(
            [i.tolist() for i in sampler._group_indices], self.subset_indices
        )

    def test_iter(self):
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices, batch_size=6, drop_last=True
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 1)  # Only 1 full batch of size 6 (2+2+2)
        self.assertEqual(len(batches[0]), 6)

    def test_len_drop_last_true(self):
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices, batch_size=6, drop_last=True
        )
        self.assertEqual(len(sampler), 1)

    def test_balanced_batches(self):
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices, batch_size=6, drop_last=True
        )
        batch = next(iter(sampler))
        # Check that each subset contributes 2 samples
        self.assertEqual(len([i for i in batch if i in self.subset_indices[0]]), 2)
        self.assertEqual(len([i for i in batch if i in self.subset_indices[1]]), 2)
        self.assertEqual(len([i for i in batch if i in self.subset_indices[2]]), 2)

    def test_balanced_extended_batches(self):
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices,
            batch_size=6,
            drop_last=True,
            extend_groups=[0, 1, 2],
        )
        batch = next(iter(sampler))
        # Check that each subset contributes 2 samples
        self.assertEqual(len([i for i in batch if i in self.subset_indices[0]]), 2)
        self.assertEqual(len([i for i in batch if i in self.subset_indices[1]]), 2)
        self.assertEqual(len([i for i in batch if i in self.subset_indices[2]]), 2)

    

    def test_extend_large_batchsize(self):
        # check AssertionError on batch_size / n_groups > size of one of the groups
        with self.assertRaises(AssertionError):
            BalancedBatchSampler(
                group_indices=self.subset_indices,
                batch_size=9,
                drop_last=True,
                extend_groups=[0, 1],
            )

    def test_len_matches_iter_when_extended(self):
        # __len__ must agree with __iter__: groups of 2/3/5 all extended to 5, so 2 batches
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices,
            batch_size=6,
            drop_last=True,
            extend_groups=[0, 1],
        )
        self.assertEqual(len(sampler), 2)
        self.assertEqual(len(list(sampler)), len(sampler))

    def test_len_matches_iter_partial_extend(self):
        # only group 0 extended, so group 1 (size 3) still binds the epoch
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices,
            batch_size=6,
            drop_last=True,
            extend_groups=[0],
        )
        self.assertEqual(len(sampler), 1)
        self.assertEqual(len(list(sampler)), len(sampler))

    def test_extend_groups_true_equals_all_indices(self):
        # `True` is shorthand for every group
        batches = []
        for extend in (True, [0, 1, 2]):
            sampler = BalancedBatchSampler(
                group_indices=self.subset_indices,
                batch_size=6,
                drop_last=True,
                extend_groups=extend,
                generator=torch.Generator().manual_seed(0),
            )
            batches.append(list(sampler))
        self.assertListEqual(batches[0], batches[1])

    def test_extend_groups_falsy_does_not_extend(self):
        for extend in (None, False, []):
            sampler = BalancedBatchSampler(
                group_indices=self.subset_indices, batch_size=6, extend_groups=extend
            )
            self.assertEqual(len(sampler), 1)

    def test_extend_exposure_is_even(self):
        # each sample of an extended group is used floor/ceil of the average
        group_indices = [list(range(100)), list(range(100, 1100))]
        sampler = BalancedBatchSampler(
            group_indices=group_indices,
            batch_size=4,
            drop_last=True,
            extend_groups=True,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(len(sampler), 500)
        counts = Counter(i for batch in sampler for i in batch)
        for group in group_indices:
            per_sample = [counts[i] for i in group]
            self.assertEqual(len(per_sample), len(group))
            self.assertLessEqual(max(per_sample) - min(per_sample), 1)
        # the largest group is streamed exactly once through: full coverage, no repeats
        self.assertTrue(all(counts[i] == 1 for i in group_indices[1]))
        # the small group is reused to keep up
        self.assertTrue(all(counts[i] == 10 for i in group_indices[0]))

    def test_generator_reproducible(self):
        def batches(seed, extend):
            sampler = BalancedBatchSampler(
                group_indices=self.subset_indices,
                batch_size=6,
                drop_last=True,
                extend_groups=extend,
                generator=torch.Generator().manual_seed(seed),
            )
            return list(sampler)

        for extend in (None, [0, 1]):
            self.assertListEqual(batches(0, extend), batches(0, extend))
            self.assertNotEqual(batches(0, extend), batches(1, extend))

    def test_both_group_specs_raises(self):
        with self.assertRaises(ValueError):
            BalancedBatchSampler(
                group_onehot=self.subset_onehot,
                group_indices=self.subset_indices,
                batch_size=6,
            )

    def test_no_group_spec_raises(self):
        with self.assertRaises(ValueError):
            BalancedBatchSampler(batch_size=6)


class TestDataLoaderIntegration(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([[i, i + 1] for i in range(10)])
        self.labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        self.dataset = TensorDataset(self.data, self.labels)
        self.subset_indices = [
            [0, 1],  # Class 0
            [2, 3, 4],  # Class 1
            [5, 6, 7, 8, 9],  # Class 2
        ]
        self.subsets = [
            Subset(self.dataset, indices) for indices in self.subset_indices
        ]

    def test_dataloader(self):
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices, batch_size=6, drop_last=True
        )
        dataloader = DataLoader(self.dataset, batch_sampler=sampler)
        batch_data, batch_labels = next(iter(dataloader))
        self.assertEqual(batch_data.shape, (6, 2))
        self.assertEqual(len(batch_labels), 6)
        # Check balance: 2 samples from each class
        self.assertEqual((batch_labels == 0).sum().item(), 2)
        self.assertEqual((batch_labels == 1).sum().item(), 2)
        self.assertEqual((batch_labels == 2).sum().item(), 2)

    def test_dataloader_len_extended(self):
        # a DataLoader reports and trusts the sampler's __len__
        sampler = BalancedBatchSampler(
            group_indices=self.subset_indices,
            batch_size=6,
            drop_last=True,
            extend_groups=True,
        )
        dataloader = DataLoader(self.dataset, batch_sampler=sampler)
        self.assertEqual(len(dataloader), 2)
        self.assertEqual(len(list(dataloader)), len(dataloader))
        for _, batch_labels in dataloader:
            for label in range(3):
                self.assertEqual((batch_labels == label).sum().item(), 2)


if __name__ == "__main__":
    unittest.main()
