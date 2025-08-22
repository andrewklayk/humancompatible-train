from typing import Callable, Iterable

import numpy as np
import fairret
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


def _make_dataloaders(dataset, group_indices, batch_size, device, drop_last, gen=None):
    dataloaders = []
    for idx in group_indices:
        sampler = SubsetRandomSampler(idx, gen)
        dataloaders.append(iter(DataLoader(dataset, batch_size, sampler=sampler, drop_last=True)))
    return dataloaders


class FairnessConstraint:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        group_indices: Iterable[Iterable[int]],
        fn: Callable,
        batch_size: int = None,
        use_dataloaders=True,
        device="cpu",
        seed=None,
        loader_drop_last=False,
    ):
        self.dataset = dataset
        self.group_sets = [
            torch.utils.data.Subset(dataset, idx) for idx in group_indices
        ]
        self._group_indices = group_indices
        self.fn = fn
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._torch_rng = torch.manual_seed(seed) if seed is not None else torch.Generator(device=device)
        self._device = device
        self._drop_last = loader_drop_last
        if batch_size is not None:
            self._batch_size = batch_size
            if use_dataloaders:
                self.group_dataloaders = _make_dataloaders(
                    dataset, group_indices, batch_size, device, gen=self._torch_rng, drop_last=loader_drop_last
                )

    def group_sizes(self):
        return [len(group) for group in self.group_sets]

    def eval(self, net, sample, **kwargs):
        return self.fn(net, sample, **kwargs)
        
    # def eval_fairret(self, net, sample, group_id, **kwargs):
    #     if self._fairret_results is None:
    #         statistic = fairret.statistic.TruePositiveRate()
    #         loss = fairret.loss.NormLoss(statistic)
            
        
    #     return self._fairret_results[group_id]

    def sample_loader(self):
        self._fairret_results = None
        samples = []
        for i, l in enumerate(self.group_dataloaders):
            try:
                sample = next(l)
            except StopIteration:
                sampler = SubsetRandomSampler(self._group_indices[i], self._torch_rng)
                l = iter(DataLoader(self.dataset, self._batch_size, sampler=sampler, drop_last=self._drop_last))
                sample = next(l)
                self.group_dataloaders[i] = l
                
            samples.append(sample)
        return samples

    def sample_dataset(
        self, N, rng: np.random.Generator = None, indices=None, return_indices=False
    ):
        if rng is None:
            rng = self._rng

        if indices is None:
            indices = []
            # returns len(group) points if N > len(group)
            for group in self.group_sets:
                indices.append(
                    rng.choice(group.indices, N)
                    if N < len(group)
                    else rng.choice(group.indices, len(group))
                )

        sample = [self.dataset[indices[i]] for i, _ in enumerate(self.group_sets)]

        if return_indices:
            return sample, indices
        else:
            return sample
