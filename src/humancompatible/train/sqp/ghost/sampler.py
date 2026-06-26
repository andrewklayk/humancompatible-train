"""
Minibatch samplers for the Ghost-penalty SQP method.

The MLMC step estimator (Eq. 23) is unbiased **only** if every call to
``draw`` returns *fresh, independent* samples, and finite-variance **only** if
the two coarse solves use the even / odd index halves of the fine batch
returned by ``split_even_odd`` (antithetic coupling). Both contracts are
documented here and respected by :func:`mlmc_direction`.
"""

import numpy as np
import torch


class Sampler:
    """Abstract minibatch source.

    Subclasses must return independent draws so that the MLMC estimator stays
    unbiased; reusing a fixed minibatch (e.g. cycling indices) biases it.
    """

    def draw(self, J, rng):
        """Return a fresh batch of ``J`` i.i.d. samples."""
        raise NotImplementedError

    def split_even_odd(self, batch):
        """Partition ``batch`` into its even- and odd-indexed halves.

        Returns ``(even_batch, odd_batch)``. For a fine batch of size 2^{N+1}
        each half has size 2^N (the antithetic coarse samples of Eq. 23).
        """
        raise NotImplementedError


class TensorSampler(Sampler):
    """i.i.d. sampling with replacement from in-memory tensors ``(X, y)``.

    A batch is the tuple ``(X[idx], y[idx])``; the same batch is used for the
    objective and every constraint (the common case where ξ = ζ). For problems
    with distinct per-constraint data, write a custom :class:`Sampler`.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def draw(self, J, rng):
        idx = rng.integers(0, self.X.shape[0], size=J)
        idx_t = torch.as_tensor(idx, device=self.X.device, dtype=torch.long)
        return self.X.index_select(0, idx_t), self.y.index_select(0, idx_t)

    def split_even_odd(self, batch):
        X, y = batch
        return (X[0::2], y[0::2]), (X[1::2], y[1::2])
