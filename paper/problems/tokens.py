"""
Pre-tokenized token shards, and a rank-sharded block order over them.

E3's training data is a flat ``uint16`` file of token ids, prepared once by
``paper/e3/prepare_data.py`` and read here by ``numpy.memmap``. Two reasons for that
rather than streaming a dataset at train time:

* **Compute nodes need neither network access nor the ``datasets`` package.** A streaming
  job that dies at hour three because an HTTP fetch failed is a bad way to learn this.
* **The token stream is byte-identical across runs.** Determinism was checked rather than
  assumed for E0 (its CSV/JSON/Markdown artifacts reproduce byte for byte on a re-run),
  and keeping that property matters more, not less, once several methods are compared
  across seeds.

**The token width is chosen, recorded, and read back.** ``uint16`` holds any vocabulary up
to 65536, which covers SmolLM's 49152 — but Qwen2.5's is 151936, so a uint16 shard would
silently wrap and surface as an unexplained loss plateau. :func:`write_shard` picks the
narrowest safe width, writes it to a ``.meta.json`` sidecar, and :class:`TokenShard` reads
it from there rather than assuming.

The sampler shards *blocks* across ranks and truncates so every rank yields exactly the
same number of batches. That is not a nicety: a rank that runs out of batches early stops
calling collectives while its peers are still waiting, and the job hangs instead of
failing. It is also the specific defect ``BalancedBatchSampler`` has (its ``__len__``
disagrees with its ``__iter__`` when ``extend_groups`` is set), so the shape of the bug is
already known in this repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import torch

# Narrowest first: uint16 halves page-cache pressure where the vocabulary allows it.
WIDTHS = (np.uint16, np.uint32)


def _meta_path(path) -> Path:
    return Path(str(path) + ".meta.json")


def _pick_dtype(max_id: int, vocab_size: Optional[int]):
    ceiling = max(max_id, (vocab_size - 1) if vocab_size else 0)
    for dtype in WIDTHS:
        if ceiling <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"token id ceiling {ceiling} does not fit in {WIDTHS[-1].__name__}")


def write_shard(path, tokens: Iterable[int], *, vocab_size: Optional[int] = None) -> int:
    """Write a flat token shard plus its metadata sidecar. Returns the token count.

    :param tokens: Any iterable of ids, or a numpy array.
    :param vocab_size: The tokenizer's vocabulary size. Used to pick the token width, so
        a shard written for a small vocabulary is never later reread as a large one.
    """
    array = (
        np.asarray(tokens, dtype=np.int64)
        if isinstance(tokens, np.ndarray)
        else np.fromiter(tokens, dtype=np.int64)
    )
    if array.size == 0:
        raise ValueError("refusing to write an empty shard")
    if array.min() < 0:
        raise ValueError(f"token ids must be non-negative; saw {array.min()}")
    dtype = _pick_dtype(int(array.max()), vocab_size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(dtype).tofile(path)
    _meta_path(path).write_text(
        json.dumps(
            {
                "dtype": np.dtype(dtype).name,
                "n_tokens": int(array.size),
                "vocab_size": vocab_size,
                "max_id": int(array.max()),
            },
            indent=2,
        )
    )
    return int(array.size)


def write_synthetic_shard(path, n_tokens: int, vocab_size: int, seed: int = 0) -> int:
    """A shard of uniform random ids.

    Not a language model's data, and not pretending to be: it exists so the loop, the
    gates, the duals and the distributed reduction can be smoke-tested with no download
    and no tokenizer. Loss will sit near ``log(vocab_size)`` and stay there, which is the
    correct behaviour on noise and a useful sanity signal in itself.
    """
    rng = np.random.default_rng(seed)
    return write_shard(
        path,
        rng.integers(0, vocab_size, size=n_tokens, dtype=np.int64),
        vocab_size=vocab_size,
    )


class TokenShard:
    """A memory-mapped token shard served as fixed-length, non-overlapping blocks.

    Blocks are ``seq_len`` tokens long. The causal shift lives in the model (``labels``
    equal ``input_ids`` and the loss compares ``logits[:, :-1]`` with ``labels[:, 1:]``),
    so a block of ``seq_len`` tokens yields ``seq_len - 1`` predictions and no block needs
    to overlap its neighbour.
    """

    def __init__(self, path, seq_len: int, *, dtype=None):
        self.path = Path(path)
        if seq_len < 2:
            raise ValueError(f"seq_len must be at least 2; got {seq_len}")
        self.seq_len = int(seq_len)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} does not exist; run paper/e3/prepare_data.py first"
            )
        meta_file = _meta_path(self.path)
        self.meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        if dtype is None:
            if "dtype" not in self.meta:
                raise ValueError(
                    f"{meta_file} is missing, so the token width is unknown. Rewrite the "
                    f"shard with write_shard(), or pass dtype= explicitly."
                )
            dtype = np.dtype(self.meta["dtype"])
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        self.n_blocks = self.data.size // self.seq_len
        if self.n_blocks == 0:
            raise ValueError(
                f"{self.path} holds {self.data.size} tokens, fewer than one block of "
                f"{self.seq_len}"
            )

    def __len__(self) -> int:
        return self.n_blocks

    @property
    def n_tokens(self) -> int:
        return int(self.data.size)

    def block(self, index: int) -> np.ndarray:
        start = index * self.seq_len
        return np.asarray(self.data[start : start + self.seq_len])

    def batch(self, indices, device=None) -> torch.Tensor:
        """``(len(indices), seq_len)`` int64 tensor of token ids."""
        rows = np.stack([self.block(int(i)) for i in indices]).astype(np.int64)
        return torch.from_numpy(rows).to(device)


class BlockSampler:
    """Rank-sharded, epoch-seeded block order with an equal batch count on every rank.

    Rank ``r`` takes ``perm[r::world]`` of a shared permutation, so the shards are
    disjoint and cover the data; the per-rank block count is then truncated to the
    minimum so all ranks agree on the number of steps.
    """

    def __init__(
        self,
        n_blocks: int,
        batch_size: int,
        *,
        rank: int = 0,
        world: int = 1,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if world < 1 or not 0 <= rank < world:
            raise ValueError(f"invalid rank/world: {rank}/{world}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        if not drop_last:
            # A ragged final batch changes the per-rank sample count, which breaks the
            # equal-weighting that ReduceOp.AVG assumes. Not supported rather than
            # supported wrongly.
            raise NotImplementedError("drop_last=False is not supported; see the docstring")
        self.n_blocks = int(n_blocks)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world = int(world)
        self.seed = int(seed)
        per_rank = self.n_blocks // self.world
        self.batches_per_epoch = per_rank // self.batch_size
        if self.batches_per_epoch == 0:
            raise ValueError(
                f"{self.n_blocks} blocks over {self.world} ranks at batch {self.batch_size} "
                f"gives no complete batch; use a larger shard or a smaller batch"
            )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def epoch(self, epoch: int) -> Iterator[np.ndarray]:
        """Yield ``batches_per_epoch`` arrays of block indices for this rank."""
        # Every rank derives the same permutation from (seed, epoch), then takes its own
        # stride. No collective is needed to agree on the order.
        rng = np.random.default_rng((self.seed, epoch))
        order = rng.permutation(self.n_blocks)[self.rank :: self.world]
        for b in range(self.batches_per_epoch):
            yield order[b * self.batch_size : (b + 1) * self.batch_size]

    def stream(self, steps: int, start_epoch: int = 0) -> Iterator[np.ndarray]:
        """Yield exactly ``steps`` batches, rolling over epochs as needed."""
        produced, epoch = 0, start_epoch
        while produced < steps:
            for indices in self.epoch(epoch):
                yield indices
                produced += 1
                if produced == steps:
                    return
            epoch += 1
