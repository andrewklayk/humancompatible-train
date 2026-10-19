"""
Tokenize a FineWeb-Edu slice into the flat shard E3 trains on.

Run **once, on a login node** — compute nodes have no network access, and the training
path imports no ``datasets`` and reaches no network. That split is deliberate: a
job that dies at hour three because an HTTP fetch failed is an expensive way to learn that
streaming is fragile. (``transformers`` *is* needed at training time, for any ``--model``
that is not the stub; it is ``datasets`` and the network that stay on the login node.)

The shard is a flat ``.npy`` array of token ids, in the narrowest integer width that fits
the tokenizer's vocabulary; the ``.npy`` header records it.

Usage::

    python paper/e3/prepare_data.py --model Qwen/Qwen2.5-0.5B \\
        --tokens 40_000_000 --out paper/results/e3/tokens.npy

Neither ``transformers`` nor ``datasets`` is installed in this project's environment or
available as a cluster module, so both are imported lazily and only when actually needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np



def _imports():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "prepare_data.py needs `transformers` and `datasets`. Neither is in this "
            "project's conda environment nor provided by an EasyBuild module; install "
            "them into a venv layered on the PyTorch module (see paper/sbatch_tune.sh). The "
            "training path needs neither."
        ) from exc
    return load_dataset, AutoTokenizer


def build(args) -> int:
    load_dataset, AutoTokenizer = _imports()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise SystemExit(f"{args.model} has no eos_token_id; cannot delimit documents")

    stream = load_dataset(args.dataset, name=args.name, split=args.split,
                          streaming=True)

    chunks, total = [], 0
    for record in stream:
        ids = tokenizer(record[args.column])["input_ids"]
        # One EOS between documents, so a block never blends two unrelated texts without
        # a boundary the model can see.
        ids.append(eos)
        chunks.append(np.asarray(ids, dtype=np.int64))
        total += len(ids)
        if total >= args.tokens:
            break
        if len(chunks) % 2000 == 0:
            print(f"  {total:,} tokens ...", flush=True)

    if total < args.tokens:
        print(f"warning: the stream ended at {total:,} of the requested "
              f"{args.tokens:,} tokens")

    flat = np.concatenate(chunks)[: args.tokens]
    # Narrowest safe width: Qwen2.5's 151936-id vocabulary would wrap in uint16.
    flat = flat.astype(np.uint16 if len(tokenizer) <= 1 << 16 else np.uint32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:  # a file handle, so np.save keeps the name as given
        np.save(f, flat)
    written = len(flat)
    print(f"wrote {written:,} tokens -> {args.out}")
    print(f"  {written // args.seq_len:,} blocks at seq_len={args.seq_len}")
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                        help="whose tokenizer to use; must match the training model")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--column", default="text")
    parser.add_argument("--tokens", type=int, default=40_000_000)
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="only used to report the resulting block count")
    parser.add_argument("--out", default="paper/results/e3/tokens.npy")
    build(parser.parse_args(argv))


if __name__ == "__main__":
    main()
