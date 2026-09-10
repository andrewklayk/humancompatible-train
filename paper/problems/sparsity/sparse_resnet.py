"""
Structured L0 gates on a CIFAR ResNet — the vision half of arXiv:2208.04425.

Same constraint as :mod:`paper.problems.sparse_lm`, which owns the gate, the group
bookkeeping, the density arithmetic and the forward hook; this module only says *what* to
gate. One gate per *intermediate* channel of a residual block: gating the input of a
block's second (or third) convolution zeroes that channel's contribution exactly, so an open probability of
``p`` means a purged block keeps ``p`` of the filters that produce those channels and the
matching input slices of the convolution that consumes them. The block's output channels
are left ungated — they are added to the identity path, so pruning one is a change to the
residual stream rather than to a single block.

``params_per_gate`` is therefore ``(producing filter) + (consuming input slice)``, read
off the two weight tensors. BatchNorm's two per-channel parameters are ignored, matching
the "weight matrices only" rule in ``sparse_lm``.
"""

from __future__ import annotations

import torch
from torch import nn

from .sparsity_gates import GateParamGroup, GateSet, HardConcreteGate

STAGES = ("layer1", "layer2", "layer3", "layer4")


def cifar_resnet(depth: int = 18, num_classes: int = 10) -> nn.Module:
    """A torchvision ResNet with the usual CIFAR stem: 3x3, stride 1, no max-pool.

    The ImageNet stem downsamples 32x32 inputs to 8x8 before the first block, which costs
    several points of accuracy and is not what the sparsity literature reports on.
    """
    from torchvision.models import resnet

    model = getattr(resnet, f"resnet{depth}")(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def residual_blocks(model: nn.Module):
    """``(index, block)`` over the residual blocks in forward order."""
    blocks = [block for stage in STAGES for block in getattr(model, stage)]
    if not blocks:
        raise TypeError(f"{type(model).__name__} has no layer1..layer4; expected a ResNet")
    return list(enumerate(blocks))


def _gated_convs(block: nn.Module):
    """``(position, conv, producer)`` for every convolution that consumes gated channels.

    Every convolution in a block but the first: its input channels are exactly the output
    channels of the one before it, so one gate prunes a producing filter and the input
    slice that reads it. BasicBlock yields one, Bottleneck two. The downsample branch is
    an ``nn.Sequential`` and so is never reached by ``children()``, which is what keeps
    the residual path ungated.
    """
    convs = [m for m in block.children() if isinstance(m, nn.Conv2d)]
    return [(i, conv, convs[i - 1]) for i, conv in enumerate(convs[1:], start=1)]


def attach_gates(
    model: nn.Module,
    *,
    init_open: float = 0.95,
    init_std: float = 0.01,
    seed: int = 0,
    holder: str = "l0_gates",
    device=None,
    dtype=torch.float32,
) -> GateSet:
    """Install one hard-concrete gate vector per intermediate channel set."""
    generator = torch.Generator().manual_seed(seed)
    groups = [
        GateParamGroup(
            name=f"block{index:02d}.conv{position}",
            layer=index,
            kind=f"conv{position}",
            module=conv,
            dim=1,  # NCHW: the gates index the channel axis
            gate=HardConcreteGate(
                producer.out_channels,
                init_open=init_open,
                init_std=init_std,
                generator=generator,
                device=device,
                dtype=dtype,
            ),
            # the producing filter, plus the slice of the consumer it feeds
            params_per_gate=producer.weight[0].numel() + conv.weight[:, 0].numel(),
        )
        for index, block in residual_blocks(model)
        for position, conv, producer in _gated_convs(block)
    ]
    if not groups:
        raise ValueError("no gates were attached; the blocks hold fewer than two convolutions")
    return GateSet(model, groups, holder)
