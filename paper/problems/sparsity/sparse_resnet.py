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
from torch.nn import functional as F

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


class _WideBlock(nn.Module):
    """Pre-activation wide-residual block (Zagoruyko & Komodakis).

    ``downsample`` is wrapped in an ``nn.Sequential`` so that ``children()`` yields only
    the two 3x3 convolutions -- the same convention torchvision uses, and what keeps
    :func:`_gated_convs` off the residual path.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = x if self.downsample is None else self.downsample(out)
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))
        return out + shortcut


class _WideResNet(nn.Module):
    """WRN-depth-width. Stages are named ``layer1..layer3`` so the gate plumbing finds them."""

    def __init__(self, depth: int, width: int, num_classes: int) -> None:
        super().__init__()
        if (depth - 4) % 6:
            raise ValueError(f"WRN depth must be 6n+4; got {depth}")
        n = (depth - 4) // 6
        w = [16, 16 * width, 32 * width, 64 * width]
        self.conv1 = nn.Conv2d(3, w[0], 3, stride=1, padding=1, bias=False)
        for i, stride in enumerate((1, 2, 2), start=1):
            setattr(self, f"layer{i}", nn.Sequential(*[
                _WideBlock(w[i - 1] if j == 0 else w[i], w[i], stride if j == 0 else 1)
                for j in range(n)
            ]))
        self.bn = nn.BatchNorm2d(w[3])
        self.fc = nn.Linear(w[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(self.layer2(self.layer1(self.conv1(x))))
        x = F.relu(self.bn(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def wide_resnet(depth: int = 28, width: int = 10, num_classes: int = 10) -> nn.Module:
    """WRN-28-10 by default: the CIFAR architecture of arXiv:2208.04425 (their Table 4).

    ``(28 - 4) / 6 = 4`` blocks in each of three stages gives the 12 sparsifiable
    convolutions their Appendix J.4 counts.
    """
    return _WideResNet(depth, width, num_classes)


def residual_blocks(model: nn.Module):
    """``(index, block)`` over the residual blocks in forward order."""
    blocks = [block for stage in STAGES for block in getattr(model, stage, ())]
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


def gated_l2(model: nn.Module, gates: GateSet) -> torch.Tensor:
    """``sum_j stopgrad(pi_j) * ||theta_j||^2``: the weight decay of their Eq. (13).

    Plain L2 decays a weight that its gate has already closed. Theirs decays each gated
    weight in proportion to its gate's open probability, with ``pi`` stop-gradded so the
    penalty pushes only on the weights and gate deactivation stays driven by the
    constraint alone. Weights with no gate over them (stem, shortcuts, BatchNorm, the
    classifier) are decayed as usual.

    Multiply by ``weight_decay / 2`` to match what ``SGD(weight_decay=...)`` would add.
    """
    pairs = [(producer, conv) for _, block in residual_blocks(model)
             for _, conv, producer in _gated_convs(block)]
    if len(pairs) != len(gates.groups):
        raise ValueError(
            f"{len(pairs)} gated convolution(s) but {len(gates.groups)} gate group(s); "
            "gated_l2 assumes attach_gates' ordering"
        )

    total = None
    scaled = set()
    for group, (producer, conv) in zip(gates.groups, pairs):
        pi = group.gate.open_proba().detach()
        # the producing filter (gates index its output axis) and the slice of the
        # consumer that reads it (gates index its input axis) -- the same two tensors
        # params_per_gate counts
        terms = (
            pi @ producer.weight.flatten(1).pow(2).sum(1),
            pi @ conv.weight.transpose(0, 1).flatten(1).pow(2).sum(1),
        )
        for term in terms:
            total = term if total is None else total + term
        scaled.update((id(producer.weight), id(conv.weight)))

    gate_ids = {id(p) for p in gates.gate_parameters()}
    for param in model.parameters():
        if id(param) not in scaled and id(param) not in gate_ids:
            total = param.pow(2).sum() if total is None else total + param.pow(2).sum()
    return total


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
            block=index,
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
