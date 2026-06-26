"""
Point set storage and proximity-based update, a port of
``NonOptPointSetUpdateProximity`` from https://github.com/frankecurtis/NonOpt.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor


class Point(NamedTuple):
    """A previously visited point: iterate, objective value and (sub)gradient."""

    x: Tensor
    f: float
    g: Tensor


class PointSet:
    """
    Set of previously visited points whose gradients may enter the cutting-plane
    bundle.  Pruned by age and by proximity to the current iterate.

    :param size_factor: If the size of the point set exceeds this factor times
        the number of variables, the oldest members are removed.
    :type size_factor: float
    :param size_maximum: Hard cap on the size of the point set.  The C++ default
        is infinity; here it defaults to 100 to bound memory, since each point
        stores a full gradient copy.
    :type size_maximum: int
    :param envelope_factor: A point is removed when its distance to the current
        iterate exceeds this factor times the stationarity radius.
    :type envelope_factor: float
    """

    def __init__(
        self,
        size_factor: float = 5e-02,
        size_maximum: int = 100,
        envelope_factor: float = 1e+02,
    ) -> None:
        self.size_factor = size_factor
        self.size_maximum = size_maximum if size_maximum is not None else math.inf
        self.envelope_factor = envelope_factor
        self.points: list[Point] = []

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def add(self, point: Point) -> None:
        """Appends a point (its tensors are stored as detached clones)."""
        self.points.append(
            Point(point.x.detach().clone(), float(point.f), point.g.detach().clone())
        )

    def update(self, x_current: Tensor, stationarity_radius: float) -> None:
        """
        Prunes the point set: removes the oldest members while the set is too
        large, then removes all points farther than
        ``envelope_factor * stationarity_radius`` from the current iterate.

        :param x_current: Current iterate (flat).
        :type x_current: torch.Tensor
        :param stationarity_radius: Current stationarity radius.
        :type stationarity_radius: float
        """
        n = x_current.numel()
        limit = min(self.size_factor * n, self.size_maximum)
        if len(self.points) > limit:
            del self.points[: len(self.points) - max(int(limit), 0)]
        radius = self.envelope_factor * stationarity_radius
        self.points = [
            p for p in self.points if float(torch.norm(x_current - p.x)) <= radius
        ]
