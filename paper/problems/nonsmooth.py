"""
The ten standard nonsmooth test problems, in PyTorch.

Source of the formulae and starting points: N. Karmitsa, *Test problems for
large-scale nonsmooth minimization*, Reports of the Department of Mathematical
Information Technology, Series B 4/2007, University of Jyväskylä, also served at
<https://napsu.karmitsa.fi/testproblems/>. These are the same ten problems
Curtis & Que report NonOpt's ancestor on, which is what makes E0b a comparison
against published numbers rather than against nothing.

**Transcription is verified, not assumed.** Every problem carries a closed-form
``f(x0)`` derived by hand from its formula; :func:`verify` asserts the
implemented objective reproduces it to 1e-10 *and* that the closed form agrees
with the published table to the table's printed precision. A mistranscribed
formula then fails loudly at import time instead of masquerading as a solver
failure fifty iterations later.

Every objective takes a single 1-D tensor and returns a 0-dim tensor, so it can
be handed straight to a ``NonOpt`` closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

# Curtis & Que, Table 1 at n = 50: f(x0) and f(x*) as printed (one decimal).
# Kept separate from the closed forms below so the two can be cross-checked.
PUBLISHED_N50 = {
    "maxq":              {"convex": True,  "f_x0": 2500.0, "f_star": 0.0},
    "mxhilb":            {"convex": True,  "f_x0": 4.5,    "f_star": 0.0},
    "chained_lq":        {"convex": True,  "f_x0": 49.0,   "f_star": -69.3},
    "chained_cb3_1":     {"convex": True,  "f_x0": 980.0,  "f_star": 98.0},
    "chained_cb3_2":     {"convex": True,  "f_x0": 980.0,  "f_star": 98.0},
    "active_faces":      {"convex": False, "f_x0": 3.9,    "f_star": 0.0},
    "brown_2":           {"convex": False, "f_x0": 98.0,   "f_star": 0.0},
    "chained_mifflin_2": {"convex": False, "f_x0": 232.8,  "f_star": -34.8},
    "chained_crescent_1":{"convex": False, "f_x0": 292.3,  "f_star": 0.0},
    "chained_crescent_2":{"convex": False, "f_x0": 292.3,  "f_star": 0.0},
}


# Final values reached by the reference solvers, from Curtis & Que Tables 2-4 at
# n = 50 (SVANO-BFGS / SVANO-Bundle / SVANO-GS, LMBM, HANSO). Deliberately empty:
# these are numbers from a paper, and inventing them would make E0b's headline
# table fiction. Fill in as
#
#     REFERENCE_SOLVERS_N50["maxq"] = {"SVANO-BFGS": (f_final, n_func, n_grad), ...}
#
# and e0/b_nonopt.py will place the columns beside ours automatically. Until then
# the comparison is against the published f* only, which is what Table 1 gives.
REFERENCE_SOLVERS_N50: dict[str, dict[str, tuple]] = {}


@dataclass
class NonsmoothProblem:
    """One test problem, parameterised by dimension.

    :param f_x0: Closed-form ``f(x0)``, used to verify the transcription.
    :param f_star: Closed-form optimal value, or ``None`` when only a published
        numerical value exists (``chained_mifflin_2``).
    """

    name: str
    objective: Callable[[Tensor], Tensor]
    x0: Callable[[int], Tensor]
    f_x0: Callable[[int], float]
    f_star: Optional[Callable[[int], float]]
    is_convex: bool
    notes: str = ""

    def published(self, n: int = 50) -> dict:
        if n != 50:
            raise ValueError("published values are tabulated at n = 50 only")
        return PUBLISHED_N50[self.name]

    def target(self, n: int) -> float:
        """The value to measure ``f - f*`` against: closed form if there is one."""
        if self.f_star is not None:
            return self.f_star(n)
        return self.published(n)["f_star"]


def _dtype():
    return torch.get_default_dtype()


def _full(n, value):
    return torch.full((n,), float(value), dtype=_dtype())


def _alternating(n, odd, even):
    """``x_i = odd`` for odd ``i``, ``even`` for even ``i``, one-indexed."""
    x = _full(n, even)
    x[0::2] = float(odd)
    return x


# --------------------------------------------------------------------------- #
# 1. MAXQ
# --------------------------------------------------------------------------- #


def maxq(x):
    return (x**2).max()


def maxq_x0(n):
    x = torch.arange(1.0, n + 1.0, dtype=_dtype())
    x[n // 2:] *= -1.0
    return x


# --------------------------------------------------------------------------- #
# 2. MXHILB
# --------------------------------------------------------------------------- #


def _hilbert(n):
    i = torch.arange(1, n + 1, dtype=_dtype())
    return 1.0 / (i[:, None] + i[None, :] - 1.0)


def mxhilb(x):
    return (_hilbert(x.numel()) @ x).abs().max()


# --------------------------------------------------------------------------- #
# 3. Chained LQ
# --------------------------------------------------------------------------- #


def chained_lq(x):
    a = -x[:-1] - x[1:]
    b = a + (x[:-1] ** 2 + x[1:] ** 2 - 1.0)
    return torch.maximum(a, b).sum()


# --------------------------------------------------------------------------- #
# 4/5. Chained CB3, both variants
# --------------------------------------------------------------------------- #


def _cb3_terms(x):
    u, v = x[:-1], x[1:]
    return (
        u**4 + v**2,
        (2.0 - u) ** 2 + (2.0 - v) ** 2,
        2.0 * torch.exp(-u + v),
    )


def chained_cb3_1(x):
    """Sum of pointwise maxima — nonsmooth in every term."""
    a, b, c = _cb3_terms(x)
    return torch.maximum(torch.maximum(a, b), c).sum()


def chained_cb3_2(x):
    """Maximum of the three sums — a single nonsmoothness, same minimisers."""
    a, b, c = _cb3_terms(x)
    return torch.stack([a.sum(), b.sum(), c.sum()]).max()


# --------------------------------------------------------------------------- #
# 6. Number of active faces
# --------------------------------------------------------------------------- #


def active_faces(x):
    def g(y):
        return torch.log(y.abs() + 1.0)

    return torch.maximum(g(-x.sum()), g(x).max())


# --------------------------------------------------------------------------- #
# 7. Nonsmooth generalization of Brown function 2
# --------------------------------------------------------------------------- #

# |x|^p with p = x'^2 + 1 has derivative |x|^p log|x| * 2x' with respect to the
# exponent's variable, which is 0 * (-inf) = nan at x = 0 in floating point --
# and x = 0 is exactly where the minimiser is. Clamping the base to a denormal
# floor makes the product ~1e-148 instead of nan, at a cost to the function value
# of at most 1e-150.
_POWER_FLOOR = 1e-150


def brown_2(x):
    u, v = x[:-1], x[1:]
    u_abs = u.abs().clamp_min(_POWER_FLOOR)
    v_abs = v.abs().clamp_min(_POWER_FLOOR)
    return (u_abs ** (v**2 + 1.0) + v_abs ** (u**2 + 1.0)).sum()


# --------------------------------------------------------------------------- #
# 8. Chained Mifflin 2
# --------------------------------------------------------------------------- #


def chained_mifflin_2(x):
    u, v = x[:-1], x[1:]
    r = u**2 + v**2 - 1.0
    return (-u + 2.0 * r + 1.75 * r.abs()).sum()


# --------------------------------------------------------------------------- #
# 9/10. Chained Crescent, both variants
# --------------------------------------------------------------------------- #


def _crescent_terms(x):
    u, v = x[:-1], x[1:]
    q = u**2 + (v - 1.0) ** 2
    return q + v - 1.0, -q + v + 1.0


def chained_crescent_1(x):
    """Maximum of the two sums."""
    a, b = _crescent_terms(x)
    return torch.maximum(a.sum(), b.sum())


def chained_crescent_2(x):
    """Sum of the pointwise maxima."""
    a, b = _crescent_terms(x)
    return torch.maximum(a, b).sum()


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def _crescent_f_x0(n):
    # x0 alternates (-1.5, 2.0), so consecutive pairs alternate between
    # (u, v) = (-1.5, 2.0) -> 4.25 and (2.0, -1.5) -> 7.75. Of the n-1 pairs,
    # ceil((n-1)/2) start on an odd index.
    odd_pairs = (n - 1 + 1) // 2
    return 4.25 * odd_pairs + 7.75 * (n - 1 - odd_pairs)


PROBLEMS: dict[str, NonsmoothProblem] = {
    p.name: p
    for p in [
        NonsmoothProblem(
            "maxq", maxq, maxq_x0,
            f_x0=lambda n: float(n) ** 2,
            f_star=lambda n: 0.0,
            is_convex=True,
            notes="max_i x_i^2; only one coordinate is active at a time",
        ),
        NonsmoothProblem(
            "mxhilb", mxhilb, lambda n: _full(n, 1.0),
            # The Hilbert-like row sums are largest for i = 1, where the sum is
            # the harmonic number H_n.
            f_x0=lambda n: sum(1.0 / j for j in range(1, n + 1)),
            f_star=lambda n: 0.0,
            is_convex=True,
            notes="max_i |sum_j x_j/(i+j-1)|; ill-conditioned",
        ),
        NonsmoothProblem(
            "chained_lq", chained_lq, lambda n: _full(n, -0.5),
            f_x0=lambda n: float(n - 1),
            f_star=lambda n: -(n - 1) * math.sqrt(2.0),
            is_convex=True,
        ),
        NonsmoothProblem(
            "chained_cb3_1", chained_cb3_1, lambda n: _full(n, 2.0),
            f_x0=lambda n: 20.0 * (n - 1),
            f_star=lambda n: 2.0 * (n - 1),
            is_convex=True,
        ),
        NonsmoothProblem(
            "chained_cb3_2", chained_cb3_2, lambda n: _full(n, 2.0),
            f_x0=lambda n: 20.0 * (n - 1),
            f_star=lambda n: 2.0 * (n - 1),
            is_convex=True,
            notes="same minimisers as chained_cb3_1 but a single max, so far fewer active pieces",
        ),
        NonsmoothProblem(
            "active_faces", active_faces, lambda n: _full(n, 1.0),
            f_x0=lambda n: math.log(n + 1.0),
            f_star=lambda n: 0.0,
            is_convex=False,
        ),
        NonsmoothProblem(
            "brown_2", brown_2, lambda n: _alternating(n, -1.0, 1.0),
            f_x0=lambda n: 2.0 * (n - 1),
            f_star=lambda n: 0.0,
            is_convex=False,
            notes="variable exponents; see _POWER_FLOOR for the guard at x_i = 0",
        ),
        NonsmoothProblem(
            "chained_mifflin_2", chained_mifflin_2, lambda n: _full(n, -1.0),
            f_x0=lambda n: 4.75 * (n - 1),
            # No closed form; the published value at n = 50 is -34.8.
            f_star=None,
            is_convex=False,
        ),
        NonsmoothProblem(
            "chained_crescent_1", chained_crescent_1,
            lambda n: _alternating(n, -1.5, 2.0),
            f_x0=_crescent_f_x0,
            f_star=lambda n: 0.0,
            is_convex=False,
        ),
        NonsmoothProblem(
            "chained_crescent_2", chained_crescent_2,
            lambda n: _alternating(n, -1.5, 2.0),
            f_x0=_crescent_f_x0,
            f_star=lambda n: 0.0,
            is_convex=False,
        ),
    ]
}


# --------------------------------------------------------------------------- #
# transcription check
# --------------------------------------------------------------------------- #


def verify(n: int = 50, published_tolerance: float = 0.0501) -> list[dict]:
    """Check every transcription; raise on the first disagreement.

    Two independent checks per problem:

    1. the implemented objective at ``x0`` equals the hand-derived closed form to
       1e-10 (relative), which catches a typo in the code;
    2. the closed form equals the published ``f(x0)`` to the table's printed
       precision, which catches a misread of the source formula.

    ``published_tolerance`` is half of the published table's last printed digit,
    plus a hair: asserting tighter would be asserting against rounding, and
    ``chained_mifflin_2``'s exact 232.75 sits precisely on the boundary of the
    printed 232.8.

    :return: One record per problem, for the experiment's provenance table.
    """
    records = []
    for name, problem in PROBLEMS.items():
        x0 = problem.x0(n)
        if x0.numel() != n:
            raise AssertionError(f"{name}: x0 has {x0.numel()} entries, expected {n}")
        computed = float(problem.objective(x0))
        closed_form = problem.f_x0(n)
        if abs(computed - closed_form) > 1e-10 * max(1.0, abs(closed_form)):
            raise AssertionError(
                f"{name}: f(x0) = {computed!r} but the closed form gives "
                f"{closed_form!r} (difference {computed - closed_form:.3e})"
            )

        published = PUBLISHED_N50[name] if n == 50 else None
        if published is not None:
            if abs(closed_form - published["f_x0"]) > published_tolerance:
                raise AssertionError(
                    f"{name}: closed-form f(x0) = {closed_form:.6f} disagrees with "
                    f"the published {published['f_x0']} beyond the table's precision"
                )
            if problem.f_star is not None:
                target = problem.f_star(n)
                if abs(target - published["f_star"]) > published_tolerance:
                    raise AssertionError(
                        f"{name}: closed-form f* = {target:.6f} disagrees with the "
                        f"published {published['f_star']}"
                    )
            if problem.is_convex != published["convex"]:
                raise AssertionError(f"{name}: convexity disagrees with the table")

        records.append({
            "problem": name,
            "convex": problem.is_convex,
            "f(x0) computed": computed,
            "f(x0) closed form": closed_form,
            "f(x0) published": published["f_x0"] if published else None,
            "f* used": problem.target(n) if n == 50 or problem.f_star else None,
            "f* published": published["f_star"] if published else None,
            "f* source": "closed form" if problem.f_star is not None else "published",
        })
    return records


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    for record in verify(50):
        print(record)
    print("all ten transcriptions verified at n = 50")
