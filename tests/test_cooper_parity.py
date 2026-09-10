"""
Guards the E0e claim that ``ALM(penalty=0)`` is Cooper's projected GDA.

E0e establishes the correspondence at length; this keeps it from rotting
silently. The pairing is the part worth pinning: our ``forward_update`` builds
the surrogate from *post-update* multipliers, so it matches Cooper's
``AlternatingDualPrimalOptimizer``, while the split
``forward``/``backward``/``update`` idiom matches ``SimultaneousOptimizer``. A
change to that ordering on either side -- ours or theirs, on a version bump --
would silently turn E0e's comparison into a comparison of two different
algorithms, and the mismatch test below is what would catch it.

Deliberately small and fast: 20 steps on one 2-variable problem, no reference
solutions, no artifacts. E0e owns the real validation.
"""

import math
import unittest

import torch

try:
    import cooper

    HAVE_COOPER = True
except ImportError:  # pragma: no cover - optional [compare] extra
    HAVE_COOPER = False

from humancompatible.train.dual_optim import ALM

STEPS = 20
PRIMAL_LR = 0.05
DUAL_LR = 0.1
M = 2


def _objective(x):
    return 0.5 * (x**2).sum()


def _constraints(x):
    # min ||x||^2/2 s.t. 1 - x_0 <= 0, x_1 - 0.5 <= 0: one constraint active at
    # the solution and one inactive, so the projection is exercised.
    return torch.stack([1.0 - x[0], x[1] - 0.5])


def _make_ours():
    x = torch.nn.Parameter(torch.zeros(M, dtype=torch.float64))
    dual = ALM(
        m=M, lr=DUAL_LR, penalty=0.0, init_duals=0.0, is_ineq=True,
        # ALM's default safeguard box has no Cooper counterpart; its projection
        # is relu. Opening the box is what makes these the same algorithm.
        dual_range=(0.0, math.inf), momentum=0.0, dampening=0.0, restart=False,
    )
    return x, dual, torch.optim.SGD([x], lr=PRIMAL_LR)


def _make_cooper(cooper_class):
    x = torch.nn.Parameter(torch.zeros(M, dtype=torch.float64))

    class Wrapped(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__()
            self.constraint = cooper.Constraint(
                constraint_type=cooper.ConstraintType.INEQUALITY,
                formulation_type=cooper.formulations.Lagrangian,
                # dtype must be passed twice over: DenseMultiplier defaults to
                # float32 and initialize_weight applies that default to `init`,
                # so a float64 init alone is silently downcast.
                multiplier=cooper.multipliers.DenseMultiplier(
                    init=torch.zeros(M, dtype=torch.float64),
                    dtype=torch.float64,
                ),
            )

        def compute_cmp_state(self, params):
            return cooper.CMPState(
                loss=_objective(params),
                observed_constraints={
                    self.constraint: cooper.ConstraintState(
                        violation=_constraints(params))
                },
            )

    cmp = Wrapped()
    optimizer = cooper_class(
        cmp=cmp,
        primal_optimizers=torch.optim.SGD([x], lr=PRIMAL_LR),
        dual_optimizers=torch.optim.SGD(cmp.dual_parameters(), lr=DUAL_LR,
                                        maximize=True),
    )
    return x, cmp, optimizer


def _run(fused, cooper_class):
    x_ours, dual, primal = _make_ours()
    x_theirs, cmp, optimizer = _make_cooper(cooper_class)
    for _ in range(STEPS):
        loss, c = _objective(x_ours), _constraints(x_ours)
        primal.zero_grad()
        if fused:
            dual.forward_update(loss, c).backward()
            primal.step()
        else:
            dual.forward(loss, c).backward()
            primal.step()
            dual.update(c)
        optimizer.roll(compute_cmp_state_kwargs={"params": x_theirs})
    return (x_ours.detach(), dual.duals.detach(),
            x_theirs.detach(), cmp.constraint.multiplier.weight.detach())


@unittest.skipUnless(HAVE_COOPER, "requires the [compare] extra (cooper-optim)")
class TestCooperParity(unittest.TestCase):
    # ALM takes its dual dtype from the global default, and E0e runs in double
    # precision, so the test matches it -- restoring the default afterwards so
    # this does not leak into the rest of the suite.
    def setUp(self):
        self._default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self._default_dtype)

    def test_forward_update_matches_alternating_dual_primal(self):
        x_a, y_a, x_b, y_b = _run(True,
                                  cooper.optim.AlternatingDualPrimalOptimizer)
        # Bitwise: both execute add_(c, alpha=lr) followed by a projection at 0.
        self.assertTrue(torch.equal(y_a, y_b), f"{y_a} != {y_b}")
        self.assertTrue(torch.allclose(x_a, x_b, atol=1e-14, rtol=0.0),
                        f"{x_a} != {x_b}")

    def test_split_api_matches_simultaneous(self):
        x_a, y_a, x_b, y_b = _run(False, cooper.optim.SimultaneousOptimizer)
        self.assertTrue(torch.equal(y_a, y_b), f"{y_a} != {y_b}")
        self.assertTrue(torch.allclose(x_a, x_b, atol=1e-14, rtol=0.0),
                        f"{x_a} != {x_b}")

    def test_mismatched_ordering_is_distinguishable(self):
        """The control: without this, the two tests above prove nothing.

        Both pairings could agree merely because every configuration converges to
        the same point, so the wrong pairing has to be shown to differ.
        """
        _, y_a, _, y_b = _run(True, cooper.optim.SimultaneousOptimizer)
        self.assertFalse(torch.equal(y_a, y_b))
        self.assertGreater(float((y_a - y_b).abs().max()), 1e-6)

    def test_projection_agrees_at_exactly_zero(self):
        """Our ``clamp_`` and Cooper's ``relu`` must pin the same entries to 0.0."""
        _, y_a, _, y_b = _run(True,
                              cooper.optim.AlternatingDualPrimalOptimizer)
        # The second constraint is inactive, so its multiplier sits on the bound.
        self.assertEqual(float(y_a[1]), 0.0)
        self.assertTrue(torch.equal(y_a == 0.0, y_b == 0.0))


if __name__ == "__main__":
    unittest.main()
