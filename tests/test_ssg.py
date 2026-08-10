"""
Tests for the stochastic switching subgradient method.

The behavioural reference is the ``switching`` updater in
``benchmark/new_bench/algorithms.py``: descend ``max_i c_i`` when the largest
violation exceeds the tolerance, and the objective otherwise.
"""

import unittest

import torch

from humancompatible.train.dual_optim import SSG
from humancompatible.train.dual_optim.base import DualOptimizer

LOSS = torch.tensor(5.0)


class TestSSGSwitching(unittest.TestCase):
    def test_is_a_dual_optimizer(self):
        opt = SSG(m=3)
        self.assertIsInstance(opt, DualOptimizer)
        self.assertIsInstance(opt, torch.optim.Optimizer)

    def test_feasible_returns_the_objective(self):
        opt = SSG(m=3)
        surrogate = opt.forward_update(LOSS, torch.tensor([-1.0, -0.5, -0.2]))
        self.assertTrue(torch.equal(surrogate, LOSS))
        self.assertFalse(opt.last_switched_to_constraints)

    def test_violated_returns_the_max_violation(self):
        opt = SSG(m=3)
        c = torch.tensor([-1.0, 0.25, -0.2])
        surrogate = opt.forward_update(LOSS, c)
        self.assertTrue(torch.equal(surrogate, torch.tensor(0.25)))
        self.assertTrue(opt.last_switched_to_constraints)

    def test_tolerance_shifts_the_switch(self):
        c = torch.tensor([0.05])
        self.assertTrue(torch.equal(SSG(m=1).forward_update(LOSS, c), torch.tensor(0.05)))
        # with a tolerance above the violation, the objective branch is taken
        self.assertTrue(
            torch.equal(SSG(m=1, constraint_tol=0.1).forward_update(LOSS, c), LOSS)
        )

    def test_gradient_routes_to_the_argmax_constraint_only(self):
        # The constraint branch is a max, so only the largest violation gets a
        # subgradient -- this is the defining property of the method.
        opt = SSG(m=3)
        c = torch.tensor([-1.0, 0.25, 0.1], requires_grad=True)
        opt.forward_update(LOSS, c).backward()
        self.assertTrue(torch.equal(c.grad, torch.tensor([0.0, 1.0, 0.0])))

    def test_objective_branch_leaves_constraints_gradient_free(self):
        opt = SSG(m=2)
        loss = (torch.ones(2, requires_grad=True) * 3).sum()
        c = torch.tensor([-1.0, -2.0], requires_grad=True)
        opt.forward_update(loss, c).backward()
        self.assertTrue(c.grad is None or torch.equal(c.grad, torch.zeros(2)))

    def test_constraint_scale_scales_only_the_constraint_branch(self):
        c_bad, c_ok = torch.tensor([0.25]), torch.tensor([-0.25])
        opt = SSG(m=1, constraint_scale=3.0)
        self.assertTrue(torch.equal(opt.forward_update(LOSS, c_bad), torch.tensor(0.75)))
        self.assertTrue(torch.equal(opt.forward_update(LOSS, c_ok), LOSS))

    def test_matches_the_benchmark_switching_updater(self):
        # Reproduce benchmark/new_bench/algorithms.py's `switching` branch exactly.
        def reference(loss, c, tol):
            max_c = max(c)
            return max_c if max_c > tol else loss

        opt = SSG(m=4, constraint_tol=0.01)
        for values in ([-1.0, -0.5, -0.2, -0.1], [-1.0, 0.5, -0.2, 0.4], [0.0, 0.0, 0.0, 0.005]):
            c = torch.tensor(values)
            self.assertTrue(
                torch.equal(opt.forward_update(LOSS, c), reference(LOSS, c, 0.01)),
                values,
            )


class TestSSGDuals(unittest.TestCase):
    def test_duals_exist_but_never_move(self):
        # Kept at zero so diagnostics that read multipliers get the right answer
        # for a method that maintains none.
        opt = SSG(m=3)
        for _ in range(5):
            opt.forward_update(LOSS, torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.equal(opt.duals, torch.zeros(3)))

    def test_update_alone_does_not_raise(self):
        SSG(m=2).update(torch.tensor([1.0, 2.0]))


class TestSSGGroups(unittest.TestCase):
    def test_switch_is_taken_across_all_groups(self):
        opt = SSG(m=2)
        opt.add_constraint_group(m=1, name="second")
        self.assertEqual(opt.m, 3)
        self.assertEqual(opt.names, ["group0", "second"])
        # the violation lives in the second group
        surrogate = opt.forward_update(LOSS, torch.tensor([-1.0, -1.0, 0.5]))
        self.assertTrue(torch.equal(surrogate, torch.tensor(0.5)))

    def test_accepts_mapping_input(self):
        opt = SSG(m=2)
        opt.add_constraint_group(m=1, name="second")
        surrogate = opt.forward_update(
            LOSS, {"group0": torch.tensor([-1.0, -1.0]), "second": torch.tensor([0.5])}
        )
        self.assertTrue(torch.equal(surrogate, torch.tensor(0.5)))

    def test_switched_to_constraints_query(self):
        opt = SSG(m=2, constraint_tol=0.1)
        self.assertTrue(opt.switched_to_constraints(torch.tensor([-1.0, 0.2])))
        self.assertFalse(opt.switched_to_constraints(torch.tensor([-1.0, 0.05])))

    def test_violation_uses_declared_bounds(self):
        opt = SSG(m=1)
        opt.add_constraint_group(m=1, bound=1.0)
        self.assertAlmostEqual(
            opt.violation(torch.tensor([-0.5, 0.9])).item(), -0.1, places=6
        )


class TestSSGDistributed(unittest.TestCase):
    def test_switch_uses_reduced_constraints(self):
        from unittest.mock import patch

        # Locally feasible, globally violated: all replicas must switch together.
        opt = SSG(m=1, process_group=object())
        local = torch.tensor([-0.5])
        with patch(
            "torch.distributed.all_reduce",
            side_effect=lambda t, **kw: t.copy_(torch.tensor([0.5])),
        ):
            surrogate = opt.forward_update(LOSS, local)
        self.assertTrue(opt.last_switched_to_constraints)
        self.assertTrue(torch.equal(surrogate, torch.tensor(-0.5)))


if __name__ == "__main__":
    unittest.main()
