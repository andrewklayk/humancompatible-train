import unittest
from unittest.mock import patch
from torch.optim import Optimizer
from humancompatible.train.dual_optim import ALM
import torch
import torch.distributed as dist

# Unit tests
class TestALM(unittest.TestCase):
    def setUp(self):
        # Initialize ALM instances for reuse in tests
        self.alm_default = ALM(m=3, lr=0.1, penalty=1.0)
        self.alm_custom_range = ALM(m=3, lr=0.1, penalty=1.0, dual_range=(-1.0, 1.0))
        self.alm_momentum = ALM(m=3, lr=0.1, penalty=1.0, momentum=0.9, dampening=0.5)

        # Common test data
        self.loss = torch.tensor(5.0)
        self.constraints = torch.tensor([1.0, 2.0, 3.0])
        self.large_constraints = torch.tensor([10.0, 20.0, 30.0])
        self.momentum = 0.9
        self.dampening = 0.5

    def test_alm_initialization(self):
        # Test initialization with m
        self.assertEqual(len(self.alm_default.duals), 3)

        # Test initialization with init_duals
        init_duals = torch.tensor([1.0, 2.0, 3.0])
        alm = ALM(init_duals=init_duals, lr=0.01, penalty=1.0)
        self.assertTrue(torch.all(alm.duals == init_duals))

        # Test invalid initialization
        with self.assertRaises(ValueError):
            ALM(m=None, init_duals=None)

    def test_alm_forward(self):
        lagrangian = self.alm_default.forward(self.loss, self.constraints)
        expected_lagrangian = self.loss + torch.dot(self.alm_default.duals, self.constraints) + 0.5 * self.alm_default.penalty * torch.dot(self.constraints, self.constraints)
        self.assertTrue(torch.allclose(lagrangian, expected_lagrangian))

    def test_alm_update(self):
        expected_duals = self.alm_default.duals + 0.1 * self.constraints
        # breakpoint()
        self.alm_default.update(self.constraints)
        self.assertTrue(torch.allclose(self.alm_default.duals, expected_duals))

    def test_alm_momentum_update(self):

        def update_buffer(buffer):
            return self.momentum * buffer + (1 - self.dampening) * self.constraints
        
        # first: 
        momentum_buffer = update_buffer(torch.zeros_like(self.alm_momentum.duals))
        expected_duals = self.alm_momentum.duals + 0.1 * momentum_buffer

        self.alm_momentum.update(self.constraints)

        self.assertTrue(torch.allclose(self.alm_momentum.duals, expected_duals))
        # second:
        momentum_buffer = update_buffer(momentum_buffer)
        expected_duals = expected_duals + 0.1 * momentum_buffer

        self.alm_momentum.update(self.constraints)

        # Check if momentum is applied correctly
        self.assertTrue(torch.allclose(self.alm_momentum.duals, expected_duals))


    def test_alm_forward_update(self):
        lagrangian = self.alm_default.forward_update(self.loss, self.constraints)
        expected_lagrangian = self.loss + torch.dot(self.alm_default.duals, self.constraints) + 0.5 * self.alm_default.penalty * torch.dot(self.constraints, self.constraints)
        self.assertTrue(torch.allclose(lagrangian, expected_lagrangian))

    def test_alm_add_constraint_group(self):
        self.alm_default.add_constraint_group(m=2, lr=0.02)
        self.assertEqual(len(self.alm_default.duals), 5)
        self.assertEqual(self.alm_default.param_groups[1]["lr"], 0.02)

    def test_alm_dual_range_clamping(self):
        self.alm_custom_range.update(self.large_constraints)
        self.assertTrue(torch.all(self.alm_custom_range.duals <= 1.0) and torch.all(self.alm_custom_range.duals >= -1.0))


    def test_step_is_update_alias(self):
        alm = ALM(m=3, lr=0.1, penalty=1.0)
        duals_before = alm.duals.clone()
        alm.step(self.constraints)
        alm2 = ALM(m=3, lr=0.1, penalty=1.0)
        alm2.update(self.constraints)
        self.assertTrue(torch.allclose(alm.duals, alm2.duals))
        self.assertFalse(torch.allclose(alm.duals, duals_before))

    def test_alm_state_dict(self):
        alm = ALM(m=3, lr=0.1, penalty=2.0, dual_range=(-1.0, 1.0))
        state_dict = alm.state_dict()
        self.assertEqual(state_dict["state"]["penalty"], 2.0)

class TestALMFixes(unittest.TestCase):
    """Tests for fix 1 (momentum buffer in forward) and fix 2 (multi-group slicing)."""

    def setUp(self):
        self.loss = torch.tensor(5.0)
        self.constraints = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

    # --- Fix 1: forward() must not advance the momentum buffer ---

    def test_forward_does_not_corrupt_momentum_buffer(self):
        # Calling forward() then update() must give the same duals as update() alone.
        c = torch.tensor([1.0, 2.0, 3.0])
        alm_direct = ALM(m=3, lr=0.1, penalty=1.0, momentum=0.9)
        alm_via_forward = ALM(m=3, lr=0.1, penalty=1.0, momentum=0.9)

        alm_direct.update(c)

        alm_via_forward.forward(self.loss, c)
        alm_via_forward.update(c)

        self.assertTrue(torch.allclose(alm_direct.duals, alm_via_forward.duals))

    def test_forward_update_and_separate_forward_update_agree(self):
        # forward_update() and forward() + update() must produce identical duals.
        c = torch.tensor([1.0, 2.0, 3.0])
        alm_combined = ALM(m=3, lr=0.1, penalty=1.0, momentum=0.9)
        alm_separate = ALM(m=3, lr=0.1, penalty=1.0, momentum=0.9)

        alm_combined.forward_update(self.loss, c)
        alm_separate.forward(self.loss, c)
        alm_separate.update(c)

        self.assertTrue(torch.allclose(alm_combined.duals, alm_separate.duals))

    # --- Fix 2: multi-group constraint slicing ---

    def test_multi_group_update_slices_correctly(self):
        alm = ALM(m=2, lr=0.1, penalty=1.0)
        alm.add_constraint_group(m=3, lr=0.2)

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        alm.update(c)

        self.assertTrue(torch.allclose(alm.param_groups[0]["params"][0], 0.1 * c[:2]))
        self.assertTrue(torch.allclose(alm.param_groups[1]["params"][0], 0.2 * c[2:]))

    def test_multi_group_forward_lagrangian_correct(self):
        init0 = torch.tensor([1.0, 1.0])
        init1 = torch.tensor([1.0, 1.0, 1.0])
        alm = ALM(m=2, lr=0.1, penalty=1.0, init_duals=init0)
        alm.add_constraint_group(m=3, lr=0.2, init_duals=init1)

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        lagrangian = alm.forward(self.loss, c)

        expected = (self.loss
                    + init0 @ c[:2]
                    + init1 @ c[2:]
                    + 0.5 * alm.penalty * torch.dot(c, c))
        self.assertTrue(torch.allclose(lagrangian, expected))

    def test_multi_group_forward_update_slices_correctly(self):
        alm = ALM(m=2, lr=0.1, penalty=1.0)
        alm.add_constraint_group(m=3, lr=0.2)

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        alm.forward_update(self.loss, c)

        self.assertTrue(torch.allclose(alm.param_groups[0]["params"][0], 0.1 * c[:2]))
        self.assertTrue(torch.allclose(alm.param_groups[1]["params"][0], 0.2 * c[2:]))


class TestALMHPR(unittest.TestCase):
    """The ``augmentation="hpr"`` option.

    Constraints are ``c <= 0``; ``rho`` is the penalty and ``sigma = y + rho*c`` the
    trial multiplier. HPR differs from the default only on inequality groups, and there
    only in that the surrogate's weight on ``dc/dtheta`` is ``max(0, sigma)`` rather
    than ``max(y, sigma)``.
    """

    def setUp(self):
        self.loss = torch.tensor(5.0)

    @staticmethod
    def _hpr(m=3, **kwargs):
        kwargs.setdefault("lr", 0.1)
        kwargs.setdefault("penalty", 1.0)
        return ALM(m=m, augmentation="hpr", **kwargs)

    # --- equality groups: HPR must be a no-op ---

    def test_equality_groups_match_quadratic(self):
        c = torch.tensor([1.0, -2.0, 0.5])
        quad = ALM(m=3, lr=0.1, penalty=2.0, init_duals=0.7)
        hpr = self._hpr(m=3, penalty=2.0, init_duals=0.7)

        for _ in range(5):
            lag_q = quad.forward_update(self.loss, c)
            lag_h = hpr.forward_update(self.loss, c)
            self.assertTrue(torch.equal(lag_q, lag_h))
            self.assertTrue(torch.equal(quad.duals, hpr.duals))

    # --- the two branches, in closed form ---

    def test_active_branch(self):
        # sigma = y + rho*c > 0: surrogate is y'c + (rho/2)|c|^2 and the dual step is
        # lr*c, exactly as in the default mode.
        rho, lr, y0 = 2.0, 0.1, 1.0
        c = torch.tensor([1.0, 0.5, -0.25])  # sigma = 3.0, 2.0, 0.5 > 0
        hpr = self._hpr(m=3, lr=lr, penalty=rho, init_duals=y0, is_ineq=True)

        lag = hpr.forward_update(self.loss, c)

        y1 = torch.full((3,), y0) + lr * c
        self.assertTrue(torch.allclose(hpr.duals, y1))
        expected = self.loss + y1 @ c + 0.5 * rho * torch.dot(c, c)
        self.assertTrue(torch.allclose(lag, expected))

    def test_inactive_branch(self):
        # sigma = y + rho*c < 0: the surrogate is the constant -|y|^2/(2 rho) and the
        # dual decays geometrically by (1 - lr/rho).
        rho, lr, y0 = 2.0, 0.5, 1.0
        c = torch.tensor([-3.0, -1.0, -2.0])  # sigma = -5, -1, -3 < 0
        hpr = self._hpr(m=3, lr=lr, penalty=rho, init_duals=y0, is_ineq=True)

        lag = hpr.forward_update(self.loss, c)

        y1 = torch.full((3,), (1 - lr / rho) * y0)
        self.assertTrue(torch.allclose(hpr.duals, y1))
        expected = self.loss - torch.dot(y1, y1) / (2 * rho)
        self.assertTrue(torch.allclose(lag, expected))

    def test_general_case_matches_formula(self):
        # A mix of both branches, checked against the formula as written in the paper.
        rho, lr = 1.5, 0.3
        c = torch.tensor([2.0, -0.4, -1.0, 0.0])
        init = torch.tensor([0.2, 0.9, 0.5, 1.3])
        hpr = self._hpr(m=4, lr=lr, penalty=rho, init_duals=init.clone(), is_ineq=True)

        lag = hpr.forward_update(self.loss, c)

        y1 = (1 - lr / rho) * init + (lr / rho) * torch.clamp(init + rho * c, min=0.0)
        self.assertTrue(torch.allclose(hpr.duals, y1))
        shifted = torch.clamp(y1 + rho * c, min=0.0)
        expected = self.loss + (shifted @ shifted - y1 @ y1) / (2 * rho)
        self.assertTrue(torch.allclose(lag, expected))

    # --- the behavioural point: no pull on a comfortably feasible constraint ---

    def test_inactive_constraint_contributes_no_primal_gradient(self):
        # theta is scalar, c(theta) = theta - 1, so dc/dtheta = 1. At theta = -3 the
        # constraint is feasible with sigma = y + rho*c < 0, so HPR must leave the
        # primal gradient at df/dtheta alone while the default adds y * dc/dtheta.
        rho, y0 = 1.0, 0.8

        def grad_of(augmentation):
            theta = torch.tensor(-3.0, requires_grad=True)
            opt = ALM(
                m=1, lr=0.0, penalty=rho, init_duals=y0, is_ineq=True,
                augmentation=augmentation,
            )
            # lr=0 freezes the duals, isolating the surrogate's primal gradient.
            opt.forward_update(torch.zeros(()), (theta - 1.0).reshape(1)).backward()
            return theta.grad.item()

        self.assertEqual(grad_of("hpr"), 0.0)
        self.assertAlmostEqual(grad_of("quadratic"), y0, places=6)

    def test_lr_equal_penalty_gives_same_duals_as_quadratic(self):
        # At lr == penalty both modes perform y <- [y + rho*c]_+, the default via its
        # non-negativity clamp; only the surrogate differs.
        rho = 1.0
        c = torch.tensor([0.5, -0.3, -2.0])
        quad = ALM(m=3, lr=rho, penalty=rho, init_duals=0.6, is_ineq=True)
        hpr = self._hpr(m=3, lr=rho, penalty=rho, init_duals=0.6, is_ineq=True)

        lag_q = quad.forward_update(self.loss, c)
        lag_h = hpr.forward_update(self.loss, c)

        self.assertTrue(torch.allclose(quad.duals, torch.clamp(0.6 + rho * c, min=0.0)))
        self.assertTrue(torch.allclose(quad.duals, hpr.duals))
        self.assertFalse(torch.allclose(lag_q, lag_h))

    # --- momentum smooths the HPR ascent direction ---

    def test_momentum_smooths_ascent_direction(self):
        rho, lr, y0 = 2.0, 0.1, 1.0
        momentum, dampening = 0.9, 0.5
        c = torch.tensor([1.0, -3.0, -0.5])
        hpr = self._hpr(
            m=3, lr=lr, penalty=rho, init_duals=y0, is_ineq=True,
            momentum=momentum, dampening=dampening,
        )

        duals = torch.full((3,), y0)
        buffer = torch.zeros(3)
        for _ in range(3):
            d = (torch.clamp(duals + rho * c, min=0.0) - duals) / rho
            buffer = momentum * buffer + (1 - dampening) * d
            duals = duals + lr * buffer

            hpr.update(c)
            self.assertTrue(torch.allclose(hpr.duals, duals))
            self.assertTrue(
                torch.allclose(hpr.param_groups[0]["momentum_buffer"], buffer)
            )

    def test_forward_does_not_corrupt_momentum_buffer(self):
        c = torch.tensor([1.0, -3.0, -0.5])
        direct = self._hpr(m=3, init_duals=1.0, is_ineq=True, momentum=0.9)
        via_forward = self._hpr(m=3, init_duals=1.0, is_ineq=True, momentum=0.9)

        direct.update(c)
        via_forward.forward(self.loss, c)
        via_forward.update(c)

        self.assertTrue(torch.equal(direct.duals, via_forward.duals))

    def test_forward_update_and_separate_forward_update_agree(self):
        c = torch.tensor([1.0, -3.0, -0.5])
        combined = self._hpr(m=3, init_duals=1.0, is_ineq=True, momentum=0.9)
        separate = self._hpr(m=3, init_duals=1.0, is_ineq=True, momentum=0.9)

        combined.forward_update(self.loss, c)
        separate.forward(self.loss, c)
        separate.update(c)

        self.assertTrue(torch.equal(combined.duals, separate.duals))

    # --- multiple groups: the global quadratic must not be double counted ---

    def test_mixed_groups(self):
        rho = 1.0
        hpr = self._hpr(m=2, lr=0.1, penalty=rho, init_duals=0.5, is_ineq=True)
        hpr.add_constraint_group(m=2, lr=0.1, init_duals=torch.full((2,), 0.5))

        c = torch.tensor([0.5, -2.0, 1.0, -1.0])
        lag = hpr.forward(self.loss, c)

        y = torch.full((2,), 0.5)
        shifted = torch.clamp(y + rho * c[:2], min=0.0)
        expected = (
            self.loss
            + (shifted @ shifted - y @ y) / (2 * rho)     # inequality group
            + y @ c[2:] + 0.5 * rho * torch.dot(c[2:], c[2:])  # equality group
        )
        self.assertTrue(torch.allclose(lag, expected))

    # --- validation and checkpointing ---

    def test_rejects_nonpositive_penalty(self):
        with self.assertRaises(ValueError):
            ALM(m=3, lr=0.1, penalty=0.0, augmentation="hpr")
        with self.assertRaises(ValueError):
            ALM(m=3, lr=0.1, penalty=-1.0, augmentation="hpr")
        # zero penalty stays valid in the default mode (Cooper parity relies on it)
        ALM(m=3, lr=0.1, penalty=0.0)

    def test_rejects_unknown_augmentation(self):
        with self.assertRaises(ValueError):
            ALM(m=3, lr=0.1, penalty=1.0, augmentation="quadatric")

    def test_state_dict_round_trip(self):
        hpr = self._hpr(m=3, penalty=2.0, is_ineq=True)
        state_dict = hpr.state_dict()
        self.assertEqual(state_dict["state"]["augmentation"], "hpr")

        restored = ALM(m=3, lr=0.1, penalty=1.0)
        restored.load_state_dict(state_dict)
        self.assertEqual(restored.augmentation, "hpr")
        self.assertEqual(restored.penalty, 2.0)

    def test_legacy_state_dict_defaults_to_quadratic(self):
        hpr = self._hpr(m=3, penalty=2.0, is_ineq=True)
        state_dict = hpr.state_dict()
        del state_dict["state"]["augmentation"]

        hpr.load_state_dict(state_dict)
        self.assertEqual(hpr.augmentation, "quadratic")


class TestALMDDP(unittest.TestCase):
    def setUp(self):
        self.loss = torch.tensor(5.0)
        self.constraints = torch.tensor([1.0, 2.0, 3.0])
        self.pg = object()  # sentinel; real value only matters to dist.all_reduce

    def test_no_process_group_skips_all_reduce(self):
        alm = ALM(m=3, lr=0.1, penalty=1.0)
        with patch('torch.distributed.all_reduce') as mock_ar:
            alm.update(self.constraints)
            alm.forward_update(self.loss, self.constraints)
        mock_ar.assert_not_called()

    def test_update_calls_all_reduce_with_correct_args(self):
        alm = ALM(m=3, lr=0.1, penalty=1.0, process_group=self.pg)
        with patch('torch.distributed.all_reduce') as mock_ar:
            alm.update(self.constraints)
        mock_ar.assert_called_once()
        _, kwargs = mock_ar.call_args
        self.assertEqual(kwargs['op'], dist.ReduceOp.AVG)
        self.assertEqual(kwargs['group'], self.pg)

    def test_update_uses_reduced_constraints(self):
        # Simulate all_reduce replacing the tensor with worker-averaged values.
        reduced = torch.tensor([2.0, 4.0, 6.0])
        def fake_all_reduce(tensor, **kwargs):
            tensor.copy_(reduced)

        alm = ALM(m=3, lr=0.1, penalty=1.0, process_group=self.pg)
        with patch('torch.distributed.all_reduce', side_effect=fake_all_reduce):
            alm.update(self.constraints)

        self.assertTrue(torch.allclose(alm.duals, 0.1 * reduced))

    def test_update_does_not_mutate_input(self):
        # The all_reduce clone must be a detached copy; original tensor must be untouched.
        original = self.constraints.clone()
        alm = ALM(m=3, lr=0.1, penalty=1.0, process_group=self.pg)
        with patch('torch.distributed.all_reduce', side_effect=lambda t, **kw: t.fill_(99.0)):
            alm.update(self.constraints)
        self.assertTrue(torch.allclose(self.constraints, original))

    def test_forward_update_uses_reduced_constraints_for_dual(self):
        reduced = torch.tensor([2.0, 4.0, 6.0])
        def fake_all_reduce(tensor, **kwargs):
            tensor.copy_(reduced)

        alm = ALM(m=3, lr=0.1, penalty=1.0, process_group=self.pg)
        with patch('torch.distributed.all_reduce', side_effect=fake_all_reduce):
            alm.forward_update(self.loss, self.constraints)

        self.assertTrue(torch.allclose(alm.duals, 0.1 * reduced))

    def test_forward_update_lagrangian_uses_original_constraints(self):
        # Duals are updated with reduced constraints, but the Lagrangian must be
        # computed with the original constraints so autograd flows through ∂c/∂θ.
        reduced = torch.tensor([2.0, 4.0, 6.0])
        def fake_all_reduce(tensor, **kwargs):
            tensor.copy_(reduced)

        alm = ALM(m=3, lr=0.1, penalty=1.0, process_group=self.pg)
        with patch('torch.distributed.all_reduce', side_effect=fake_all_reduce):
            lagrangian = alm.forward_update(self.loss, self.constraints)

        updated_duals = 0.1 * reduced
        expected = (
            self.loss
            + updated_duals @ self.constraints
            + 0.5 * alm.penalty * torch.dot(self.constraints, self.constraints)
        )
        self.assertTrue(torch.allclose(lagrangian, expected))


if __name__ == "__main__":
    unittest.main()