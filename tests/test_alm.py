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