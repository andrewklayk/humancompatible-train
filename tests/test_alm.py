import unittest
from torch.optim import Optimizer
from humancompatible.train.dual_optim import ALM
import torch

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
        self.assertEqual(self.alm_default.penalty, 1.0)
        self.assertEqual(self.alm_default.dual_range, (0.0, 100.0))

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


    def test_alm_state_dict(self):
        alm = ALM(m=3, lr=0.1, penalty=2.0, dual_range=(-1.0, 1.0))
        state_dict = alm.state_dict()
        self.assertEqual(state_dict["state"]["penalty"], 2.0)
        self.assertEqual(state_dict["state"]["dual_range"], (-1.0, 1.0))

if __name__ == "__main__":
    unittest.main()