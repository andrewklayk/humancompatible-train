import unittest
import torch
from humancompatible.train.dual_optim import iALM


class TestiALM(unittest.TestCase):
    def setUp(self):
        self.loss = torch.tensor(5.0)
        self.constraints = torch.tensor([1.0, 2.0, 3.0])

    def test_ialm_initialization(self):
        alm = iALM(m=3, beta=1.0, penalty=1.0)
        self.assertEqual(len(alm.duals), 3)

    def test_ialm_forward(self):
        alm = iALM(m=3, beta=1.0, penalty=1.0)
        lagrangian = alm.forward(self.loss, self.constraints)
        beta = alm.param_groups[0]["beta"]
        expected = (self.loss
                    + alm.duals @ self.constraints
                    + 0.5 * beta * torch.dot(self.constraints, self.constraints))
        self.assertTrue(torch.allclose(lagrangian, expected))

    def test_ialm_update(self):
        # init_duals=zeros so the baseline is 0; duals should increase toward constraints
        alm = iALM(m=3, beta=1.0, gamma=1.0, sigma=1.0, penalty=1.0, init_duals=torch.zeros(3))
        duals_before = alm.duals.clone()
        alm.update(self.constraints)
        self.assertTrue(torch.all(alm.duals > duals_before))


class TestiALMFixes(unittest.TestCase):
    """Tests for fix 1 (momentum buffer in forward) and fix 2 (multi-group slicing)."""

    def setUp(self):
        self.loss = torch.tensor(5.0)

    # --- Fix 1: forward() must not advance the momentum buffer ---

    def test_forward_does_not_corrupt_momentum_buffer(self):
        # Calling forward() then update() must give the same duals as update() alone.
        c = torch.tensor([1.0, 2.0, 3.0])
        alm_direct = iALM(m=3, beta=1.0, gamma=1e6, sigma=1.0, momentum=0.9)
        alm_via_forward = iALM(m=3, beta=1.0, gamma=1e6, sigma=1.0, momentum=0.9)

        alm_direct.update(c)

        alm_via_forward.forward(self.loss, c)
        alm_via_forward.update(c)

        self.assertTrue(torch.allclose(alm_direct.duals, alm_via_forward.duals))

    def test_forward_update_and_separate_forward_update_agree(self):
        c = torch.tensor([1.0, 2.0, 3.0])
        alm_combined = iALM(m=3, beta=1.0, gamma=1e6, sigma=1.0, momentum=0.9)
        alm_separate = iALM(m=3, beta=1.0, gamma=1e6, sigma=1.0, momentum=0.9)

        alm_combined.forward_update(self.loss, c)
        alm_separate.forward(self.loss, c)
        alm_separate.update(c)

        self.assertTrue(torch.allclose(alm_combined.duals, alm_separate.duals))

    # --- Fix 2: multi-group constraint slicing ---

    def test_multi_group_update_slices_correctly(self):
        alm = iALM(m=2, beta=1.0, gamma=1e6, sigma=1.0, init_duals=torch.zeros(2))
        alm.add_constraint_group(m=3, beta=1.0, gamma=1e6, sigma=1.0, init_duals=torch.zeros(3))

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        alm.update(c)

        # With large gamma step ≈ 1.0 for both groups, so duals ≈ c_slice
        self.assertTrue(torch.allclose(alm.param_groups[0]["params"][0], c[:2], atol=1e-4))
        self.assertTrue(torch.allclose(alm.param_groups[1]["params"][0], c[2:], atol=1e-4))

    def test_multi_group_forward_lagrangian_correct(self):
        init0 = torch.tensor([1.0, 1.0])
        init1 = torch.tensor([1.0, 1.0, 1.0])
        alm = iALM(m=2, beta=1.0, gamma=1.0, sigma=1.0, init_duals=init0)
        alm.add_constraint_group(m=3, beta=1.0, gamma=1.0, sigma=1.0, init_duals=init1)

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        lagrangian = alm.forward(self.loss, c)

        beta = alm.param_groups[0]["beta"]
        expected = (self.loss
                    + init0 @ c[:2]
                    + init1 @ c[2:]
                    + 0.5 * beta * torch.dot(c, c))
        self.assertTrue(torch.allclose(lagrangian, expected))


if __name__ == "__main__":
    unittest.main()
