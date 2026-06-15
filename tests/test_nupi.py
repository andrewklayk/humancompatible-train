import unittest
import torch
from humancompatible.train.dual_optim import nuPI


class TestnuPI(unittest.TestCase):
    def setUp(self):
        self.loss = torch.tensor(5.0)
        self.constraints = torch.tensor([1.0, 2.0, 3.0])

    def test_nupi_initialization(self):
        opt = nuPI(m=3, nu=0.9, ki=0.01, kp=0.01, penalty=1.0)
        self.assertEqual(len(opt.duals), 3)

    def test_nupi_forward(self):
        opt = nuPI(m=3, nu=0.9, ki=0.01, kp=0.01, penalty=1.0)
        lagrangian = opt.forward(self.loss, self.constraints)
        expected = (self.loss
                    + opt.duals @ self.constraints
                    + 0.5 * opt.penalty * torch.dot(self.constraints, self.constraints))
        self.assertTrue(torch.allclose(lagrangian, expected))

    def test_nupi_update(self):
        # With zero buffer (initial state) and kp=0, update is purely integral: λ += ki * c
        opt = nuPI(m=3, nu=0.9, ki=0.1, kp=0.0, penalty=1.0)
        opt.update(self.constraints)
        self.assertTrue(torch.allclose(opt.duals, 0.1 * self.constraints))


class TestnuPIFixes(unittest.TestCase):
    """Tests for fix 1 (buffer in forward) and fix 2 (multi-group slicing)."""

    def setUp(self):
        self.loss = torch.tensor(5.0)

    # --- Fix 1: forward() must not advance the EMA buffer ---
    # nuPI's buffer is unconditionally updated in the original code, making this
    # the most severe instance of the bug: it fires even without any momentum setting.

    def test_forward_does_not_corrupt_ema_buffer(self):
        # Calling forward() then update() must give the same duals as update() alone.
        c = torch.tensor([1.0, 2.0, 3.0])
        opt_direct = nuPI(m=3, nu=0.9, ki=0.01, kp=0.05, penalty=1.0)
        opt_via_forward = nuPI(m=3, nu=0.9, ki=0.01, kp=0.05, penalty=1.0)

        opt_direct.update(c)

        opt_via_forward.forward(self.loss, c)
        opt_via_forward.update(c)

        self.assertTrue(torch.allclose(opt_direct.duals, opt_via_forward.duals))

    def test_forward_update_and_separate_forward_update_agree(self):
        c = torch.tensor([1.0, 2.0, 3.0])
        opt_combined = nuPI(m=3, nu=0.9, ki=0.01, kp=0.05, penalty=1.0)
        opt_separate = nuPI(m=3, nu=0.9, ki=0.01, kp=0.05, penalty=1.0)

        opt_combined.forward_update(self.loss, c)
        opt_separate.forward(self.loss, c)
        opt_separate.update(c)

        self.assertTrue(torch.allclose(opt_combined.duals, opt_separate.duals))

    # --- Fix 2: multi-group constraint slicing ---

    def test_multi_group_update_slices_correctly(self):
        # kp=0 so update is purely λ += ki * c, easy to verify
        opt = nuPI(m=2, nu=0.9, ki=0.1, kp=0.0, penalty=1.0)
        opt.add_constraint_group(m=3, nu=0.9, ki=0.2, kp=0.0)

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        opt.update(c)

        self.assertTrue(torch.allclose(opt.param_groups[0]["params"][0], 0.1 * c[:2]))
        self.assertTrue(torch.allclose(opt.param_groups[1]["params"][0], 0.2 * c[2:]))

    def test_multi_group_forward_lagrangian_correct(self):
        init0 = torch.tensor([1.0, 1.0])
        init1 = torch.tensor([1.0, 1.0, 1.0])
        opt = nuPI(m=2, nu=0.9, ki=0.1, kp=0.0, penalty=1.0, init_duals=init0)
        opt.add_constraint_group(m=3, nu=0.9, ki=0.2, kp=0.0, init_duals=init1)

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        lagrangian = opt.forward(self.loss, c)

        expected = (self.loss
                    + init0 @ c[:2]
                    + init1 @ c[2:]
                    + 0.5 * opt.penalty * torch.dot(c, c))
        self.assertTrue(torch.allclose(lagrangian, expected))

    def test_multi_group_forward_update_slices_correctly(self):
        opt = nuPI(m=2, nu=0.9, ki=0.1, kp=0.0, penalty=1.0)
        opt.add_constraint_group(m=3, nu=0.9, ki=0.2, kp=0.0)

        c = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        opt.forward_update(self.loss, c)

        self.assertTrue(torch.allclose(opt.param_groups[0]["params"][0], 0.1 * c[:2]))
        self.assertTrue(torch.allclose(opt.param_groups[1]["params"][0], 0.2 * c[2:]))


class TestnuPIFirstStep(unittest.TestCase):
    """Tests for the first-step correction (paper Lemma 2, eq. 15a vs 15c)."""

    def setUp(self):
        self.loss = torch.tensor(5.0)

    def test_first_step_no_proportional_term(self):
        # With ξ₀=0 (default) the paper says θ₁ = θ₀ + κᵢe₀ + κₚ·0 = θ₀ + κᵢe₀.
        # The old code wrongly applied the t≥1 formula: θ₁ = θ₀ + (κᵢ + κₚ(1−ν))e₀.
        c = torch.tensor([1.0, 2.0, 3.0])
        opt = nuPI(m=3, nu=0.9, ki=0.1, kp=0.5, penalty=1.0)
        opt.update(c)
        self.assertTrue(torch.allclose(opt.duals, 0.1 * c))

    def test_second_step_uses_general_formula(self):
        # After the first step, the t≥1 formula must kick in.
        # With ki=0.1, kp=0.5, nu=0.9 and the same c both steps:
        #   step 1: θ₁ = ki*c,  ξ₁ = (1-nu)*c = 0.1*c
        #   step 2: θ₂ = θ₁ + (ki + kp*(1-nu))*c - kp*(1-nu)*ξ₁
        c = torch.tensor([1.0, 2.0, 3.0])
        ki, kp, nu = 0.1, 0.5, 0.9
        opt = nuPI(m=3, nu=nu, ki=ki, kp=kp, penalty=1.0)
        opt.update(c)   # step 1
        theta1 = opt.duals.clone()  # = ki*c
        xi1 = (1 - nu) * c          # buffer after step 1
        opt.update(c)   # step 2
        expected = theta1 + (ki + kp * (1 - nu)) * c - kp * (1 - nu) * xi1
        self.assertTrue(torch.allclose(opt.duals, expected))

    def test_initialized_flag_set_after_first_update(self):
        opt = nuPI(m=3, nu=0.9, ki=0.1, kp=0.5, penalty=1.0)
        self.assertFalse(opt.param_groups[0].get("_momentum_initialized", False))
        opt.update(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(opt.param_groups[0]["_momentum_initialized"])

    def test_forward_does_not_set_initialized_flag(self):
        # forward() must not advance state; the flag must remain False after a Lagrangian call.
        opt = nuPI(m=3, nu=0.9, ki=0.1, kp=0.5, penalty=1.0)
        opt.forward(self.loss, torch.tensor([1.0, 2.0, 3.0]))
        self.assertFalse(opt.param_groups[0].get("_momentum_initialized", False))


if __name__ == "__main__":
    unittest.main()
