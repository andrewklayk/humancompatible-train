import unittest
import torch
from torch import Tensor
from humancompatible.train.algorithms import SSLALM_Adam

class TestSSLALMAdam(unittest.TestCase):
    def setUp(self):
        # Simple model for testing
        self.model = torch.nn.Linear(2, 1)
        self.params = list(self.model.parameters())
        self.m = 2  # Number of constraints
        self.optimizer = SSLALM_Adam(
            self.params,
            m=self.m,
            lr=0.01,
            dual_lr=0.01,
            dual_bound=100.0,
            rho=1.0,
            mu=2.0,
            beta=0.5,
            beta1=0.9,
            beta2=0.999
        )

    def test_initialization(self):
        # Test if the optimizer is initialized correctly
        self.assertEqual(len(self.optimizer.param_groups), 1)
        self.assertEqual(self.optimizer.m, self.m)
        self.assertEqual(self.optimizer.dual_lr, 0.01)
        self.assertEqual(self.optimizer.dual_bound, 100.0)
        self.assertEqual(self.optimizer.rho, 1.0)
        self.assertEqual(self.optimizer.mu, 2.0)
        self.assertEqual(self.optimizer.beta, 0.5)
        self.assertTrue(isinstance(self.optimizer._dual_vars, Tensor))
        self.assertEqual(self.optimizer._dual_vars.shape, (self.m,))

    def test_dual_step(self):
        # Test dual variable update
        for p in self.params:
            p.grad = torch.ones_like(p)
        c_val = torch.tensor([0.5, 0.1])
        self.optimizer.dual_step(0, c_val[0])
        self.assertEqual(self.optimizer._dual_vars[0], 0.005)  # 0 + 0.01 * 0.5
        self.optimizer.dual_step(1, c_val[1])
        self.assertEqual(self.optimizer._dual_vars[1], 0.001)  # 0 + 0.01 * 0.1
        for p in self.params:
            torch.testing.assert_close(
                self.optimizer.state[p]['l_term_grad'],
                p.grad * self.optimizer._dual_vars[0] + p.grad * self.optimizer._dual_vars[1]
            )
            torch.testing.assert_close(
                self.optimizer.state[p]['aug_term_grad'],
                p.grad * c_val[0] + p.grad * c_val[1]
            )

    def test_dual_bound(self):
        # Test dual variable bounding
        for p in self.params:
            p.grad = torch.ones_like(p)
        self.optimizer._dual_vars = torch.tensor([101.0, -1.0])
        c_val = torch.tensor([1.0, -1.0])
        self.optimizer.dual_step(0, c_val[0])
        self.optimizer.dual_step(1, c_val[1])
        self.assertEqual(self.optimizer._dual_vars[0].item(), 0.0)  # Should be zeroed out
        self.assertNotEqual(self.optimizer._dual_vars[1].item(), 0.0)  # Should NOT be zeroed out

    # ADD TEST DEALING WITH CONSTRAINTS THAT DONT USE SOME OF THE PARAMS

    def test_step(self):
        # Test primal parameter update
        # Mock gradients and constraint gradients
        p_pre_step = {}
        for p in self.params:
            p.grad = torch.ones_like(p)
            p_pre_step[p] = p.detach().clone()

        c_val = torch.tensor([0.1, -0.1])
        self.optimizer._dual_vars = torch.ones(2)
        
        for p in self.params:
            # self.optimizer.state[p]["c_grad"] = [g.clone() for g in c_grads[p]]
            self.optimizer.state[p]["smoothing"] = p.detach().clone()
            
            self.optimizer.state[p]["step"] = 0
            self.optimizer.state[p]["exp_avg"] = torch.ones_like(p)
            self.optimizer.state[p]["exp_avg_sq"] = torch.ones_like(p)
            self.optimizer.state[p]["l_term_grad"] = torch.ones_like(p)
            self.optimizer.state[p]["aug_term_grad"] = torch.ones_like(p)

        self.optimizer.step(c_val)
        # Check if G is computed and parameters are updated
        for i, p in enumerate(self.params):
            # assert correct update of params
            beta1 = 0.9
            beta2 = 0.999
            G_i = torch.zeros_like(p)
            G_i.add_(p.grad).add_(torch.ones_like(p)).add_(torch.ones_like(p), alpha=self.optimizer.rho)

            fm = beta1*torch.ones_like(p) + (1-beta1)*G_i
            sm = beta2*torch.ones_like(p) + (1-beta2)*(torch.pow(G_i,2))
            fm_bc = fm/(1-beta1)
            sm_bc = sm/(1-beta2)

            self.assertTrue(
                torch.equal(
                    p,
                    p_pre_step[p] - 0.01*fm_bc/(sm_bc.sqrt() + 1e-8)
                )
            )
            # assert correct update of exp_avg and exp_avg_sq
            self.assertTrue(
                torch.equal(
                    self.optimizer.state[p]['exp_avg'],
                    fm
                )
            )
            self.assertTrue(
                torch.equal(
                    self.optimizer.state[p]['exp_avg_sq'],
                    sm
                )
            )

    def test_dual_step_with_invalid_c_val(self):
        # Test step with invalid c_val (wrong shape)
        with self.assertRaises(ValueError):
            self.optimizer.dual_step(0, torch.tensor([0.1, 0.5]))

    def test_smoothing_update(self):
        # Test smoothing term update
        p_pre_step = {}
        for p in self.params:
            p.grad = torch.ones_like(p)
            p_pre_step[p] = p.detach().clone()
        c_val = torch.tensor([0.1, -0.1])
        self.optimizer.step(c_val)
        for p in self.params:
            state = self.optimizer.state[p]
            self.assertTrue("smoothing" in state)
            self.assertTrue(torch.all(state["smoothing"] == p_pre_step[p]))

    def test_error_handling(self):
        # Test error handling for invalid inputs
        with self.assertRaises(ValueError):
            SSLALM_Adam(self.params, m=self.m, lr=-0.01)
        with self.assertRaises(ValueError):
            SSLALM_Adam(self.params, m=self.m, dual_lr=-0.01)
        with self.assertRaises(ValueError):
            SSLALM_Adam(self.params, m=self.m, init_dual_vars=torch.tensor([1.0]))

if __name__ == "__main__":
    unittest.main()
