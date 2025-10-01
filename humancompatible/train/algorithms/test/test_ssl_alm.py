import unittest
import torch
from torch import Tensor
from humancompatible.train.algorithms import SSLALM

class TestSSLALM(unittest.TestCase):
    def setUp(self):
        # Simple model for testing
        self.model = torch.nn.Linear(2, 1)
        self.params = list(self.model.parameters())
        self.m = 2  # Number of constraints
        self.optimizer = SSLALM(
            self.params,
            m=self.m,
            lr=0.01,
            dual_lr=0.01,
            dual_bound=100.0,
            rho=1.0,
            mu=2.0,
            beta=0.5,
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
        c_val = torch.tensor([0.5, 0.1])
        self.optimizer.dual_step(0, c_val[0])
        self.assertEqual(self.optimizer._dual_vars[0], 0.005)  # 0 + 0.01 * 0.5
        self.optimizer.dual_step(1, c_val[1])
        self.assertEqual(self.optimizer._dual_vars[1], 0.001)  # 0 + 0.01 * 0.1

    def test_dual_bound(self):
        # Test dual variable bounding
        self.optimizer._dual_vars = torch.tensor([101.0, -1.0])
        c_val = torch.tensor([1.0, -1.0])
        self.optimizer.dual_step(0, c_val[0])
        self.optimizer.dual_step(1, c_val[1])
        self.assertEqual(self.optimizer._dual_vars[0], 0.0)  # Should be zeroed out
        self.assertNotEqual(self.optimizer._dual_vars[1], 0.0)  # Should NOT be zeroed out

# ADD TEST DEALING WITH CONSTRAINTS THAT DONT USE SOME OF THE PARAMS

    def test_step(self):
        # Test primal parameter update
        # Mock gradients and constraint gradients
        p_pre_step = {} 
        for p in self.params:
            p.grad = torch.ones_like(p)
            p_pre_step[p] = p.detach().clone()

        c_val = torch.tensor([0.1, -0.1])
        c_grads = {p: [torch.ones_like(p) for _ in c_val] for p in self.params}
        
        for p in self.params:
            self.optimizer.state[p]["c_grad"] = [g.clone() for g in c_grads[p]]
            self.optimizer.state[p]["smoothing"] = p.detach().clone()
        
        self.optimizer._dual_vars = torch.ones(2)
        G = self.optimizer.step(c_val)
        # Check if G is computed and parameters are updated
        self.assertEqual(len(G), len(self.params))
        for i, p in enumerate(self.params):
            self.assertTrue(
                torch.equal(
                    G[i],
                    (
                        p.grad +
                        sum(_lambda  * c_grads[p][j] for j, _lambda in enumerate(self.optimizer._dual_vars)) +
                        sum([c_grads[p][j] * cv for j, cv in enumerate(c_val)])
                    )
                )
            )

    def test_step_with_none_c_val(self):
        # Test step with None c_val (should use self.c_vals)
        c_val = torch.tensor([0.1, -0.1])
        self.optimizer.dual_step(0, c_val[0])
        self.optimizer.dual_step(1, c_val[1])

        p_pre_step = {}
        for p in self.params:
            p.grad = torch.ones_like(p)
            p_pre_step[p] = p.detach().clone()

        c_grads = {p: [torch.ones_like(p) for _ in c_val] for p in self.params}
        for p in self.params:
            self.optimizer.state[p]["c_grad"] = [g.clone() for g in c_grads[p]]
            self.optimizer.state[p]["smoothing"] = p.detach().clone()
        
        G = self.optimizer.step()
        # Check if G is computed and parameters are updated
        self.assertEqual(len(G), len(self.params))
        for i, p in enumerate(self.params):
            self.assertTrue(
                torch.equal(
                    G[i],
                    (
                        p.grad +
                        sum(_lambda  * c_grads[p][j] for j, _lambda in enumerate(self.optimizer._dual_vars)) +
                        sum([c_grads[p][j] * cv for j, cv in enumerate(c_val)])
                    )
                )
            )

    def test_step_with_invalid_c_val(self):
        # Test step with invalid c_val (wrong shape)
        with self.assertRaises(ValueError):
            self.optimizer.step(torch.tensor([0.1]))

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
            SSLALM(self.params, m=self.m, lr=-0.01)
        with self.assertRaises(ValueError):
            SSLALM(self.params, m=self.m, dual_lr=-0.01)
        with self.assertRaises(ValueError):
            SSLALM(self.params, m=self.m, init_dual_vars=torch.tensor([1.0]))

if __name__ == "__main__":
    unittest.main()
