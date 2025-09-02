import unittest
import torch
from humancompatible.train.algorithms import SSG

class TestSSG(unittest.TestCase):
    def setUp(self):
        # Simple model for testing
        self.model = torch.nn.Linear(2, 1)
        self.params = list(self.model.parameters())
        self.m = 1  # Number of constraints
        self.optimizer = SSG(
            self.params,
            m=self.m,
            lr=0.01,
            dual_lr=0.01
        )

    def test_initialization(self):
        # Test if the optimizer is initialized correctly
        self.assertEqual(len(self.optimizer.param_groups), 1)
        self.assertEqual(self.optimizer.m, self.m)
        self.assertEqual(self.optimizer.dual_lr, 0.01)

    def test_dual_step(self):
        # Test dual step saving constraint gradients
        for p in self.params:
            p.grad = torch.ones_like(p)
        self.optimizer.dual_step(0)
        
        for p in self.params:
            state = self.optimizer.state[p]
            self.assertTrue(state["c_grad"] is not None)

    def test_step_obj(self):
        p_pre_step = {}
        for p in self.params:
            p.grad = torch.ones_like(p)
            self.optimizer.state[p]['c_grad'] = [-1.*torch.ones_like(p)]
            p_pre_step[p] = p.detach().clone()

        self.optimizer.step(c_val=torch.tensor(-1.))

        for p in self.params:
            self.assertTrue(
                torch.all(
                        p == p_pre_step[p]-0.01*torch.ones_like(p)
                    )
                )

    def test_step_c(self):
        p_pre_step = {}
        for p in self.params:
            p.grad = torch.ones_like(p)
            self.optimizer.state[p]['c_grad'] = [-1.*torch.ones_like(p)]
            p_pre_step[p] = p.detach().clone()

        self.optimizer.step(c_val=torch.tensor(1.))

        for p in self.params:
            self.assertTrue(
                torch.all(
                        p == p_pre_step[p]+0.01*torch.ones_like(p)
                    )
                )