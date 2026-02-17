import unittest
import torch
from torch.optim import Optimizer
from humancompatible.train.dual_optim.pbm import PBM
from humancompatible.train.dual_optim.barrier import quad_log, quad_log_der, quad_recipr, quad_recipr_der

class TestPBM(unittest.TestCase):
    def setUp(self):
        # Initialize PBM instances for reuse in tests
        self.pbm_default = PBM(m=3, mu=0.3, lr=0.1, penalty_update='dimin', pbf='quadratic_logarithmic')
        self.pbm_reciprocal = PBM(m=3, mu=0.3, lr=0.1, penalty_update='dimin', pbf='quadratic_reciprocal')
        self.pbm_const_penalty = PBM(m=3, mu=0.3, lr=0.1, penalty_update='const', pbf='quadratic_logarithmic')
        self.pbm_dimin_dual = PBM(m=3, mu=0.3, lr=0.1, penalty_update='dimin_dual', pbf='quadratic_logarithmic', init_duals=2.0)

        # Common test data
        self.loss = torch.tensor(5.0)
        self.constraints = torch.tensor([1.0, 2.0, 3.0])
        self.large_constraints = torch.tensor([10.0, 20.0, 30.0])

    def test_pbm_initialization(self):
        # Test initialization with m
        self.assertEqual(len(self.pbm_default.duals), 3)
        self.assertEqual(len(self.pbm_default.penalties), 3)
        self.assertEqual(self.pbm_default.defaults['mu'], 0.3)
        self.assertEqual(self.pbm_default.defaults['pbf'], 'quadratic_logarithmic')

        # Test initialization with init_duals and init_penalties
        init_duals = torch.tensor([1.0, 2.0, 3.0])
        init_penalties = torch.tensor([10.0, 20.0, 30.0])
        pbm = PBM(init_duals=init_duals, init_penalties=init_penalties, mu=0.3, lr=0.1, penalty_update='dimin', pbf='quadratic_logarithmic')
        self.assertTrue(torch.all(pbm.duals == init_duals))
        self.assertTrue(torch.all(pbm.penalties == init_penalties))

        # Test invalid initialization
        with self.assertRaises(ValueError):
            PBM(m=None, init_duals=None)

    def test_pbm_forward(self):
        # Test Lagrangian computation
        lagrangian = self.pbm_default.forward(self.loss, self.constraints)
        cdivp = self.constraints / self.pbm_default.penalties
        pbf_val = quad_log(cdivp)
        expected_lagrangian = self.loss + torch.dot(self.pbm_default.duals * self.pbm_default.penalties, pbf_val)
        self.assertTrue(torch.allclose(lagrangian, expected_lagrangian))

    def test_pbm_update_duals(self):
        # Test dual variable update
        cdivp = self.constraints / self.pbm_default.penalties
        pbf_der_val = quad_log_der(cdivp)
        expected_duals = self.pbm_default.duals * torch.clamp(pbf_der_val, 0.3, 1/0.3)

        self.pbm_default.update(self.constraints)
        self.assertTrue(torch.allclose(self.pbm_default.duals, expected_duals))

    def test_pbm_update_penalties_dimin(self):
        # Test penalty update with 'dimin' strategy
        initial_penalties = self.pbm_default.penalties.clone()
        self.pbm_default.update_penalties()
        expected_penalties = initial_penalties * 0.1
        self.assertTrue(torch.allclose(self.pbm_default.penalties, expected_penalties))

    def test_pbm_update_penalties_dimin_dual(self):
        # Test penalty update with 'dimin_dual' strategy
        initial_penalties = self.pbm_dimin_dual.penalties.clone()
        expected_penalties = initial_penalties * 0.1 * self.pbm_dimin_dual.duals

        self.pbm_dimin_dual.update_penalties()
        self.assertTrue(torch.allclose(self.pbm_dimin_dual.penalties, expected_penalties))

    def test_pbm_update_penalties_const(self):
        # Test penalty update with 'const' strategy
        initial_penalties = self.pbm_const_penalty.penalties.clone()
        self.pbm_const_penalty.update_penalties()
        self.assertTrue(torch.allclose(self.pbm_const_penalty.penalties, initial_penalties))

    def test_pbm_forward_update(self):
        # Test combined forward and update
        lagrangian = self.pbm_default.forward_update(self.loss, self.constraints)
        cdivp = self.constraints / self.pbm_default.penalties
        pbf_val = quad_log(cdivp)
        expected_lagrangian = self.loss + torch.dot(self.pbm_default.duals * self.pbm_default.penalties, pbf_val)
        self.assertTrue(torch.allclose(lagrangian, expected_lagrangian))

    def test_pbm_add_constraint_group(self):
        # Test adding a new constraint group
        self.pbm_default.add_constraint_group(m=2, mu=0.5, penalty_update='dimin', pbf='quadratic_reciprocal')
        self.assertEqual(len(self.pbm_default.duals), 5)
        self.assertEqual(len(self.pbm_default.penalties), 5)
        self.assertEqual(self.pbm_default.param_groups[1]['mu'], 0.5)
        self.assertEqual(self.pbm_default.param_groups[1]['pbf'], 'quadratic_reciprocal')
        self.assertEqual(self.pbm_default.param_groups[1]['lr'], 0.1)

if __name__ == '__main__':
    unittest.main()
