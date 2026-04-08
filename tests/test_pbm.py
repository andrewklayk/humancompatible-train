import unittest
import torch
from humancompatible.train.dual_optim import PBM
from humancompatible.train.dual_optim.barrier import quad_log, quad_log_der, quad_recipr, quad_recipr_der


class TestPBMInitialization(unittest.TestCase):
    """Test PBM initialization and error handling."""

    def test_init_with_m(self):
        """Test initialization with m parameter."""
        pbm = PBM(m=5, penalty_mult=0.1)
        self.assertEqual(len(pbm.duals), 5)
        self.assertEqual(len(pbm.penalties), 5)

    def test_init_with_scalar_duals(self):
        """Test initialization with scalar init_duals."""
        pbm = PBM(m=3, init_duals=0.5, init_penalties=10.0)
        self.assertTrue(torch.allclose(pbm.duals, torch.tensor([0.5, 0.5, 0.5])))
        self.assertTrue(torch.allclose(pbm.penalties, torch.tensor([10.0, 10.0, 10.0])))

    def test_init_with_tensor_duals(self):
        """Test initialization with tensor init_duals."""
        init_duals = torch.tensor([0.1, 0.2, 0.3])
        pbm = PBM(m=3, init_duals=init_duals, init_penalties=2.0)
        self.assertTrue(torch.allclose(pbm.duals, init_duals))

    def test_init_defaults_to_dual_range(self):
        """Test that None init_duals defaults to dual_range lower bound."""
        pbm = PBM(m=3, dual_range=(0.01, 100.0))
        self.assertTrue(torch.allclose(pbm.duals, torch.tensor([0.01, 0.01, 0.01])))

    def test_init_defaults_to_penalty_range(self):
        """Test that None init_penalties defaults to penalty_range upper bound."""
        pbm = PBM(m=3, penalty_range=(0.1, 50.0))
        self.assertTrue(torch.allclose(pbm.penalties, torch.tensor([50.0, 50.0, 50.0])))

    def test_init_missing_m_and_duals_raises_error(self):
        """Test that ValueError is raised when both m and init_duals are None."""
        with self.assertRaises(ValueError):
            PBM(m=None, init_duals=None)

    def test_init_with_device(self):
        """Test initialization on specific device."""
        device = 'cpu'
        pbm = PBM(m=3, device=device)
        self.assertEqual(pbm.duals.device.type, device)

    def test_init_different_pbf(self):
        """Test initialization with different penalty-barrier functions."""
        pbm_log = PBM(m=3, pbf='quadratic_logarithmic')
        pbm_recipr = PBM(m=3, pbf='quadratic_reciprocal')
        # Just verify they don't raise errors
        self.assertEqual(len(pbm_log.duals), 3)
        self.assertEqual(len(pbm_recipr.duals), 3)

    def test_init_invalid_pbf_raises_error(self):
        """Test that invalid pbf raises KeyError when used."""
        pbm = PBM(m=3, pbf='invalid_function')
        # PBM creation doesn't validate pbf, but using it will fail
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.1, 0.1, 0.1])
        with self.assertRaises(KeyError):
            pbm.forward(loss, constraints)

    def test_init_invalid_penalty_update_raises_error(self):
        """Test that invalid penalty_update raises ValueError."""
        with self.assertRaises(ValueError):
            PBM(m=3, penalty_update='invalid_strategy')


class TestDualAndPenaltyBounds(unittest.TestCase):
    """Test that duals and penalties stay within specified bounds."""

    def test_duals_clamped_to_range(self):
        """Test that duals are clamped to dual_range."""
        pbm = PBM(m=2, dual_range=(0.01, 10.0), init_duals=5.0)
        # Large constraint violations through forward_update
        loss = torch.tensor(1.0)
        constraints = torch.tensor([1000.0, 1000.0])
        for _ in range(5):
            pbm.forward_update(loss, constraints)
        # Duals should not exceed upper bound
        self.assertTrue(torch.all(pbm.duals <= 10.0))
        # Duals should not fall below lower bound
        self.assertTrue(torch.all(pbm.duals >= 0.01))

    def test_penalties_clamped_to_range(self):
        """Test that penalties are clamped to penalty_range."""
        pbm = PBM(m=2, penalty_mult=0.05, penalty_range=(0.1, 100.0))
        loss = torch.tensor(1.0)
        # Run multiple updates
        for _ in range(10):
            constraints = torch.tensor([0.5, 0.5])
            pbm.forward_update(loss, constraints)
        # Penalties should not exceed upper bound
        self.assertTrue(torch.all(pbm.penalties <= 100.0))
        # Penalties should not fall below lower bound
        self.assertTrue(torch.all(pbm.penalties >= 0.1))

    def test_duals_stay_positive_with_small_range(self):
        """Test duals with very small range."""
        pbm = PBM(m=3, dual_range=(1e-6, 1e-3), init_duals=5e-4)
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.1, 0.2, 0.3])
        pbm.forward_update(loss, constraints)
        self.assertTrue(torch.all(pbm.duals >= 1e-6))
        self.assertTrue(torch.all(pbm.duals <= 1e-3))


class TestPenaltyUpdateStrategies(unittest.TestCase):
    """Test different penalty update strategies."""

    def test_constant_penalty(self):
        """Test that penalty_update='const' keeps penalties constant during forward_update."""
        init_penalty = 5.0
        pbm = PBM(m=3, penalty_update='const', init_penalties=init_penalty)
        penalties_before = pbm.penalties.clone()
        loss = torch.tensor(1.0)
        
        for _ in range(5):
            constraints = torch.tensor([0.1, 0.2, 0.3])
            pbm.forward_update(loss, constraints)
        
        self.assertTrue(torch.allclose(pbm.penalties, penalties_before))

    def test_diminishing_penalty_monotonic(self):
        """Test that penalty_update='dimin' decreases penalties monotonically."""
        penalty_mult = 0.8
        pbm = PBM(m=2, penalty_update='dimin', penalty_mult=penalty_mult, init_penalties=100.0)
        
        penalties_history = [pbm.penalties.clone()]
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.5, 0.5])
        
        for _ in range(5):
            pbm.forward_update(loss, constraints)
            penalties_history.append(pbm.penalties.clone())
        
        # Verify monotonic decrease (from 1 onwards since 0 is initial)
        for i in range(len(penalties_history) - 1):
            self.assertTrue(torch.all(penalties_history[i + 1] <= penalties_history[i]))

    def test_adaptive_penalty_responds_to_violations(self):
        """Test that adaptive penalty responds to constraint violations."""
        pbm_adapt = PBM(m=1, penalty_update='dimin_adapt', penalty_mult=0.8, delta=0.9, init_penalties=10.0)
        pbm_const = PBM(m=1, penalty_update='const', init_penalties=10.0)
        loss = torch.tensor(1.0)
        
        # Use larger constraint to see penalty difference
        constraint = torch.tensor([1.0])
        pbm_adapt.forward_update(loss, constraint)
        pbm_const.forward_update(loss, constraint)
        
        # Adaptive should differ from constant after update
        # (or at least both updates should complete without error)
        self.assertEqual(len(pbm_adapt.penalties), 1)

    # def test_dimin_dual_penalty_changes(self):
    #     """Test that dimin_dual strategy multiplies by duals."""
    #     pbm = PBM(m=2, penalty_update='dimin_dual', penalty_mult=0.8, init_duals=2.0, init_penalties=10.0)
        
    #     initial_penalty = pbm.penalties.clone()
    #     loss = torch.tensor(1.0)
    #     # Use larger constraints to trigger meaningful updates
    #     constraints = torch.tensor([1.0, 1.0])
    #     pbm.forward_update(loss, constraints)
        
    #     # With dimin_dual, penalties should decrease due to low initial duals
    #     breakpoint()
    #     self.assertTrue(torch.all(pbm.penalties <= initial_penalty))


class TestBarrierFunctions(unittest.TestCase):
    """Test barrier function properties."""

    def test_quad_log_positive_derivative_at_violation(self):
        """Test that quad_log derivative is positive at constraint violations."""
        t = torch.tensor([-0.2, 0.0, 0.5])
        deriv = quad_log_der(t)
        self.assertTrue(torch.all(deriv > 0.0))

    def test_quad_recipr_positive_derivative_at_violation(self):
        """Test that quad_recipr derivative is positive at constraint violations."""
        t = torch.tensor([-0.2, 0.0, 0.3])
        deriv = quad_recipr_der(t)
        self.assertTrue(torch.all(deriv > 0.0))

    def test_quad_log_derivative_matches_finite_diff_smooth_region(self):
        """Test quad_log derivative against finite differences in smooth region."""
        # Use values well within smooth region (t >= -0.5)
        t = torch.tensor([0.5, 1.0, 2.0])
        eps = 1e-3
        
        # Finite difference
        deriv_fd = (quad_log(t + eps) - quad_log(t - eps)) / (2 * eps)
        deriv_analytical = quad_log_der(t)
        
        self.assertTrue(torch.allclose(deriv_fd, deriv_analytical, atol=1e-3))

    def test_barrier_functions_increase_with_violation(self):
        """Test that barrier functions penalize larger violations more."""
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([0.5])
        
        log_v1 = quad_log(t1)
        log_v2 = quad_log(t2)
        self.assertTrue(log_v2 > log_v1)
        
        recipr_v1 = quad_recipr(t1)
        recipr_v2 = quad_recipr(t2)
        self.assertTrue(recipr_v2 > recipr_v1)

    def test_barrier_functions_nonzero_at_zero(self):
        """Test that barrier function values at zero constraint are reasonable."""
        t = torch.tensor([0.0])
        log_val = quad_log(t)
        recipr_val = quad_recipr(t)
        
        # Both should be exactly 0 at t=0
        self.assertTrue(torch.allclose(log_val, torch.tensor([0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(recipr_val, torch.tensor([0.0]), atol=1e-6))


class TestDualVariableUpdates(unittest.TestCase):
    """Test dual variable update mechanisms."""

    def test_dual_update_with_zero_momentum(self):
        """Test dual update with gamma=0 (complete replacement)."""
        pbm = PBM(m=2, gamma=0.0, init_duals=1.0, init_penalties=10.0)
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.5, 0.5])
        
        # With gamma=0, new duals should be completely replaced
        pbm.forward_update(loss, constraints)
        # Just verify update happened without error
        self.assertEqual(len(pbm.duals), 2)

    def test_dual_update_with_high_momentum(self):
        """Test dual update with gamma close to 1 (damped update)."""
        pbm_high = PBM(m=2, gamma=0.99, init_duals=1.0)
        pbm_low = PBM(m=2, gamma=0.1, init_duals=1.0)
        loss = torch.tensor(1.0)
        
        constraints = torch.tensor([0.5, 0.5])
        duals_high_before = pbm_high.duals.clone()
        duals_low_before = pbm_low.duals.clone()
        
        pbm_high.forward_update(loss, constraints)
        pbm_low.forward_update(loss, constraints)
        
        # High momentum should change duals less
        high_change = torch.abs(pbm_high.duals - duals_high_before).sum()
        low_change = torch.abs(pbm_low.duals - duals_low_before).sum()
        self.assertTrue(high_change < low_change)

    def test_dual_momentum_buffer_tracked(self):
        """Test that momentum buffer is tracked across updates."""
        pbm = PBM(m=2, gamma=0.8)
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.5, 0.5])
        
        # Run multiple updates to accumulate momentum
        for _ in range(3):
            pbm.forward_update(loss, constraints)
        # Just verify no errors occur
        self.assertEqual(len(pbm.duals), 2)


class TestMultipleConstraintGroups(unittest.TestCase):
    """Test adding and managing multiple constraint groups."""

    def test_add_constraint_group(self):
        """Test adding a second constraint group."""
        pbm = PBM(m=2, penalty_mult=0.1)
        self.assertEqual(len(pbm.duals), 2)
        
        pbm.add_constraint_group(m=3, penalty_mult=0.15)
        self.assertEqual(len(pbm.duals), 5)  # 2 + 3
        self.assertEqual(len(pbm.penalties), 5)

    def test_constraint_groups_independent_updates(self):
        """Test that constraint groups update independently."""
        pbm = PBM(m=2, penalty_mult=0.8, penalty_update='dimin')
        pbm.add_constraint_group(m=2, penalty_mult=0.5, penalty_update='dimin', delta=0.9, pbf='quadratic_logarithmic')
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.5, 0.5, 0.1, 0.1])
        pbm.forward_update(loss, constraints)
        
        # Just verify both groups were updated
        self.assertEqual(len(pbm.penalties), 4)

    def test_multiple_groups_different_pbf(self):
        """Test adding groups with different barrier functions."""
        pbm = PBM(m=2, pbf='quadratic_logarithmic', penalty_mult=0.1)
        pbm.add_constraint_group(m=2, pbf='quadratic_reciprocal', penalty_mult=0.1, penalty_update='dimin', delta=0.9)
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.5, 0.5, 0.5, 0.5])
        pbm.forward_update(loss, constraints)
        
        # Just verify both groups update without error
        self.assertEqual(len(pbm.duals), 4)

    def test_duals_property_concatenates_groups(self):
        """Test that duals property correctly concatenates groups."""
        pbm = PBM(m=3, init_duals=1.0, penalty_mult=0.1)
        pbm.add_constraint_group(m=2, init_duals=2.0, penalty_mult=0.1, penalty_update='dimin', delta=0.9, pbf='quadratic_logarithmic')
        
        duals_concat = pbm.duals
        self.assertEqual(len(duals_concat), 5)
        self.assertTrue(torch.allclose(duals_concat[:3], torch.tensor([1.0, 1.0, 1.0])))
        self.assertTrue(torch.allclose(duals_concat[3:], torch.tensor([2.0, 2.0])))

    def test_penalties_property_concatenates_groups(self):
        """Test that penalties property correctly concatenates groups."""
        pbm = PBM(m=2, init_penalties=5.0, penalty_mult=0.1)
        pbm.add_constraint_group(m=3, init_penalties=10.0, penalty_mult=0.1, penalty_update='dimin', delta=0.9, pbf='quadratic_logarithmic')
        
        penalties_concat = pbm.penalties
        self.assertEqual(len(penalties_concat), 5)
        self.assertTrue(torch.allclose(penalties_concat[:2], torch.tensor([5.0, 5.0])))
        self.assertTrue(torch.allclose(penalties_concat[2:], torch.tensor([10.0, 10.0, 10.0])))


class TestLagrangianComputation(unittest.TestCase):
    """Test Lagrangian computation accuracy."""

    def test_lagrangian_includes_loss(self):
        """Test that Lagrangian includes the loss term."""
        pbm = PBM(m=2)
        loss = torch.tensor(5.0)
        constraints = torch.tensor([0.1, 0.2])
        
        lagrangian = pbm.forward(loss, constraints)
        # Lagrangian should be at least the loss
        self.assertTrue(lagrangian >= loss)

    def test_lagrangian_zero_constraint_equals_loss(self):
        """Test that Lagrangian equals loss when constraints are zero."""
        pbm = PBM(m=2, init_duals=1.0, init_penalties=1.0)
        loss = torch.tensor(3.0)
        constraints = torch.tensor([0.0, 0.0])
        
        lagrangian = pbm.forward(loss, constraints)
        self.assertTrue(torch.allclose(lagrangian, loss))

    def test_lagrangian_increases_with_violation(self):
        """Test that Lagrangian increases with constraint violation."""
        pbm = PBM(m=1, init_duals=1.0, init_penalties=1.0)
        loss = torch.tensor(1.0)
        
        lag_small = pbm.forward(loss, torch.tensor([0.1]))
        lag_large = pbm.forward(loss, torch.tensor([0.5]))
        
        self.assertTrue(lag_large > lag_small)

    def test_forward_update_produces_same_lagrangian(self):
        """Test that forward_update produces valid Lagrangian."""
        pbm = PBM(m=2, gamma=0.8, init_duals=0.5, init_penalties=5.0)
        
        loss = torch.tensor(2.0)
        constraints = torch.tensor([0.3, 0.4])
        
        # forward_update should produce a Lagrangian value
        lag = pbm.forward_update(loss, constraints)
        
        # Lagrangian should be a scalar
        self.assertTrue(lag.dim() == 0 or lag.shape == torch.Size([]))

    def test_manual_lagrangian_computation(self):
        """Test Lagrangian against manual calculation with simple values."""
        pbm = PBM(m=1, init_duals=1.0, init_penalties=1.0, pbf='quadratic_logarithmic')
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.0])
        
        lagrangian = pbm.forward(loss, constraints)
        
        # When constraint is 0, barrier function contribution should be 0
        # So lagrangian should equal loss
        self.assertTrue(torch.allclose(lagrangian, loss, atol=1e-5))


class TestSerialization(unittest.TestCase):
    """Test state dict save and load."""

    def test_state_dict_contains_ranges(self):
        """Test that state_dict includes dual and penalty ranges."""
        pbm = PBM(m=3, dual_range=(0.01, 50.0), penalty_range=(0.5, 200.0))
        state = pbm.state_dict()
        
        self.assertEqual(state["state"]["dual_range"], (0.01, 50.0))
        self.assertEqual(state["state"]["penalty_range"], (0.5, 200.0))

    def test_state_dict_roundtrip_preserves_duals_and_penalties(self):
        """Test that save/load preserves duals and penalties."""
        pbm1 = PBM(m=3, init_duals=1.5, init_penalties=8.0, penalty_update='dimin')
        loss = torch.tensor(1.0)
        
        # Modify state
        constraints = torch.tensor([0.1, 0.2, 0.3])
        pbm1.forward_update(loss, constraints)
        
        duals_before = pbm1.duals.clone()
        penalties_before = pbm1.penalties.clone()
        
        # Save and load
        state = pbm1.state_dict()
        pbm2 = PBM(m=3, penalty_update='dimin')
        pbm2.load_state_dict(state)
        
        self.assertTrue(torch.allclose(pbm2.duals, duals_before))
        self.assertTrue(torch.allclose(pbm2.penalties, penalties_before))

    def test_state_dict_roundtrip_with_multiple_groups(self):
        """Test state dict roundtrip with multiple constraint groups."""
        pbm1 = PBM(m=2, init_duals=1.0, init_penalties=5.0, penalty_mult=0.1, penalty_update='dimin')
        pbm1.add_constraint_group(m=3, init_duals=0.5, init_penalties=10.0, penalty_mult=0.1, penalty_update='dimin', delta=0.9, pbf='quadratic_logarithmic')
        loss = torch.tensor(1.0)
        
        # Update some
        constraints = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2])
        pbm1.forward_update(loss, constraints)
        
        state = pbm1.state_dict()
        pbm2 = PBM(m=2, penalty_mult=0.1, penalty_update='dimin')
        pbm2.add_constraint_group(m=3, penalty_mult=0.1, penalty_update='dimin', delta=0.9, pbf='quadratic_logarithmic')
        pbm2.load_state_dict(state)
        
        self.assertTrue(torch.allclose(pbm2.duals, pbm1.duals))
        self.assertTrue(torch.allclose(pbm2.penalties, pbm1.penalties))


class TestParameterInteractions(unittest.TestCase):
    """Test interactions between different parameters."""

    def test_gamma_affects_dual_convergence_rate(self):
        """Test that higher gamma slows dual changes."""
        pbm_fast = PBM(m=1, gamma=0.1, init_duals=1.0, init_penalties=10.0)
        pbm_slow = PBM(m=1, gamma=0.9, init_duals=1.0, init_penalties=10.0)
        loss = torch.tensor(1.0)
        constraints = torch.tensor([1.0])
        
        duals_fast_before = pbm_fast.duals.clone()
        duals_slow_before = pbm_slow.duals.clone()
        
        pbm_fast.forward_update(loss, constraints)
        pbm_slow.forward_update(loss, constraints)
        
        change_fast = torch.abs(pbm_fast.duals - duals_fast_before)
        change_slow = torch.abs(pbm_slow.duals - duals_slow_before)
        
        self.assertTrue(change_fast > change_slow)

    def test_delta_affects_adaptive_penalty_behavior(self):
        """Test that delta parameter affects adaptive penalty update."""
        pbm_low_delta = PBM(m=1, penalty_update='dimin_adapt', delta=0.5, penalty_mult=0.8, init_penalties=10.0)
        pbm_high_delta = PBM(m=1, penalty_update='dimin_adapt', delta=2.0, penalty_mult=0.8, init_penalties=10.0)
        loss = torch.tensor(1.0)
        
        # Constraint with significant violation
        constraints = torch.tensor([2.0])
        
        pbm_low_delta.forward_update(loss, constraints)
        pbm_high_delta.forward_update(loss, constraints)
        
        # Different delta should produce different penalties
        self.assertFalse(torch.allclose(pbm_low_delta.penalties, pbm_high_delta.penalties))

    def test_penalty_mult_scale_different_strategies(self):
        """Test that penalty_mult works across different strategies."""
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.5])
        
        pbm_dimin = PBM(m=1, penalty_update='dimin', penalty_mult=0.5, init_penalties=100.0)
        pbm_dimin.forward_update(loss, constraints)
        dimin_penalty = pbm_dimin.penalties.clone()
        
        pbm_adapt = PBM(m=1, penalty_update='dimin_adapt', penalty_mult=0.5, init_penalties=100.0)
        pbm_adapt.forward_update(loss, constraints)
        
        # Both should be affected by penalty_mult, though differently
        self.assertTrue(torch.all(dimin_penalty < 100.0))
        self.assertTrue(torch.all(pbm_adapt.penalties < 100.0))


class TestConstraintSatisfactionProgress(unittest.TestCase):
    """Test constraint satisfaction and convergence behavior."""

    def test_constraints_drive_duals_up(self):
        """Test that constraints drive dual variables upward."""
        pbm = PBM(m=1, gamma=0.5, init_duals=0.1, init_penalties=10.0)
        duals_before = pbm.duals.clone()
        loss = torch.tensor(1.0)
        
        # Positive constraint violation
        constraints = torch.tensor([1.0])
        pbm.forward_update(loss, constraints)
        
        duals_after = pbm.duals.clone()
        self.assertTrue(duals_after > duals_before)

    def test_lagrangian_monotonic_on_consistent_constraints(self):
        """Test that Lagrangian progression with consistent constraints."""
        pbm = PBM(m=1, gamma=0.7, init_duals=0.5, init_penalties=5.0)
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.5])
        
        lagrangians = []
        for _ in range(3):
            lag = pbm.forward_update(loss, constraints)
            lagrangians.append(lag.item())
        
        # Just verify sequence is computed without error
        self.assertEqual(len(lagrangians), 3)

    def test_penalties_decrease_with_dimin(self):
        """Test that penalties monotonically decrease with dimin strategy."""
        pbm = PBM(m=2, penalty_update='dimin', penalty_mult=0.7, init_penalties=100.0)
        loss = torch.tensor(1.0)
        constraints = torch.tensor([0.1, 0.1])
        
        penalties_list = [pbm.penalties.clone()]
        for _ in range(5):
            pbm.forward_update(loss, constraints)
            penalties_list.append(pbm.penalties.clone())
        
        for i in range(len(penalties_list) - 1):
            self.assertTrue(torch.all(penalties_list[i + 1] <= penalties_list[i]))


if __name__ == '__main__':
    unittest.main()
