"""
Tests for the shared DualOptimizer base class.

Covers the behaviour introduced with the base class -- constraint-input
normalization and validation, group names, declared bounds -- plus two defects
that the per-class implementations shared and that centralising fixed.
"""

import unittest

import torch

from humancompatible.train.dual_optim import ALM, PBM, iALM, nuPI
from humancompatible.train.dual_optim.base import DualOptimizer

LOSS = torch.tensor(5.0)


def _all_optimizers(m=5, **kw):
    """One instance of every dual optimizer, each with a single group of size m."""
    return [
        ALM(m=m, lr=0.1, **kw),
        iALM(m=m, beta=1.0, sigma=1.0, gamma=1.0, **kw),
        nuPI(m=m, nu=0.5, ki=0.1, kp=1.0, **kw),
        PBM(m=m, gamma_annealing=False, penalty_annealing=False, **kw),
    ]


def _two_group(cls):
    """A two-group optimizer of sizes 2 and 3, named 'a' and 'b'."""
    if cls is ALM:
        opt = ALM(m=2, lr=0.1)
        opt.add_constraint_group(m=3, lr=0.2, name="b")
    elif cls is iALM:
        opt = iALM(m=2, beta=1.0, sigma=1.0, gamma=1.0)
        opt.add_constraint_group(m=3, beta=1.0, sigma=1.0, gamma=1.0, name="b")
    elif cls is nuPI:
        opt = nuPI(m=2, nu=0.5, ki=0.1, kp=1.0)
        opt.add_constraint_group(m=3, nu=0.5, ki=0.1, kp=1.0, name="b")
    else:
        opt = PBM(m=2, gamma_annealing=False, penalty_annealing=False)
        opt.add_constraint_group(m=3, name="b")
    opt.param_groups[0]["name"] = "a"
    return opt


class TestInheritance(unittest.TestCase):
    def test_all_dual_optimizers_share_the_base(self):
        for opt in _all_optimizers():
            self.assertIsInstance(opt, DualOptimizer)
            self.assertIsInstance(opt, torch.optim.Optimizer)

    def test_step_is_update_alias(self):
        # Identity comparison would fail: Optimizer.__init_subclass__ wraps each
        # subclass's `step` with a profiling hook, so compare behaviour instead.
        c = torch.tensor([1.0, 2.0, 3.0])
        for via_step, via_update in zip(_all_optimizers(m=3), _all_optimizers(m=3)):
            via_step.step(c)
            via_update.update(c)
            self.assertTrue(
                torch.equal(via_step.duals, via_update.duals),
                type(via_step).__name__,
            )

    def test_m_and_names_properties(self):
        for cls in (ALM, iALM, nuPI, PBM):
            opt = _two_group(cls)
            self.assertEqual(opt.m, 5, cls.__name__)
            self.assertEqual(opt.names, ["a", "b"], cls.__name__)

    def test_groups_get_default_names(self):
        opt = ALM(m=2, lr=0.1)
        opt.add_constraint_group(m=3, lr=0.2)
        self.assertEqual(opt.names, ["group0", "group1"])


class TestGatherConstraints(unittest.TestCase):
    """The three accepted input forms must agree, and mistakes must be caught."""

    def test_sequence_form_matches_flat_form(self):
        flat = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        parts = [torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0, 30.0])]
        for cls in (ALM, iALM, nuPI, PBM):
            a, b = _two_group(cls), _two_group(cls)
            la = a.forward_update(LOSS, flat)
            lb = b.forward_update(LOSS, parts)
            self.assertTrue(torch.equal(la, lb), cls.__name__)
            self.assertTrue(torch.equal(a.duals, b.duals), cls.__name__)

    def test_mapping_form_matches_flat_form(self):
        flat = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        mapping = {"b": torch.tensor([10.0, 20.0, 30.0]), "a": torch.tensor([1.0, 2.0])}
        for cls in (ALM, iALM, nuPI, PBM):
            a, b = _two_group(cls), _two_group(cls)
            la = a.forward_update(LOSS, flat)
            lb = b.forward_update(LOSS, mapping)
            self.assertTrue(torch.equal(la, lb), cls.__name__)
            self.assertTrue(torch.equal(a.duals, b.duals), cls.__name__)

    def test_mapping_is_order_independent(self):
        # The point of the mapping form: the caller's ordering cannot silently
        # attach a dual to the wrong constraint.
        forward = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0, 5.0])}
        reverse = {"b": torch.tensor([3.0, 4.0, 5.0]), "a": torch.tensor([1.0, 2.0])}
        a, b = _two_group(ALM), _two_group(ALM)
        a.update(forward)
        b.update(reverse)
        self.assertTrue(torch.equal(a.duals, b.duals))

    def test_flat_form_is_not_copied(self):
        # The returned tensor must keep the caller's autograd graph.
        opt = ALM(m=3, lr=0.1)
        c = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        lagrangian = opt.forward(LOSS, c)
        lagrangian.backward()
        self.assertIsNotNone(c.grad)

    def test_wrong_flat_length_raises(self):
        for cls in (ALM, iALM, nuPI, PBM):
            opt = _two_group(cls)
            with self.assertRaises(ValueError, msg=cls.__name__):
                opt.update(torch.tensor([1.0, 2.0, 3.0]))
            with self.assertRaises(ValueError, msg=cls.__name__):
                opt.update(torch.ones(6))

    def test_wrong_sequence_arity_raises(self):
        opt = _two_group(ALM)
        with self.assertRaises(ValueError):
            opt.update([torch.ones(2)])
        with self.assertRaises(ValueError):
            opt.update([torch.ones(2), torch.ones(3), torch.ones(1)])

    def test_wrong_sequence_group_size_raises(self):
        opt = _two_group(ALM)
        with self.assertRaises(ValueError):
            opt.update([torch.ones(3), torch.ones(2)])

    def test_bad_mapping_keys_raise(self):
        opt = _two_group(ALM)
        with self.assertRaises(ValueError):
            opt.update({"a": torch.ones(2)})  # missing 'b'
        with self.assertRaises(ValueError):
            opt.update({"a": torch.ones(2), "b": torch.ones(3), "c": torch.ones(1)})

    def test_scalar_constraint_is_promoted(self):
        # ALM/iALM/nuPI promoted a 0-dim tensor; PBM did not and broke on one.
        for opt in _all_optimizers(m=1):
            lagrangian = opt.forward_update(LOSS, torch.tensor(0.25))
            self.assertEqual(lagrangian.shape, ())
            self.assertEqual(len(opt.duals), 1)


class TestBoundsAndViolation(unittest.TestCase):
    def test_no_bounds_declared(self):
        opt = ALM(m=3, lr=0.1)
        self.assertIsNone(opt.bounds)
        # With no bounds, violation is just the largest constraint value.
        self.assertEqual(opt.violation(torch.tensor([-1.0, 0.5, 0.2])).item(), 0.5)

    def test_declared_bounds_shift_the_violation(self):
        opt = ALM(m=2, lr=0.1)
        opt.add_constraint_group(m=1, lr=0.1, bound=1.0)
        self.assertTrue(torch.equal(opt.bounds, torch.tensor([0.0, 0.0, 1.0])))
        # third constraint is 0.9 against a bound of 1.0 -> satisfied by 0.1
        v = opt.violation(torch.tensor([-0.5, -0.2, 0.9]))
        self.assertAlmostEqual(v.item(), -0.1, places=6)

    def test_violation_accepts_every_input_form(self):
        opt = _two_group(ALM)
        flat = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
        self.assertEqual(opt.violation(flat).item(), 30.0)
        self.assertEqual(
            opt.violation([torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0, 30.0])]).item(),
            30.0,
        )


class TestSharedDefectsFixed(unittest.TestCase):
    def test_multi_group_state_dict_roundtrips(self):
        # PyTorch numbers params globally across groups, so the old per-class
        # `params[param_id]` lookup raised IndexError for any second group.
        for cls in (ALM, iALM, nuPI, PBM):
            src = _two_group(cls)
            src.update(torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0]))
            state = src.state_dict()

            dst = _two_group(cls)
            dst.load_state_dict(state)
            self.assertTrue(torch.equal(dst.duals, src.duals), cls.__name__)
            if isinstance(src, PBM):
                self.assertTrue(torch.equal(dst.penalties, src.penalties))

    def test_defaults_carry_no_tensor_state(self):
        # A tensor left in `defaults` would be shared by reference with every
        # group added later, silently aliasing their momentum buffers.
        for opt in _all_optimizers(m=3, **{}):
            for key, value in opt.defaults.items():
                self.assertNotIsInstance(
                    value, torch.Tensor, f"{type(opt).__name__}.defaults[{key!r}]"
                )

    def test_momentum_buffers_are_not_shared_between_groups(self):
        opt = ALM(m=2, lr=0.1, momentum=0.9)
        opt.add_constraint_group(m=3, lr=0.1, momentum=0.9)
        a, b = opt.param_groups[0], opt.param_groups[1]
        self.assertIsNot(a["momentum_buffer"], b["momentum_buffer"])
        opt.update(torch.tensor([1.0, 1.0, 5.0, 5.0, 5.0]))
        self.assertFalse(torch.equal(a["momentum_buffer"], b["momentum_buffer"][:2]))


class TestDistributedAvailableEverywhere(unittest.TestCase):
    """Every dual optimizer accepts a process group, not just ALM."""

    def test_process_group_accepted_and_used(self):
        from unittest.mock import patch

        sentinel = object()
        reduced = torch.tensor([2.0, 4.0, 6.0])

        def fake_all_reduce(tensor, **kwargs):
            tensor.copy_(reduced)

        for cls, kw in (
            (ALM, dict(lr=0.1)),
            (iALM, dict(beta=1.0, sigma=1.0, gamma=1.0)),
            (nuPI, dict(nu=0.5, ki=0.1, kp=1.0)),
            (PBM, dict(gamma_annealing=False, penalty_annealing=False)),
        ):
            opt = cls(m=3, process_group=sentinel, **kw)
            local = torch.tensor([1.0, 1.0, 1.0])
            with patch("torch.distributed.all_reduce", side_effect=fake_all_reduce) as mock:
                opt.update(local)
            mock.assert_called_once()
            # the caller's tensor must not be touched
            self.assertTrue(torch.equal(local, torch.ones(3)), cls.__name__)

    def test_lagrangian_uses_local_constraints(self):
        # Duals follow the reduced values; the surrogate must keep the local ones
        # so autograd still sees this rank's dependence on the parameters.
        from unittest.mock import patch

        reduced = torch.tensor([2.0, 4.0, 6.0])
        opt = nuPI(m=3, nu=0.0, ki=0.1, kp=0.0, penalty=0.0, process_group=object())
        local = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        with patch("torch.distributed.all_reduce", side_effect=lambda t, **kw: t.copy_(reduced)):
            lagrangian = opt.forward_update(LOSS, local)
        expected = LOSS + opt.duals @ local
        self.assertTrue(torch.allclose(lagrangian, expected))


if __name__ == "__main__":
    unittest.main()
