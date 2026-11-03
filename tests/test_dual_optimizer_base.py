"""
Tests for the shared DualOptimizer base class.

Covers the behaviour introduced with the base class -- constraint-input
normalization and validation, group names -- plus two defects
that the per-class implementations shared and that centralising fixed.
"""

import unittest

import torch

import copy

from humancompatible.train.dual_optim import ALM, PBM, SSG, MoreauEnvelope, iALM, nuPI
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


def _one_group(cls, **kw):
    """A single-group optimizer of `cls` with unit step sizes."""
    if cls is ALM:
        return ALM(m=1, lr=1.0, **kw)
    if cls is iALM:
        return iALM(m=1, beta=1.0, sigma=1.0, gamma=1e6, **kw)
    if cls is nuPI:
        return nuPI(m=1, nu=0.0, ki=1.0, kp=0.0, **kw)
    if cls is PBM:
        return PBM(m=1, gamma_annealing=False, penalty_annealing=False, **kw)
    return SSG(m=1, **kw)


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


class TestInequalityPenalty(unittest.TestCase):
    """The quadratic penalty must act on [c]+ for inequality groups.

    Penalising the raw value of an inequality constraint also penalises being
    strictly feasible, and is minimised by driving c to 0 -- it pulls feasible
    iterates back onto the boundary. Only the quadratic term is affected; the
    linear term and the dual update keep the raw values.
    """

    LOSS = torch.tensor(1.0)
    C = torch.tensor([-1.0, 3.0])  # one satisfied, one violated
    m = len(C)

    def test_equality_group_uses_raw_values(self):
        opt = ALM(m=2, lr=0.1, penalty=2.0, init_duals=0.5, is_ineq=False)
        got = opt.forward(self.LOSS, self.C)
        # loss + y.c + (rho/2)||c||^2 = 1 + 0.5*(-1+3) + 1.0*(1+9)
        self.assertAlmostEqual(got.item(), 1.0 + 1.0 + 10.0, places=6)

    def test_inequality_group_uses_violations_only(self):
        penalty = 2.0
        duals = 0.5
        opt = ALM(m=2, lr=0.1, penalty=penalty, init_duals=duals, is_ineq=True)
        got = opt.forward(self.LOSS, self.C)
        # the satisfied constraint contributes 0 to the quadratic, not 1
        # loss + (1/(2*rho))( || [y + rho*c]_+ ||^2 - ||y||^2 ) = 11.5
        duals = torch.ones(self.m)*duals
        constr_term = torch.relu(duals + penalty*self.C)
        self.assertAlmostEqual(
            got.item(),
            (
                self.LOSS + ( constr_term @ constr_term - duals @ duals ) / (2*penalty)
            ).item() ,
            places=6
        )

    def test_ialm_per_group_quadratic(self):
        eq = iALM(m=2, beta=2.0, sigma=1.0, gamma=1.0, init_duals=0.5, is_ineq=False)
        ineq = iALM(m=2, beta=2.0, sigma=1.0, gamma=1.0, init_duals=0.5, is_ineq=True)
        self.assertAlmostEqual(eq.forward(self.LOSS, self.C).item(), 1.0 + 1.0 + 10.0, places=6)
        self.assertAlmostEqual(ineq.forward(self.LOSS, self.C).item(), 1.0 + 1.0 + 9.0, places=6)

    def test_mixed_groups_are_treated_separately(self):
        opt = ALM(m=1, lr=0.1, penalty=2.0, init_duals=0.0, is_ineq=False)
        opt.add_constraint_group(m=1, lr=0.1, init_duals=0.0, is_ineq=True)
        # duals are 0, so only the quadratic contributes: 1.0*((-2)^2 + [-3]+^2)
        got = opt.forward(self.LOSS, torch.tensor([-2.0, -3.0]))
        self.assertAlmostEqual(got.item(), 1.0 + 4.0, places=6)

    def test_violated_inequality_does_get_penalty_gradient(self):
        opt = ALM(m=1, lr=0.1, penalty=2.0, init_duals=0.5, is_ineq=True)
        c = torch.tensor([3.0], requires_grad=True)
        opt.forward(self.LOSS, c).backward()
        # d/dc [ y*c + (rho/2)c^2 ] = 0.5 + 2*3
        self.assertAlmostEqual(c.grad.item(), 6.5, places=6)

    def test_equality_only_path_is_untouched(self):
        # The fast path must return the argument itself, so the all-equality
        # trajectories stay bit-exact.
        opt = ALM(m=3, lr=0.1, penalty=1.0, is_ineq=False)
        c = torch.tensor([1.0, -2.0, 3.0])
        self.assertIs(opt._penalty_constraints(c), c)

    def test_penalty_zero_short_circuits(self):
        for opt in (
            ALM(m=2, lr=0.1, penalty=0.0, init_duals=0.5, is_ineq=True),
            nuPI(m=2, nu=0.5, ki=0.1, kp=1.0, init_duals=0.5, is_ineq=True),
        ):
            got = opt.forward(self.LOSS, self.C)
            self.assertAlmostEqual(got.item(), 1.0 + 1.0, places=6, msg=type(opt).__name__)


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
        opt = nuPI(m=3, nu=0.0, ki=0.1, kp=0.0, process_group=object())
        local = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        with patch("torch.distributed.all_reduce", side_effect=lambda t, **kw: t.copy_(reduced)):
            lagrangian = opt.forward_update(LOSS, local)
        expected = LOSS + opt.duals @ local
        self.assertTrue(torch.allclose(lagrangian, expected))


class TestAddedGroupSettings(unittest.TestCase):
    """Groups added later inherit the optimizer's settings, per group."""

    def test_equality_group_after_inequality_group_can_go_negative(self):
        # The first group's clipped lower bound (0) must not leak into an
        # equality group added after it.
        for cls in (ALM, iALM, nuPI):
            opt = _one_group(cls, is_ineq=True)
            opt.add_constraint_group(m=1, is_ineq=False)
            opt.update(torch.tensor([-1.0, -1.0]))
            self.assertEqual(opt.duals[0].item(), 0.0, cls.__name__)
            self.assertLess(opt.duals[1].item(), 0.0, cls.__name__)

    def test_added_group_follows_optimizer_device(self):
        for cls in (ALM, iALM, nuPI, PBM, SSG):
            opt = _one_group(cls, device="meta")
            opt.add_constraint_group(m=1)
            for group in opt.param_groups:
                for t in group["params"]:
                    self.assertEqual(t.device.type, "meta", cls.__name__)

    def test_ialm_added_group_inherits_settings(self):
        opt = iALM(m=1, beta=0.05, sigma=1.5, gamma=2.0, momentum=0.9, is_ineq=True)
        opt.add_constraint_group(m=1)
        g = opt.param_groups[1]
        self.assertAlmostEqual(g["beta"].item(), 0.05, places=6)
        self.assertAlmostEqual(g["sigma"].item(), 1.5, places=6)
        self.assertAlmostEqual(g["gamma"].item(), 2.0, places=6)
        self.assertEqual(g["momentum"], 0.9)
        self.assertEqual(g["dampening"], 0.9)  # EMA, as for the first group
        self.assertTrue(g["is_ineq"])

    def test_nupi_added_group_inherits_is_ineq(self):
        opt = nuPI(m=1, is_ineq=True)
        opt.add_constraint_group(m=1)
        self.assertTrue(opt.param_groups[1]["is_ineq"])


class TestConstraintAndDualInputs(unittest.TestCase):
    def test_column_vector_constraints_match_flat(self):
        for cls in (ALM, iALM, nuPI, PBM):
            a, b = _two_group(cls), _two_group(cls)
            flat = torch.tensor([1.0, 2.0, 10.0, 20.0, 30.0])
            la = a.forward_update(LOSS, flat)
            lb = b.forward_update(LOSS, flat.reshape(-1, 1))
            self.assertTrue(torch.equal(la, lb), cls.__name__)
            self.assertTrue(torch.equal(a.duals, b.duals), cls.__name__)

    def test_init_duals_tensor_is_copied(self):
        init = torch.zeros(2)
        opt = ALM(m=2, lr=1.0, init_duals=init)
        opt.update(torch.ones(2))
        self.assertTrue(torch.equal(init, torch.zeros(2)))
        self.assertTrue(torch.equal(opt.duals, torch.ones(2)))

    def test_list_of_tensors_keeps_the_graph(self):
        w = torch.tensor([0.3, 0.2], requires_grad=True)
        for form in ({"a": [2 * w[0], 2 * w[1]]}, [[2 * w[0], 2 * w[1]]]):
            w.grad = None
            opt = ALM(m=2, lr=0.1, init_duals=1.0, name="a")
            opt.forward(torch.tensor(0.0), form).backward()
            self.assertTrue(torch.equal(w.grad, torch.tensor([2.0, 2.0])))

    def test_numpy_scalar_init_duals_keeps_default_dtype(self):
        import numpy as np

        opt = ALM(m=2, init_duals=np.float64(0.5))
        self.assertEqual(opt.duals.dtype, torch.get_default_dtype())

    def test_init_duals_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ALM(m=5, init_duals=torch.zeros(3))

    def test_init_duals_dtype_and_device_are_kept(self):
        opt = ALM(m=2, init_duals=torch.zeros(2, dtype=torch.float64))
        self.assertEqual(opt.duals.dtype, torch.float64)
        loss, c = torch.tensor(1.0, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
        opt.forward_update(loss, c)  # used to fail on a Float/Double mismatch
        opt = ALM(m=2, init_duals=torch.zeros(2), device="meta")
        self.assertEqual(opt.duals.device.type, "meta")


class TestGroupNames(unittest.TestCase):
    def test_first_group_can_be_named(self):
        for cls in (ALM, iALM, nuPI, PBM, SSG):
            opt = _one_group(cls, name="first")
            opt.add_constraint_group(m=2, name="second")
            self.assertEqual(opt.names, ["first", "second"], cls.__name__)
            opt.update({"second": torch.zeros(2), "first": torch.zeros(1)})

    def test_duals_by_group_returns_live_tensors(self):
        opt = ALM(m=1, lr=1.0, name="a")
        opt.add_constraint_group(m=2, name="b")
        opt.update(torch.tensor([1.0, 2.0, 3.0]))
        by_group = opt.duals_by_group
        self.assertEqual(list(by_group), ["a", "b"])
        self.assertTrue(torch.equal(by_group["b"], torch.tensor([2.0, 3.0])))
        by_group["a"].fill_(7.0)
        self.assertEqual(opt.duals[0].item(), 7.0)


class TestMoreauEnvelope(unittest.TestCase):
    def test_param_without_grad_is_skipped(self):
        a, b = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
        opt = MoreauEnvelope(torch.optim.SGD([a, b], lr=0.1))
        a.sum().backward()
        opt.step()  # used to fail on b.grad being None
        self.assertIsNone(b.grad)

    def test_group_added_after_wrapping_gets_prox_term(self):
        w, s = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
        opt = MoreauEnvelope(torch.optim.SGD([w], lr=1.0), mu=1.0, beta=0.0)
        opt.add_param_group({"params": [s]})
        (w + s).sum().backward()
        opt.step()  # creates the prox centres at 0, then moves both to -1
        for p in (w, s):
            p.grad = torch.zeros(1)
        opt.step()  # only the prox term is left: grad = mu * (-1 - 0)
        self.assertTrue(torch.equal(w.detach(), s.detach()))
        self.assertEqual(s.item(), 0.0)

    def test_deepcopy_does_not_double_inner_hooks(self):
        p = torch.nn.Parameter(torch.ones(1))
        sgd = torch.optim.SGD([p], lr=0.1)
        calls = []
        sgd.register_step_post_hook(lambda *a: calls.append(1))
        opt = MoreauEnvelope(sgd)
        copy.deepcopy(MoreauEnvelope(torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)))
        p.grad = torch.ones(1)
        opt.step()
        self.assertEqual(len(calls), 1)

    def test_deepcopy_can_step(self):
        p = torch.nn.Parameter(torch.ones(2))
        opt = copy.deepcopy(MoreauEnvelope(torch.optim.SGD([p], lr=0.1)))
        q = opt.param_groups[0]["params"][0]
        q.sum().backward()
        opt.step()
        self.assertTrue(torch.allclose(q.detach(), torch.full((2,), 0.9)))


if __name__ == "__main__":
    unittest.main()
