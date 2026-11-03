"""
Minimal tests for the Ghost-penalty SQP solver.

This subpackage had no tests at all. These cover only the construction path and
one step end to end -- enough to catch the class being unusable, which it was:
``GhostSQP(..., rng=None)`` raised AttributeError on a misspelled
``np.random.default_rng``. The MLMC unbiasedness and ghost-relaxation properties
deserve their own tests and are not covered here.
"""

import unittest

import numpy as np
import torch

try:
    import qpsolvers  # noqa: F401
    from humancompatible.train.sqp.ghost import (GhostConfig, GhostSQP, TensorSampler, compute_kappa, direction,
                                                 evaluate_oracle, n_params, solve_subproblem)

    HAVE_GHOST = True
except ImportError:  # pragma: no cover - optional [ghost] extra
    HAVE_GHOST = False


def _problem(n=6, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = torch.randn(n, 1, generator=g)
    net = torch.nn.Linear(d, 1)

    def objective_fn(model, batch):
        xb, yb = batch
        return torch.mean((model(xb) - yb) ** 2)

    def constraint_fn(model, batch):
        xb, _ = batch
        return model(xb).mean() - 0.1

    return net, objective_fn, [constraint_fn], TensorSampler(X, y)


@unittest.skipUnless(HAVE_GHOST, "requires the [ghost] extra (qpsolvers, scipy)")
class TestGhostSQPConstruction(unittest.TestCase):
    def test_default_rng_is_constructible(self):
        # Regression: this raised AttributeError from a misspelled default_rng.
        solver = GhostSQP(*_problem())
        self.assertIsInstance(solver.rng, np.random.Generator)

    def test_explicit_rng_is_kept(self):
        rng = np.random.default_rng(1234)
        solver = GhostSQP(*_problem(), rng=rng)
        self.assertIs(solver.rng, rng)

    def test_default_config(self):
        solver = GhostSQP(*_problem())
        self.assertIsInstance(solver.cfg, GhostConfig)
        # beta is None by default (solve_relaxed=True); __post_init__ fills it otherwise
        cfg = GhostConfig(solve_relaxed=False)
        self.assertGreater(cfg.beta, cfg.rho)  # trust region > ghost radius


@unittest.skipUnless(HAVE_GHOST, "requires the [ghost] extra (qpsolvers, scipy)")
class TestGhostSQPStep(unittest.TestCase):
    def test_step_moves_the_parameters_and_reports_diagnostics(self):
        net, obj, cons, sampler = _problem()
        solver = GhostSQP(net, obj, cons, sampler, rng=np.random.default_rng(0))
        before = torch.cat([p.detach().reshape(-1).clone() for p in net.parameters()])

        info = solver.step()

        after = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
        self.assertFalse(torch.equal(before, after))
        for key in ("N", "p_N", "kappa", "constraint_values", "gamma", "step_norm"):
            self.assertIn(key, info)
        self.assertTrue(np.isfinite(info["step_norm"]))

    def test_gamma_is_diminishing(self):
        solver = GhostSQP(*_problem(), rng=np.random.default_rng(0))
        self.assertGreater(solver.gamma(1), solver.gamma(10))

    def test_train_returns_one_record_per_iteration(self):
        solver = GhostSQP(*_problem(), rng=np.random.default_rng(0))
        history = solver.train(3)
        self.assertEqual(len(history), 3)


@unittest.skipUnless(HAVE_GHOST, "requires the [ghost] extra (qpsolvers, scipy)")
class TestVectorConstraint(unittest.TestCase):
    def test_vector_form_matches_list_form(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
        batch = (torch.randn(8, 3), torch.randn(8, 1))
        weights = torch.randn(5, 8)

        def objective_fn(model, b):
            return torch.mean((model(b[0]) - b[1]) ** 2)

        def vector_fn(model, b):
            return weights @ model(b[0]).reshape(-1)

        scalar_fns = [lambda model, b, i=i: vector_fn(model, b)[i] for i in range(5)]
        for got, want in zip(evaluate_oracle(net, objective_fn, vector_fn, batch),
                             evaluate_oracle(net, objective_fn, scalar_fns, batch)):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


@unittest.skipUnless(HAVE_GHOST, "requires the [ghost] extra (qpsolvers, scipy)")
class TestFrozenParams(unittest.TestCase):
    def test_frozen_layer_is_excluded(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
        net[0].requires_grad_(False)
        _, obj, cons, sampler = _problem()
        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

        grad_f, _, cgrad = evaluate_oracle(net, obj, cons, sampler.draw(4, np.random.default_rng(0)))
        self.assertEqual(cgrad.shape[1], grad_f.shape[0])
        self.assertEqual(grad_f.shape[0], n_params(net))
        self.assertEqual(n_params(net), n_trainable)

        frozen = net[0].weight.detach().clone()
        GhostSQP(net, obj, cons, sampler, rng=np.random.default_rng(0)).step()
        self.assertTrue(torch.equal(net[0].weight, frozen))


@unittest.skipUnless(HAVE_GHOST, "requires the [ghost] extra (qpsolvers, scipy)")
class TestDirection(unittest.TestCase):
    def test_flags_come_from_cfg(self):
        net, obj, cons, sampler = _problem()
        d, _ = direction(net, sampler, GhostConfig(), np.random.default_rng(0), objective_fn=obj, constraint_fns=cons)
        self.assertEqual(len(d), n_params(net))


@unittest.skipUnless(HAVE_GHOST, "requires the [ghost] extra (qpsolvers, scipy)")
class TestSubproblems(unittest.TestCase):
    def test_infeasible_qp_raises(self):
        # d0 <= -1 and -d0 <= -1: empty feasible set (dual is unbounded)
        cgrad = np.array([[1.0, 0.0], [-1.0, 0.0]])
        cval = np.array([1.0, 1.0])
        for solve_dual in (False, True):
            for solve_relaxed in (False, True):
                with self.subTest(solve_dual=solve_dual, solve_relaxed=solve_relaxed):
                    with self.assertRaises(RuntimeError):
                        solve_subproblem(np.zeros(2), cval, cgrad, 0.0, None if solve_relaxed else 10.0, 1.0,
                                         solve_dual=solve_dual, solve_relaxed=solve_relaxed)

    def test_relaxed_primal_matches_dual(self):
        rng = np.random.default_rng(0)
        cgrad, grad_f = rng.standard_normal((3, 8)), rng.standard_normal(8)
        cval = np.array([0.5, -0.3, 0.2])
        kappa = compute_kappa(cval, cgrad, 0.5, 0.8)
        d_primal = solve_subproblem(grad_f, cval, cgrad, kappa, None, 1.0, solve_dual=False, solve_relaxed=True)
        d_dual = solve_subproblem(grad_f, cval, cgrad, kappa, None, 1.0, solve_dual=True, solve_relaxed=True)
        # OSQP's default eps_abs=1e-3 bounds the attainable agreement
        np.testing.assert_allclose(d_primal, d_dual, atol=1e-2)

    def test_kappa_is_zero_when_feasible(self):
        cgrad = np.random.default_rng(0).standard_normal((3, 8))
        self.assertEqual(compute_kappa(np.array([-1.0, 0.0, -0.5]), cgrad, 0.5, 0.8), 0.0)


if __name__ == "__main__":
    unittest.main()
