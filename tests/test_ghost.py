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
    from humancompatible.train.sqp.ghost import GhostConfig, GhostSQP, TensorSampler

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
        self.assertGreater(solver.cfg.beta, solver.cfg.rho)  # trust region > ghost radius


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


if __name__ == "__main__":
    unittest.main()
