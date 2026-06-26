import math
import unittest

import torch
from humancompatible.train.sqp import NonOpt
from humancompatible.train.sqp.nonopt import (
    LimitedMemoryInverseHessian,
    project_onto_simplex,
    solve_simplex_qp,
)


def run_nonopt(x0, objective, max_iterations=500, **options):
    """Runs NonOpt on a tensor objective and returns (optimizer, x, f_final)."""
    x = torch.nn.Parameter(x0.clone())
    optimizer = NonOpt([x], **options)

    def closure():
        optimizer.zero_grad()
        loss = objective(x)
        loss.backward()
        return loss

    for _ in range(max_iterations):
        optimizer.step(closure)
        if optimizer.converged:
            break
    return optimizer, x.detach(), float(objective(x))


def maxq(x):
    """MaxQ: f(x) = max_i x_i^2, nonsmooth convex, f* = 0."""
    return (x**2).max()


def maxq_x0(n):
    x0 = torch.arange(1.0, n + 1.0)
    x0[n // 2 :] *= -1.0
    return x0


def chained_lq(x):
    """Chained LQ, nonsmooth convex, f* = -(n-1) * sqrt(2)."""
    a = -x[:-1] - x[1:]
    b = a + (x[:-1] ** 2 + x[1:] ** 2 - 1.0)
    return torch.maximum(a, b).sum()


class TestSimplexQP(unittest.TestCase):
    """Test the simplex projection and the QP subproblem solver."""

    def test_projection_already_feasible(self):
        v = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(project_onto_simplex(v), v))

    def test_projection_sums_to_one_and_nonnegative(self):
        torch.manual_seed(0)
        for _ in range(10):
            v = torch.randn(7, dtype=torch.float64) * 5
            p = project_onto_simplex(v)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=10)
            self.assertTrue((p >= 0).all())

    def test_projection_single_dominant_coordinate(self):
        v = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        p = project_onto_simplex(v)
        self.assertTrue(torch.allclose(p, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)))

    def test_qp_identity_quadratic(self):
        # min ½||ω||² over simplex -> uniform
        Q = torch.eye(4, dtype=torch.float64)
        b = torch.zeros(4, dtype=torch.float64)
        omega = solve_simplex_qp(Q, b)
        self.assertTrue(torch.allclose(omega, torch.full((4,), 0.25, dtype=torch.float64), atol=1e-6))

    def test_qp_single_element(self):
        Q = torch.tensor([[2.0]], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        omega = solve_simplex_qp(Q, b)
        self.assertTrue(torch.allclose(omega, torch.ones(1, dtype=torch.float64)))

    def test_qp_known_solution(self):
        # min ½ ωᵀ diag(1, 4) ω over the simplex: ω = (4/5, 1/5)
        Q = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
        b = torch.zeros(2, dtype=torch.float64)
        omega = solve_simplex_qp(Q, b)
        self.assertTrue(
            torch.allclose(omega, torch.tensor([0.8, 0.2], dtype=torch.float64), atol=1e-6)
        )

    def test_qp_linear_only(self):
        # negligible quadratic: solution at the vertex maximizing b
        Q = torch.zeros(3, 3, dtype=torch.float64)
        b = torch.tensor([-1.0, 3.0, 0.5], dtype=torch.float64)
        omega = solve_simplex_qp(Q, b)
        self.assertEqual(int(torch.argmax(omega)), 1)
        self.assertAlmostEqual(float(omega.sum()), 1.0, places=10)


class TestLimitedMemoryInverseHessian(unittest.TestCase):
    """Test the self-correcting L-BFGS inverse Hessian approximation."""

    def test_identity_before_updates(self):
        W = LimitedMemoryInverseHessian()
        v = torch.randn(5)
        self.assertTrue(torch.allclose(W.apply(v), v))

    def test_secant_equation(self):
        # after an update with curvature pair (s, y), W y = s must hold
        torch.manual_seed(1)
        W = LimitedMemoryInverseHessian()
        s = torch.randn(6, dtype=torch.float64)
        y = s + 0.1 * torch.randn(6, dtype=torch.float64)
        if float(s.dot(y)) <= 0:
            y = s.clone()
        self.assertTrue(W.update(s, y))
        self.assertTrue(torch.allclose(W.apply(y), s, atol=1e-10))

    def test_update_skipped_on_tiny_displacement(self):
        W = LimitedMemoryInverseHessian()
        s = torch.full((4,), 1e-12, dtype=torch.float64)
        y = torch.full((4,), 1e-12, dtype=torch.float64)
        self.assertFalse(W.update(s, y))

    def test_self_correction_keeps_curvature_positive(self):
        # negative curvature pair must be corrected, not produce a singular update
        W = LimitedMemoryInverseHessian()
        s = torch.tensor([1.0, 0.0], dtype=torch.float64)
        y = torch.tensor([-1.0, 0.5], dtype=torch.float64)  # s·y < 0
        self.assertTrue(W.update(s, y))
        v = torch.randn(2, dtype=torch.float64)
        Wv = W.apply(v)
        # W must remain positive definite
        self.assertGreater(float(v.dot(Wv)), 0.0)


class TestNonOptOnNonsmoothProblems(unittest.TestCase):
    """Test convergence on classic nonsmooth test problems from NonOpt."""

    def test_maxq_cutting_plane(self):
        torch.manual_seed(0)
        optimizer, x, f = run_nonopt(maxq_x0(10), maxq)
        self.assertLess(f, 1e-04)

    def test_maxq_gradient_combination(self):
        torch.manual_seed(0)
        optimizer, x, f = run_nonopt(
            maxq_x0(10), maxq, direction="gradient_combination"
        )
        self.assertLess(f, 1e-04)

    def test_maxq_gradient(self):
        torch.manual_seed(0)
        optimizer, x, f = run_nonopt(maxq_x0(10), maxq, direction="gradient")
        self.assertLess(f, 1e-02)

    def test_maxq_backtracking(self):
        torch.manual_seed(0)
        optimizer, x, f = run_nonopt(maxq_x0(10), maxq, line_search="backtracking")
        self.assertLess(f, 1e-04)

    def test_maxq_dense_bfgs(self):
        torch.manual_seed(0)
        optimizer, x, f = run_nonopt(maxq_x0(10), maxq, inverse_hessian="dense")
        self.assertLess(f, 1e-04)

    def test_maxq_dense_dfp(self):
        # DFP is less robust than BFGS on nonsmooth problems (hence NonOpt
        # defaults to BFGS); it converges more slowly, so use a looser bar.
        torch.manual_seed(0)
        optimizer, x, f = run_nonopt(
            maxq_x0(10),
            maxq,
            inverse_hessian="dense",
            inverse_hessian_options={"formula": "dfp"},
        )
        self.assertLess(f, 1e-03)

    def test_chained_lq(self):
        torch.manual_seed(0)
        n = 10
        optimizer, x, f = run_nonopt(-0.5 * torch.ones(n), chained_lq)
        f_star = -(n - 1) * math.sqrt(2.0)
        self.assertLess(f - f_star, 1e-03 * abs(f_star))

    def test_l1_regression(self):
        # piecewise-linear convex: f(w) = ||A w - b||_1 with known minimizer
        torch.manual_seed(2)
        A = torch.randn(30, 5)
        w_star = torch.randn(5)
        b = A @ w_star

        optimizer, w, f = run_nonopt(torch.zeros(5), lambda w: (A @ w - b).abs().sum())
        self.assertLess(f, 1e-03)
        self.assertTrue(torch.allclose(w, w_star, atol=1e-03))

    def test_converged_flag_and_status(self):
        torch.manual_seed(0)
        optimizer, x, f = run_nonopt(maxq_x0(10), maxq, max_iterations=1000)
        self.assertTrue(optimizer.converged)
        self.assertIn(optimizer.status, ("stationary", "objective_similarity"))

    def test_smooth_problem(self):
        # smooth strongly convex sanity check
        optimizer, x, f = run_nonopt(
            torch.tensor([3.0, -4.0]), lambda x: ((x - 1.0) ** 2).sum()
        )
        self.assertTrue(torch.allclose(x, torch.ones(2), atol=1e-03))


class TestNonOptInterface(unittest.TestCase):
    """Test torch.optim.Optimizer interface compliance."""

    def test_works_with_nn_module(self):
        torch.manual_seed(3)
        model = torch.nn.Linear(4, 1)
        A = torch.randn(20, 4)
        b = torch.randn(20, 1)
        optimizer = NonOpt(model.parameters())

        def closure():
            optimizer.zero_grad()
            loss = (model(A) - b).abs().mean()
            loss.backward()
            return loss

        initial = float(closure())
        for _ in range(100):
            optimizer.step(closure)
            if optimizer.converged:
                break
        final = float(closure())
        self.assertLess(final, initial)

    def test_rejects_multiple_param_groups(self):
        p1 = torch.nn.Parameter(torch.zeros(2))
        p2 = torch.nn.Parameter(torch.zeros(2))
        with self.assertRaises(ValueError):
            NonOpt([{"params": [p1]}, {"params": [p2], "history_size": 5}])

    def test_rejects_unknown_strategies(self):
        p = torch.nn.Parameter(torch.zeros(2))
        with self.assertRaises(ValueError):
            NonOpt([p], direction="unknown")
        with self.assertRaises(ValueError):
            NonOpt([p], line_search="unknown")
        with self.assertRaises(ValueError):
            NonOpt([p], inverse_hessian="unknown")

    def test_step_returns_initial_loss(self):
        p = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = NonOpt([p])

        def closure():
            optimizer.zero_grad()
            loss = (p**2).sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertAlmostEqual(float(loss.detach().item()), 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
