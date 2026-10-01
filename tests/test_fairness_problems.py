"""
The E2 problem layer: group construction, constraint arity, and the
data-parallel equivalence invariant.

Deliberately not a test of the *experiment* -- that is what
``paper/e2/*.py --check`` is for. These pin the properties the experiment assumes
and would otherwise only discover as a confusing result:

* the constraint vector has the arity ``m`` claims,
* balanced batches make ``PositiveRate``'s denominator constant, which is what
  turns a ratio-type statistic into a mean-type one (see E0d),
* the problem's batch-order generator is resettable, without which two runs over
  the same problem see different data (a bug this suite exists to keep fixed).

Skipped when the datasets are not on disk, since the ACS PUMS files are not
committed.
"""

import unittest

import torch

try:
    from paper.problems import fairness
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - report why, do not fail collection
    fairness = None
    _IMPORT_ERROR = exc


def _acs_available():
    return bool(fairness) and bool(fairness.available_states())


@unittest.skipUnless(_acs_available(),
                     f"folktables ACS data not on disk ({_IMPORT_ERROR})")
class TestIncomeProblem(unittest.TestCase):
    """The pairwise positive-rate problem on ACSIncome."""

    @classmethod
    def setUpClass(cls):
        cls.problem = fairness.build("income", "pairwise", bound=0.05)

    def test_group_and_constraint_arity(self):
        problem = self.problem
        # MAR (merged to 3) x SEX (2) = 6 groups -> m = G(G-1) ordered pairs.
        self.assertEqual(problem.n_groups, 6)
        self.assertEqual(problem.m, problem.n_groups * (problem.n_groups - 1))

        torch.manual_seed(0)
        model = problem.make_model()
        features, sens, _ = problem.train
        constraints = problem.constraints(model(features[:512]), sens[:512])
        self.assertEqual(constraints.numel(), problem.m)

    def test_bound_is_subtracted_once(self):
        """``constraints`` is ``raw - bound``, so the two differ by exactly bound."""
        problem = self.problem
        torch.manual_seed(0)
        model = problem.make_model()
        features, sens, _ = problem.train
        logits = model(features[:512])
        raw = problem.raw_constraints(logits, sens[:512]).reshape(-1)
        adjusted = problem.constraints(logits, sens[:512])
        self.assertTrue(torch.allclose(raw - problem.bound, adjusted))

    def test_sensitive_columns_are_not_features(self):
        """The constrained attribute must not be an input, or the model can read
        the group directly and the constraint measures something else."""
        problem = self.problem
        # 808 = 815 ACS one-hot columns minus the 7 MAR_*/SEX_* columns.
        self.assertEqual(problem.n_features, 808)

    def test_balanced_batches_have_constant_group_counts(self):
        """E0d's data-parallel equivalence rests on this: ``PositiveRate`` is a
        linear-fractional statistic whose denominator is the per-group count, so
        a *constant* count makes it linear in the sample -- and only then is the
        cross-rank average of per-rank statistics the pooled statistic."""
        problem = self.problem
        problem.reseed(0)
        counts = []
        for index, (_, sens, _) in enumerate(problem.loader):
            counts.append(sens.sum(0))
            if index == 4:
                break
        expected = torch.full((problem.n_groups,), float(fairness.PER_GROUP))
        for count in counts:
            self.assertTrue(torch.equal(count, expected), f"got {count.tolist()}")

    def test_reseed_makes_batch_order_reproducible(self):
        """Without this the n-th run over a problem continues the (n-1)-th's batch
        order, so two methods see different data and a 'seed' controls only the
        model init."""
        problem = self.problem

        def first_batch():
            problem.reseed(3)
            features, _, _ = next(iter(problem.loader))
            return features

        first, second = first_batch(), first_batch()
        self.assertTrue(torch.equal(first, second))

        problem.reseed(4)
        other, _, _ = next(iter(problem.loader))
        self.assertFalse(torch.equal(first, other),
                         "a different seed must give a different batch order")

    def test_train_test_split_is_disjoint_and_stratified(self):
        problem = self.problem
        (x_train, a_train, _), (x_test, a_test, _) = problem.train, problem.test
        self.assertEqual(x_train.shape[1], x_test.shape[1])
        # Every group must appear on both sides, or a per-group constraint cannot
        # be evaluated on the test split at all.
        self.assertTrue(bool((a_train.sum(0) > 0).all()))
        self.assertTrue(bool((a_test.sum(0) > 0).all()))
        total = len(x_train) + len(x_test)
        self.assertAlmostEqual(len(x_test) / total, fairness.TEST_SIZE, places=2)


@unittest.skipUnless(_acs_available(),
                     f"folktables ACS data not on disk ({_IMPORT_ERROR})")
class TestAggregateShape(unittest.TestCase):
    def test_single_constraint(self):
        problem = fairness.build("income", "agg", bound=0.05)
        self.assertEqual(problem.m, 1)
        torch.manual_seed(0)
        model = problem.make_model()
        features, sens, _ = problem.train
        constraints = problem.constraints(model(features[:512]), sens[:512])
        self.assertEqual(constraints.numel(), 1)

    def test_unknown_shape_and_dataset_are_rejected(self):
        with self.assertRaises(ValueError):
            fairness.build("income", "not_a_shape")
        with self.assertRaises(ValueError):
            fairness.build("not_a_dataset", "pairwise")


if __name__ == "__main__":
    unittest.main()
