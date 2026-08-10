"""
Characterization tests: the dual optimizers must reproduce, bitwise, the
trajectories recorded in ``tests/golden/trajectories/``.

These guard the extraction of the shared :class:`DualOptimizer` base class. A
failure here means some algorithm's numerical behaviour changed -- the message
reports the first step at which it diverged, which is normally enough to
localise the cause.

Regenerate the goldens with ``python -m tests.golden.generate`` only when a
change in behaviour is intended.
"""

import unittest
from pathlib import Path

import torch

from tests.golden.scenarios import SCENARIOS, run_scenario

DATA_DIR = Path(__file__).parent / "golden" / "trajectories"


def _first_divergence(expected, actual):
    """First leading-axis index at which two recordings differ, or None.

    For per-step trajectories the axis is the step; for the ``initial_*`` entries
    (recorded once, before the loop) it is the element index.
    """
    for i in range(min(len(expected), len(actual))):
        if not torch.equal(expected[i], actual[i]):
            return i
    return None


class TestGoldenTrajectories(unittest.TestCase):
    """One test method per scenario, generated below."""

    def _check(self, name):
        path = DATA_DIR / f"{name}.pt"
        if not path.exists():
            self.skipTest(
                f"no golden for {name!r}; run `python -m tests.golden.generate {name}`"
            )

        payload = torch.load(path, weights_only=False)
        expected = payload["trajectory"]
        actual = run_scenario(name)

        if payload["torch_version"] != torch.__version__:
            self.skipTest(
                f"golden for {name!r} was recorded with torch "
                f"{payload['torch_version']}, running {torch.__version__}; "
                "bitwise comparison is not meaningful across versions"
            )

        self.assertEqual(
            sorted(expected), sorted(actual), f"{name}: recorded state keys changed"
        )

        for key in sorted(expected):
            exp, act = expected[key], actual[key]
            self.assertEqual(exp.shape, act.shape, f"{name}/{key}: shape changed")
            if torch.equal(exp, act):
                continue
            i = _first_divergence(exp, act)
            max_diff = (exp - act).abs().max().item()
            self.fail(
                f"{name}/{key}: diverges at index {i} "
                f"(max abs diff over recording {max_diff:.3e})\n"
                f"  expected[{i}] = {exp[i]}\n"
                f"  actual[{i}]   = {act[i]}"
            )


def _attach_tests():
    for name in sorted(SCENARIOS):
        def test(self, _name=name):
            self._check(_name)

        test.__name__ = f"test_{name}"
        test.__doc__ = f"golden trajectory: {name}"
        setattr(TestGoldenTrajectories, test.__name__, test)


_attach_tests()


if __name__ == "__main__":
    unittest.main()
