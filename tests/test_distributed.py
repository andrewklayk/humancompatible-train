"""
Data-parallel semantics of the dual optimizers, on the gloo CPU backend.

The substance lives in ``paper/e0/d_distributed.py``, which is also the paper's
E0d experiment: duals identical on every rank, ``G x B`` equivalent to
``1 x (G*B)`` for a surrogate linear in the constraint vector, and the documented
inexactness elsewhere. Re-implementing those checks here would mean maintaining
two versions of the same argument, so the test shells out to the experiment in
smoke mode and asserts its ``--check`` exit status.

Artifacts are redirected to a temporary directory via ``HC_PAPER_RESULTS`` so
running the suite cannot overwrite committed results.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "paper" / "e0" / "d_distributed.py"


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),
                     "torch.distributed with the gloo backend is required")
@unittest.skipUnless(SCRIPT.exists(), "paper/e0/d_distributed.py is not present")
class TestDataParallelEquivalence(unittest.TestCase):
    """Runs E0d on two ranks and requires every registered prediction to hold."""

    def test_two_ranks_smoke(self):
        with tempfile.TemporaryDirectory() as results:
            environment = dict(
                os.environ,
                HC_PAPER_RESULTS=results,
                OMP_NUM_THREADS="1",
                # Keep the rendezvous off any port a real job might be using.
                MASTER_PORT="29517",
            )
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--ranks", "2", "--quick", "--check"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=900,
            )
            self.assertEqual(
                completed.returncode, 0,
                msg=f"E0d reported a failed prediction.\n"
                    f"--- stdout ---\n{completed.stdout}\n"
                    f"--- stderr ---\n{completed.stderr}",
            )
            # Guard against a vacuous pass: the script must have got as far as
            # registering and reporting its predictions.
            self.assertIn("Predictions:", completed.stdout)
            self.assertIn("D1:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
