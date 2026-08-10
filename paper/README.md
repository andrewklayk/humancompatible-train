# Paper experiments

Experiments for the MPC software paper on `humancompatible-train`. Deliberately
plain scripts rather than a framework: each one is meant to be read, edited and
debugged directly.

Two conventions run through all of them.

**Every experiment states its predictions in code, before its results.** Each
script's module docstring lists numbered predictions; `register_predictions`
evaluates them and `--check` makes the process exit non-zero if one fails. A
prediction that turns out to be wrong is reported as wrong rather than quietly
edited — where that happened, the docstring says so and both the original and the
corrected claim are kept.

**Artifacts are committed.** Figures are PDF (the repo's `.gitignore` excludes
`*.png`/`*.svg`, and LaTeX wants PDF anyway), tables are Markdown plus a JSON
sidecar, raw per-iteration data is CSV. `paper/results/**` is explicitly
re-included in `.gitignore` so all of it can go into the reproducibility artifact.

## Running

```bash
bash paper/e0/run_all.sh                 # everything, writes paper/results/e0*/
bash paper/e0/run_all.sh --quick         # seconds; artifacts to a temp dir
bash paper/e0/run_all.sh --check         # non-zero exit on a violated prediction
python paper/e0/a_multipliers.py --help  # per-script options
```

`HC_PAPER_RESULTS=<dir>` redirects all artifacts, which is how `--quick` and
`tests/test_distributed.py` avoid overwriting a real run's results.

Requires `qpsolvers` (with `clarabel`), `scikit-learn`, `matplotlib` and `scipy`
on top of the package's own dependencies.

## E0 — validation

E0 is the faithfulness section: before any expensive experiment, show that each
shipped implementation does what its reference says. The algorithms are other
people's; the implementation is ours, so this is the part that earns trust in it.

| script | claim | runtime |
|---|---|---|
| [e0/a_multipliers.py](e0/a_multipliers.py) | the dual variables converge to the true Lagrange multipliers | ~20 min |
| [e0/b_nonopt.py](e0/b_nonopt.py) | `NonOpt` reaches the published optimal values on the ten standard nonsmooth problems at n = 50 | ~10 min |
| [e0/d_distributed.py](e0/d_distributed.py) | data-parallel equivalence: duals identical across ranks, `G x B` = `1 x (G*B)` where that can hold, and the measured gap where it cannot | ~1 min |

### E0a — multiplier recovery

Four problems, chosen so the ways a method can fail are separated: a convex QP
with every constraint active (all `y*_i > 0`), a convex QP where most multipliers
are *exactly zero*, a hard-margin SVM where 96 of 100 are zero, and an indefinite
QP with no reference multipliers at all.

Two protocol choices matter for reading the results:

- The primal step is **derived**, not tuned: `1/(L_f + rho*||J||^2)`. Every method
  gets the largest step that is safe for its own surrogate, which also makes
  visible that a method carrying a large quadratic term pays for it with a smaller
  primal step.
- Each method's **dual** step is swept over a grid and the best final
  `||y - y*||_inf` is reported, so no result is an artifact of a step size shared
  across methods. `iALM` is the exception, since its `beta` is simultaneously the
  quadratic coefficient and the dual step; it is reported as a labelled sweep.

Reference values are exact, not solver output: the QPs are constructed backwards
from a chosen `(x*, y*)` and cross-checked against clarabel; the SVM uses clarabel
only to identify the active set and then solves the resulting KKT system exactly.
Without that, the solver's own ~1e-9 error becomes a floor that several methods
tie at for reasons that say nothing about them.

### E0b — NonOpt against published values

The ten Karmitsa problems at n = 50. Formulae come from Karmitsa's report and the
targets from the published table, so **transcription is verified rather than
assumed**: `paper/problems/nonsmooth.py::verify` checks the implemented `f(x0)`
against a hand-derived closed form to 1e-10 *and* the closed form against the
published value to the table's printed precision. A mistranscribed formula fails
at the start of the run instead of looking like a solver failure.

Two of our defaults differ from the reference configuration and are swept rather
than hidden: `inverse_hessian` (`dense` matches the reference's full-memory BFGS;
our default is `limited_memory`) and `point_set_options["size_factor"]` (the
shipped 0.05 keeps only **two** bundle points at n = 50, and zero below n = 20).

Three interface facts are worked around and reported, since users will hit them:
there are no evaluation counters (the closure is wrapped, and one call yields both
value and gradient, so `#func` and `#grad` cannot be separated); `step()` returns
the loss only on the first call; and the stationarity test scales with the initial
gradient, so on `maxq` the effective tolerance is 1e-2 rather than the requested
1e-4. Both tolerances are in the results table.

The direct comparison against the C++ NonOpt is deferred. The reference solvers'
own final values (Curtis & Que Tables 2-4) are **not yet transcribed** — see
`REFERENCE_SOLVERS_N50` in [problems/nonsmooth.py](problems/nonsmooth.py); fill it
in and the summary table grows those columns automatically. Until then the
comparison is against the published `f*` only.

### E0d — data-parallel equivalence

Two gloo ranks against one process at the pooled batch size. The headline is that
equivalence is **conditional**, and the conditions are worth stating in the paper:

- Duals are bitwise identical on every rank, always.
- A surrogate **linear in the constraint vector** (`ALM(rho=0)`, `nuPI(rho=0)`)
  reproduces the pooled-batch run exactly, because `ReduceOp.AVG` of per-rank
  means is the mean over the union and DDP averages gradients.
- A surrogate with a **quadratic term** does not, since
  `mean_r ||[c_r]_+||^2 != ||[mean_r c_r]_+||^2`. After one step only the
  parameters differ — the dual update sees only the reduced vector — and over more
  steps that discrepancy propagates into the duals through the constraint values.
- A **ratio-type** constraint (the per-group rates E2 and E3 use) is a nonlinear
  functional of expectations, so nothing matches exactly regardless of the
  surrogate. The gap is measured against per-rank batch size, which is what bounds
  how small a per-rank batch those experiments may use.

## Layout

```
paper/
  README.md                 this file
  _harness.py               seeding, float64, CSV/JSON/Markdown writers, PDF figures, Checks
  problems/
    __init__.py             the Problem dataclass, including the derived primal step size
    qp.py                   convex and nonconvex QPs with exact (x*, y*)
    svm.py                  hard-margin SVM on separable Iris, exact active-set reference
    nonsmooth.py            the ten Karmitsa problems + transcription verification
  e0/
    a_multipliers.py  b_nonopt.py  d_distributed.py  run_all.sh
  results/e0a/ e0b/ e0d/    *.csv (raw), *.md + *.json (tables), *.pdf (figures)
```

Figure style comes from [benchmark/new_bench/plotting/plot_style.py](../benchmark/new_bench/plotting/plot_style.py),
loaded by file path because that tree is not an importable package. It is the
repo's declared single style source, and duplicating its rcParams here would let
the two drift.

## Not here yet

E1 (noisy CUTEst via S2MPJ), E2 (fairness-constrained learning on folktables), E3
(sparsity-constrained LLM fine-tuning on multiple GPUs) and E4 (nonsmooth
reformulations). `problems/` is shared with them by design.
